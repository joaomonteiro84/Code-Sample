import pickle
import random
import os
from typing import List, Optional, Tuple, Dict

import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib import ticker

import torch
from torch.nn import Module, Linear, Softmax, ReLU, LayerNorm, ModuleList, Dropout, Embedding, CrossEntropyLoss
from torch.optim import Adam

class PositionalEncodingLayer(Module):

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X has shape (batch_size, sequence_length, embedding_dim)

        This function should create the positional encoding matrix
        and return the sum of X and the encoding matrix.

        The positional encoding matrix is defined as follow:

        P_(pos, 2i) = sin(pos / (10000 ^ (2i / d)))
        P_(pos, 2i + 1) = cos(pos / (10000 ^ (2i / d)))

        The output will have shape (batch_size, sequence_length, embedding_dim)
        """
        shape_output = tuple(X.shape)

        pos = np.array([range(shape_output[1])]*(shape_output[2]//2), dtype='float32').T
        denominator = 10000**(2*np.array([range(shape_output[2]//2)]*shape_output[1], dtype='float32')/shape_output[2])

        sin_term = np.sin(pos/denominator)
        cos_term = np.cos(pos/denominator)

        pos_enc_matrix = np.empty(shape=(sin_term.shape[0], 2*sin_term.shape[1]), dtype=sin_term.dtype)
        pos_enc_matrix[:,0::2] = sin_term
        pos_enc_matrix[:,1::2] = cos_term

        output = X + torch.from_numpy(pos_enc_matrix)

        return output

class SelfAttentionLayer(Module):

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()

        self.linear_Q = Linear(in_dim, out_dim)
        self.linear_K = Linear(in_dim, out_dim)
        self.linear_V = Linear(in_dim, out_dim)

        self.softmax = Softmax(-1)

        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, query_X: torch.Tensor, key_X: torch.Tensor, value_X: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        query_X, key_X and value_X have shape (batch_size, sequence_length, in_dim). The sequence length
        may be different for query_X and key_X but must be the same for key_X and value_X.

        This function should return two things:
            - The output of the self-attention, which will have shape (batch_size, sequence_length, out_dim)
            - The attention weights, which will have shape (batch_size, query_sequence_length, key_sequence_length)

        If a mask is passed as input, you should mask the input to the softmax, using `float(-1e32)` instead of -infinity.
        The mask will be a tensor with 1's and 0's, where 0's represent entries that should be masked (set to -1e32).

        Hint: The following functions may be useful
            - torch.bmm (https://pytorch.org/docs/stable/generated/torch.bmm.html)
            - torch.Tensor.masked_fill (https://pytorch.org/docs/stable/generated/torch.Tensor.masked_fill.html)
        """

        Q_tild = self.linear_Q(query_X)
        K_tild = self.linear_K(key_X)
        V_tild = self.linear_V(value_X)

        if mask is None:
            attention_weights = self.softmax(torch.bmm(Q_tild, K_tild.transpose(1,2))/np.sqrt(self.out_dim))
        else:
            attention_weights = self.softmax((torch.bmm(Q_tild, K_tild.transpose(1,2))/np.sqrt(self.out_dim)).masked_fill(mask == 0, float(-1e32)))

        output = torch.bmm(attention_weights, V_tild)

        return output, attention_weights


class MultiHeadedAttentionLayer(Module):

    def __init__(self, in_dim: int, out_dim: int, n_heads: int) -> None:
        super().__init__()

        self.attention_heads = ModuleList([SelfAttentionLayer(in_dim, out_dim) for _ in range(n_heads)])

        self.linear = Linear(n_heads * out_dim, out_dim)

    def forward(self, query_X: torch.Tensor, key_X: torch.Tensor, value_X: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        This function calls the self-attention layer and returns the output of the multi-headed attention
        and the attention weights of each attention head.

        The attention_weights matrix has dimensions (batch_size, heads, query_sequence_length, key_sequence_length)
        """

        outputs, attention_weights = [], []

        for attention_head in self.attention_heads:
            out, attention = attention_head(query_X, key_X, value_X, mask)
            outputs.append(out)
            attention_weights.append(attention)

        outputs = torch.cat(outputs, dim=-1)
        attention_weights = torch.stack(attention_weights, dim=1)

        return self.linear(outputs), attention_weights

class EncoderBlock(Module):

    def __init__(self, embedding_dim: int, n_heads: int) -> None:
        super().__init__()

        self.attention = MultiHeadedAttentionLayer(embedding_dim, embedding_dim, n_heads)

        self.norm1 = LayerNorm(embedding_dim)
        self.norm2 = LayerNorm(embedding_dim)

        self.linear1 = Linear(embedding_dim, 4 * embedding_dim)
        self.linear2 = Linear(4 * embedding_dim, embedding_dim)
        self.relu = ReLU()

        self.dropout1 = Dropout(0.2)
        self.dropout2 = Dropout(0.2)

    def forward(self, X, mask=None):
        """
        Implementation of an encoder block. Both the input and output
        have shape (batch_size, source_sequence_length, embedding_dim).

        The mask is passed to the multi-headed self-attention layer,
        and is usually used for the padding in the encoder.
        """
        att_out, _ = self.attention(X, X, X, mask)

        residual = X + self.dropout1(att_out)

        X = self.norm1(residual)

        temp = self.linear1(X)
        temp = self.relu(temp)
        temp = self.linear2(temp)

        residual = X + self.dropout2(temp)

        return self.norm2(residual)

class Encoder(Module):

    def __init__(self, vocab_size: int, embedding_dim: int, n_blocks: int, n_heads: int) -> None:
        super().__init__()

        self.embedding_layer = Embedding(vocab_size + 1, embedding_dim, padding_idx=vocab_size)
        self.position_encoding = PositionalEncodingLayer(embedding_dim)
        self.blocks = ModuleList([EncoderBlock(embedding_dim, n_heads) for _ in range(n_blocks)])
        self.vocab_size = vocab_size

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Transformer encoder. The input has dimensions (batch_size, sequence_length)
        and the output has dimensions (batch_size, sequence_length, embedding_dim).

        The encoder returns its output and the location of the padding, which will be
        used by the decoder.
        """
        padding_locations = torch.ones(X.shape[0], X.shape[1])
        padding_locations = padding_locations.masked_fill(X == self.vocab_size, 0)

        padding_mask = torch.ones(X.shape[0], X.shape[1], X.shape[1])
        for b in range(X.shape[0]):
            padding_mask[b,:,:] = torch.einsum('i,j->ij', padding_locations[b,:], padding_locations[b,:])

        embedding_input = self.embedding_layer.forward(X)
        output = self.position_encoding.forward(embedding_input)

        for b in self.blocks:
            output = b.forward(output, padding_mask)

        return output, padding_locations


class DecoderBlock(Module):

    def __init__(self, embedding_dim, n_heads) -> None:
        super().__init__()

        self.attention1 = MultiHeadedAttentionLayer(embedding_dim, embedding_dim, n_heads)
        self.attention2 = MultiHeadedAttentionLayer(embedding_dim, embedding_dim, n_heads)

        self.norm1 = LayerNorm(embedding_dim)
        self.norm2 = LayerNorm(embedding_dim)
        self.norm3 = LayerNorm(embedding_dim)

        self.linear1 = Linear(embedding_dim, 4 * embedding_dim)
        self.linear2 = Linear(4 * embedding_dim, embedding_dim)
        self.relu = ReLU()

        self.dropout1 = Dropout(0.2)
        self.dropout2 = Dropout(0.2)
        self.dropout3 = Dropout(0.2)

    def forward(self, encoded_source: torch.Tensor, target: torch.Tensor,
                mask1: Optional[torch.Tensor]=None, mask2: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Implementation of a decoder block. encoded_source has dimensions (batch_size, source_sequence_length, embedding_dim)
        and target has dimensions (batch_size, target_sequence_length, embedding_dim).

        The mask1 is passed to the first multi-headed self-attention layer, and mask2 is passed
        to the second multi-headed self-attention layer.

        Returns its output of shape (batch_size, target_sequence_length, embedding_dim) and
        the attention matrices for each of the heads of the second multi-headed self-attention layer
        (the one where the source and target are "mixed").
        """
        att_out, _ = self.attention1(target, target, target, mask1)
        residual = target + self.dropout1(att_out)

        X = self.norm1(residual)

        att_out, att_weights = self.attention2(X, encoded_source, encoded_source, mask2)

        residual = X + self.dropout2(att_out)
        X = self.norm2(residual)

        temp = self.linear1(X)
        temp = self.relu(temp)
        temp = self.linear2(temp)
        residual = X + self.dropout3(temp)

        return self.norm3(residual), att_weights

class Decoder(Module):

    def __init__(self, vocab_size: int, embedding_dim: int, n_blocks: int, n_heads: int) -> None:
        super().__init__()

        self.embedding_layer = Embedding(vocab_size + 1, embedding_dim, padding_idx=vocab_size)
        self.position_encoding = PositionalEncodingLayer(embedding_dim)
        self.blocks = ModuleList([DecoderBlock(embedding_dim, n_heads) for _ in range(n_blocks)])

        self.linear = Linear(embedding_dim, vocab_size + 1)
        self.softmax = Softmax(-1)

        self.vocab_size = vocab_size

    def _lookahead_mask(self, seq_length: int) -> torch.Tensor:
        """
        Compute the mask to prevent the decoder from looking at future target values.
        The mask you return should be a tensor of shape (sequence_length, sequence_length)
        with only 1's and 0's, where a 0 represent an entry that will be masked in the
        multi-headed attention layer.

        Hint: The function torch.tril (https://pytorch.org/docs/stable/generated/torch.tril.html)
        may be useful.
        """
        return torch.tril(torch.ones(size=(seq_length, seq_length),  dtype=torch.float32))


    def forward(self, encoded_source: torch.Tensor, source_padding: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Transformer decoder. encoded_source has dimensions (batch_size, source_sequence_length, embedding),
        source_padding has dimensions (batch_size, source_seuqence_length) and target has dimensions
        (batch_size, target_sequence_length).

        Returns its output of shape (batch_size, target_sequence_length, target_vocab_size) and
        the attention weights from the first decoder block, of shape
        (batch_size, n_heads, source_sequence_length, target_sequence_length)

        Note that the output is not normalized (i.e. we don't use the softmax function).
        """
        # TODO: Implement the decoder (you should re-use the DecoderBlock class that we provided)
        look_ahead_mask = self._lookahead_mask(target.shape[1]) 

        padding_locations = torch.ones(target.shape[0], target.shape[1])
        padding_locations = padding_locations.masked_fill(target == self.vocab_size, 0)

        padding_look_ahead_mask = torch.ones(target.shape[0], target.shape[1], target.shape[1])
        mask_second_mha_layer = torch.ones(target.shape[0], target.shape[1], source_padding.shape[1])
        for b in range(target.shape[0]):
            padding_look_ahead_mask[b,:,:] = torch.einsum('i,j->ij', padding_locations[b,:], padding_locations[b,:]) * look_ahead_mask
            mask_second_mha_layer[b,:,:] = torch.einsum('i,j->ij', padding_locations[b,:], source_padding[b,:])

        embedding_target = self.embedding_layer.forward(target)
        encoded_target = self.position_encoding.forward(embedding_target)

        first_decoder_block = True
        for b in self.blocks:
            if first_decoder_block:
                encoded_target, att_weights = b.forward(encoded_source, encoded_target, padding_look_ahead_mask, mask_second_mha_layer)
            else:
                encoded_target, _ = b.forward(encoded_source, encoded_target, padding_look_ahead_mask, mask_second_mha_layer)

        output = self.linear(encoded_target)

        return output, att_weights   


class Transformer(Module):

    def __init__(self, source_vocab_size: int, target_vocab_size: int, embedding_dim: int, n_encoder_blocks: int,
                 n_decoder_blocks: int, n_heads: int) -> None:
        super().__init__()

        self.encoder = Encoder(source_vocab_size, embedding_dim, n_encoder_blocks, n_heads)
        self.decoder = Decoder(target_vocab_size, embedding_dim, n_decoder_blocks, n_heads)

    def forward(self, source, target):
        encoded_source, source_padding = self.encoder(source)
        return self.decoder(encoded_source, source_padding, target)

    def predict(self, source: List[int], beam_size=1, max_length=64) -> List[int]:
    #     """
    #     Given a sentence in the source language, you should output a sentence in the target
    #     language of length at most `max_length` that you generate using a beam search with
    #     the given `beam_size`.

    #     Note that the start of sentence token is 2 and the end of sentence token is 3.

    #     Return the final top beam (decided using average log-likelihood) and its average
    #     log-likelihood.

    #     Hint: The follow functions may be useful:
    #         - torch.topk (https://pytorch.org/docs/stable/generated/torch.topk.html)
    #         - torch.softmax (https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html)
    #     """
        self.eval() # Set the PyTorch Module to inference mode (this affects things like dropout)

        if not isinstance(source, torch.Tensor):
            source_input = torch.tensor(source).view(1, -1)
        else:
            source_input = source.view(1, -1)

        beam_width = beam_size

        final_candidates = []
        final_log_prob = []

        #SOS is 2
        candidates = [[2]]
        candidates_log_prob = [0]

        length_translation = 1

        while beam_width > 0 and length_translation < max_length:
            origin_pre_candidate = [e for e in list(range(len(candidates))) for i in range(beam_width)]

            pre_candidates_log_prob = np.zeros(len(candidates)*beam_width, dtype='float32')
            pre_candidates_indices = np.zeros(len(candidates)*beam_width, dtype='int')
            origin_index = 0
            for c in candidates:
                target = torch.tensor(c).view(1, -1)
                
                softmax_input,_= self.forward(source_input, target)
                prob_word = torch.nn.functional.softmax(softmax_input, dim = 2)
                values, indices = torch.topk(prob_word[0,-1,:], beam_width)

                pre_candidates_log_prob[origin_index*beam_width:(origin_index+1)*beam_width] = candidates_log_prob[origin_index] + np.log(values.detach().numpy())
                pre_candidates_indices[origin_index*beam_width:(origin_index+1)*beam_width] = indices.detach().numpy()
                origin_index += 1

            topk_pre_candidates_indices = np.argpartition(pre_candidates_log_prob, -beam_width)[-beam_width:]

            new_candidates_list = []
            new_candidates_log_prob = []
            for k in range(len(topk_pre_candidates_indices)):
                origin_index = origin_pre_candidate[topk_pre_candidates_indices[k]]   
                new_candidate = candidates[origin_index].copy()
                new_candidate.append(pre_candidates_indices[topk_pre_candidates_indices[k]])
                log_p = pre_candidates_log_prob[topk_pre_candidates_indices[k]]

                if pre_candidates_indices[topk_pre_candidates_indices[k]] == 3:
                    beam_width -= 1
                    final_candidates.append(new_candidate)
                    final_log_prob.append(log_p/len(new_candidate))       
                    
                else:
                    new_candidates_list.append(new_candidate)
                    new_candidates_log_prob.append(log_p)

            length_translation +=1

            if length_translation < max_length:
                candidates = new_candidates_list.copy()
                candidates_log_prob = new_candidates_log_prob.copy()
            

        if beam_width > 0 and length_translation == max_length:
            for k in range(len(topk_pre_candidates_indices)):
                origin_index = origin_pre_candidate[topk_pre_candidates_indices[k]]   
                log_p = pre_candidates_log_prob[topk_pre_candidates_indices[k]]
                new_candidate = candidates[origin_index].copy()
                new_candidate.append(pre_candidates_indices[topk_pre_candidates_indices[k]])
                if pre_candidates_indices[topk_pre_candidates_indices[k]] != 3:
                    final_candidates.append(new_candidate)
                    final_log_prob.append(log_p/max_length)
        
        idx_winner = np.argmax(final_log_prob)

        return final_candidates[idx_winner], final_log_prob[idx_winner]

def load_data() -> Tuple[Tuple[List[int], List[int]], Tuple[List[int], List[int]], Dict[int, str], Dict[int, str]]:
    """ Load the dataset.

    :return: (1) train_sentences: list of (source_sentence, target_sentence) pairs, where both source_sentence
                                  and target_sentence are lists of ints
             (2) test_sentences : list of (source_sentence, target_sentence) pairs, where both source_sentence
                                  and target_sentence are lists of ints
             (2) source_vocab   : dictionary which maps from source word index to source word
             (3) target_vocab   : dictionary which maps from target word index to target word
    """
    with open('data/translation_data.bin', 'rb') as f:
        corpus, source_vocab, target_vocab = pickle.load(f)
        test_sentences = corpus[:1000]
        train_sentences = corpus[1000:]
        print("# source vocab: {}\n"
              "# target vocab: {}\n"
              "# train sentences: {}\n"
              "# test sentences: {}\n".format(len(source_vocab), len(target_vocab), len(train_sentences),
                                              len(test_sentences)))
        return train_sentences, test_sentences, source_vocab, target_vocab

def preprocess_data(sentences: Tuple[List[int], List[int]], source_vocab_size,
                    target_vocab_size, max_length):

    source_sentences = []
    target_sentences = []

    for source, target in sentences:
        source = [0] + source + ([source_vocab_size] * (max_length - len(source) - 1))
        target = [0] + target + ([target_vocab_size] * (max_length - len(target) - 1))
        source_sentences.append(source)
        target_sentences.append(target)

    return torch.tensor(source_sentences), torch.tensor(target_sentences)

def decode_sentence(encoded_sentence: List[int], vocab: Dict) -> str:
    if isinstance(encoded_sentence, torch.Tensor):
        encoded_sentence = [w.item() for w in encoded_sentence]
    words = [vocab[w] for w in encoded_sentence if w != 0 and w != 1 and w in vocab]
    return " ".join(words)

def visualize_attention(source_sentence: List[int],
                        output_sentence: List[int],
                        source_vocab: Dict[int, str],
                        target_vocab: Dict[int, str],
                        attention_matrix: np.ndarray,
                        example_id, channel_id):
    """
    :param source_sentence_str: the source sentence, as a list of ints
    :param output_sentence_str: the target sentence, as a list of ints
    :param attention_matrix: the attention matrix, of dimension [target_sentence_len x source_sentence_len]
    :param outfile: the file to output to
    """
    source_length = 0
    while source_length < len(source_sentence) and source_sentence[source_length] != 3:
        source_length += 1

    target_length = 0
    while target_length < len(output_sentence) and output_sentence[target_length] != 3:
        target_length += 1

    source_length += 1
    target_length += 1

    # Set up figure with colorbar
    plt.clf()
    fig = plt.figure()
    fig.set_size_inches(8, 6)
    ax = fig.add_subplot(111)
    cax = ax.matshow(attention_matrix[:target_length, :source_length], cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.xaxis.set_major_locator(ticker.FixedLocator(range(source_length)))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(["PAD" if x not in source_vocab else source_vocab[x] for x in source_sentence[:source_length]]))
    ax.yaxis.set_major_locator(ticker.FixedLocator(range(target_length)))
    ax.yaxis.set_major_formatter(ticker.FixedFormatter(["PAD" if x not in target_vocab else target_vocab[x] for x in output_sentence[:target_length]]))

    ax.set_title('Training Example = '+str(example_id)+', Channel = '+str(channel_id))

    plt.show()
    #plt.close()
    fig.savefig('./Figures/VisAttWeights_Ex'+str(example_id)+'_Ch'+str(channel_id)+'.png')

def train(model: Transformer, train_source: torch.Tensor, train_target: torch.Tensor,
          test_source: torch.Tensor, test_target: torch.Tensor, target_vocab_size: int,
          epochs: int = 30, batch_size: int = 64, lr: float = 0.0001):

    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = CrossEntropyLoss(ignore_index=target_vocab_size)

    epoch_train_loss = np.zeros(epochs)
    epoch_test_loss = np.zeros(epochs)

    for ep in range(epochs):

        train_loss = 0
        test_loss = 0

        permutation = torch.randperm(train_source.shape[0])
        train_source = train_source[permutation]
        train_target = train_target[permutation]

        batches = train_source.shape[0] // batch_size
        model.train()
        for ba in tqdm(range(batches), desc=f"Epoch {ep + 1}"):

            optimizer.zero_grad()

            batch_source = train_source[ba * batch_size: (ba + 1) * batch_size]
            batch_target = train_target[ba * batch_size: (ba + 1) * batch_size]

            target_pred, _ = model(batch_source, batch_target)

            batch_loss = loss_fn(target_pred[:, :-1, :].transpose(1, 2), batch_target[:, 1:])
            batch_loss.backward()
            optimizer.step()

            train_loss += batch_loss.item()

        test_batches = test_source.shape[0] // batch_size
        model.eval()
        for ba in tqdm(range(test_batches), desc="Test", leave=False):

            batch_source = test_source[ba * batch_size: (ba + 1) * batch_size]
            batch_target = test_target[ba * batch_size: (ba + 1) * batch_size]

            target_pred, _ = model(batch_source, batch_target)

            batch_loss = loss_fn(target_pred[:, :-1, :].transpose(1, 2), batch_target[:, 1:])
            test_loss += batch_loss.item()

        epoch_train_loss[ep] = train_loss / batches
        epoch_test_loss[ep] = test_loss / test_batches
        print(f"Epoch {ep + 1}: Train loss = {epoch_train_loss[ep]:.4f}, Test loss = {epoch_test_loss[ep]:.4f}")

    return epoch_train_loss, epoch_test_loss

def get_ngrams(input_list, n):
  return [tuple(input_list[i:i+n]) for i in range(len(input_list)-n+1)]

def bleu_score(predicted: List[int], target: List[int], N: int = 4) -> float:
    """
    Implement a function to compute the BLEU-N score of the predicted
    sentence with a single reference (target) sentence.

    Please refer to the handout for details.

    Make sure you strip the SOS (2), EOS (3), and padding (anything after EOS)
    from the predicted and target sentences.

    If the length of the predicted sentence or the target is less than N,
    the BLEU score is 0.
    """
    t = [target[i] for i in range(target.index(2)+1,target.index(3))]

    if 3 in predicted:
        p = [predicted[i] for i in range(predicted.index(2)+1,predicted.index(3))]
    else:
        p = [predicted[i] for i in range(predicted.index(2)+1, len(predicted))]

    size_predict = len(p)
    size_target = len(t)

    if size_predict < N or size_target < N:
        return 0

    Bleu = 1

    for n in range(1, N+1):
        predicted_ngrams = get_ngrams(p, n)
        predicted_unique_ngrams = set(predicted_ngrams)
        target_ngrams = get_ngrams(t, n)

        score = 0
        for ngram in predicted_unique_ngrams:
            score += min(target_ngrams.count(ngram), predicted_ngrams.count(ngram))

        Bleu *= (score/(size_predict-n+1))**(1/N)

    Bleu *= min(1, np.exp(1-size_target/size_predict))

    return Bleu

def seed_everything(seed=10707):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    # Loads data from English -> Spanish machine translation task
    train_sentences, test_sentences, source_vocab, target_vocab = load_data()

    max_length = 64
    # Generates train/test data based on english and french vocabulary sizes and caps max length of sentence at 64
    train_source, train_target = preprocess_data(train_sentences, len(source_vocab), len(target_vocab), max_length)
    test_source, test_target = preprocess_data(test_sentences, len(source_vocab), len(target_vocab), max_length)

    source_vocab_size = len(source_vocab)
    target_vocab_size = len(target_vocab)

    n_encoder_blocks = [1, 1, 2, 2, 2]
    n_decoder_blocks = [1, 1, 2, 2, 4]
    n_attention_heads = [1, 3, 1, 3, 3]
    model_name = ['A', 'B', 'C', 'D', 'E']

    n_experiments = len(n_encoder_blocks)

    for i in range(n_experiments):
        transformerQ2 = Transformer(source_vocab_size = source_vocab_size, target_vocab_size = target_vocab_size, embedding_dim = 256,
                                     n_encoder_blocks = n_encoder_blocks[i], n_decoder_blocks = n_decoder_blocks[i], n_heads = n_attention_heads[i])
        

        print('Model Q2' + model_name[i])
        train_loss, test_loss = train(model = transformerQ2, train_source = train_source, train_target = train_target,
                                      test_source = test_source, test_target = test_target, target_vocab_size = target_vocab_size)

        #loss plot
        fig = plt.figure(figsize = (5, 5)) 
        plt.clf()
        plt.ylim([0, 6.5])
        plt.plot(list(range(len(train_loss))), train_loss, 'g', label='Training loss')
        plt.plot(list(range(len(test_loss))), test_loss, 'b', label='Test loss')
        plt.title('Training and Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        fig.savefig('./Figures/loss_modelQ2'+model_name[i]+'.png')

        #save results
        np.savez('./Results/loss_modelQ2'+model_name[i]+'.npz', train_loss, test_loss) 
        torch.save(transformerQ2.state_dict(), './ModelState/ModelQ2'+model_name[i]+'.pkl')
        
    
    #Question 3
    for i in range(8):
        ss = decode_sentence(encoded_sentence= test_source[i].tolist(), vocab=source_vocab) 
        ts = decode_sentence(encoded_sentence= test_target[i].tolist(), vocab=target_vocab) 
        encoded_prediction, loglik = transformerQ2.predict(source= test_source[i].tolist(), beam_size=3)
        ps = decode_sentence(encoded_sentence= encoded_prediction, vocab=target_vocab) 

        print(i, '||', ss, '||', ts, '||', ps, '||', round(loglik,4))

        #_, attweights = transformerQ2.forward(train_source[ex,].view(1,-1), train_target[ex,].view(1,-1))

    #Question 4
    for ex in range(3):
        source = train_source[ex,].tolist()
        target = train_target[ex,].tolist()

        #t = [target[i] for i in range(target.index(2),target.index(3)+1)]
        #s = [source[i] for i in range(source.index(2),source.index(3)+1)]
        # source.remove(0)
        # target.remove(0)

        _, attweights = transformerQ2.forward(torch.tensor(source).view(1, -1), torch.tensor(target).view(1, -1))
    
        for ch in range(3):
            visualize_attention(source_sentence= source,
                                output_sentence= target,
                                source_vocab= source_vocab,
                                target_vocab= target_vocab,
                                attention_matrix= attweights[0,ch,:,:].detach().numpy(), 
                                example_id = ex, channel_id = ch)


    #Question 5
    average_loglik = []
    for b in range(1, 9):
        avg_loglik_b = 0
        for i in range(100):
            source= test_source[i].tolist()
            s = [source[i] for i in range(source.index(2),source.index(3)+1)]
            _, loglik = transformerQ2.predict(source= s, beam_size = b)
            avg_loglik_b += loglik
        average_loglik.append(avg_loglik_b/100)        


    fig = plt.figure(figsize = (5, 5)) 
    plt.clf()
    plt.ylim([-1.2, -0.6])
    plt.plot(list(range(1, 9)), average_loglik, 'g')
    plt.title('Average normalized log-likelihood')
    plt.xlabel('Beam size')
    plt.ylabel('Average normalized log-likelihood')       
    plt.show()
    fig.savefig('./Figures/avgloglik_modelQ2E.png')


    #Question 6
    n_tests = test_source.shape[0]
    for m in range(5):
        my_transformer = Transformer(source_vocab_size = source_vocab_size, target_vocab_size = target_vocab_size, embedding_dim = 256,
                                    n_encoder_blocks = n_encoder_blocks[m], n_decoder_blocks = n_decoder_blocks[m], n_heads = n_attention_heads[m])

        my_transformer.load_state_dict(torch.load('./ModelState/ModelQ2'+model_name[m]+'.pkl'))

        avg_BLEU_1 = 0; avg_BLEU_2 = 0; avg_BLEU_3 =0; avg_BLEU_4=0
        for i in range(n_tests):
            source = test_source[i].tolist()
            target = test_target[i].tolist()
            s = [source[i] for i in range(source.index(2),source.index(3)+1)]
            t = [target[i] for i in range(target.index(2),target.index(3)+1)]

            p, _ = my_transformer.predict(source= s, beam_size = 3)

            avg_BLEU_1 += bleu_score(predicted = p, target = t, N = 1)
            avg_BLEU_2 += bleu_score(predicted = p, target = t, N = 2)
            avg_BLEU_3 += bleu_score(predicted = p, target = t, N = 3)
            avg_BLEU_4 += bleu_score(predicted = p, target = t, N = 4)

        print(f"Model2{model_name[m]}: BLEU_1 = {round(avg_BLEU_1/n_tests,4)}, BLEU_2 = {round(avg_BLEU_2/n_tests,4)}, BLEU_3 = {round(avg_BLEU_3/n_tests,4)}, BLEU_4 = {round(avg_BLEU_4/n_tests,4)}")     
                                     
