import os
import argparse
import json
import torch
import torch.nn as nn
import sentencepiece
import numpy as np
import matplotlib.pyplot as plt
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# adapter to convert from VILT space to T5 space
class EmbeddingAdapter(nn.Module):
    def __init__(self):
        super(EmbeddingAdapter, self).__init__()
        self.VILT_HIDDEN_SIZE = 768
        self.T5_EMBEDDING_SIZE = 512
        
        self.adapter = nn.Sequential(
            nn.Linear(self.VILT_HIDDEN_SIZE, self.T5_EMBEDDING_SIZE),
            nn.GELU(),
            nn.Linear(self.T5_EMBEDDING_SIZE, self.T5_EMBEDDING_SIZE),
            nn.GELU(),
            nn.Linear(self.T5_EMBEDDING_SIZE, self.T5_EMBEDDING_SIZE)
        )

    def forward(self, x):
        x = self.adapter(x)
        return x

# Idea 1 Model: Attentive Patching
class Idea1Model(nn.Module):
    def __init__(self, ):
        super(Idea1Model, self).__init__()
        self.adapter = EmbeddingAdapter()
        self.t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
        self.t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")

    def forward(self, questions_input, captions1_input, captions2_input, img1_prompts_input,
                img2_prompts_input, ie1, ie2, att_ie1, att_ie2, answers_input, generate = False):

      adapted_img1_embeds = self.adapter(ie1).unsqueeze(1)
      adapted_img2_embeds = self.adapter(ie2).unsqueeze(1)

      questions_embeds = self.t5_model.get_input_embeddings()(questions_input['input_ids'])
      caption1_embeds = self.t5_model.get_input_embeddings()(captions1_input['input_ids'])
      caption2_embeds = self.t5_model.get_input_embeddings()(captions2_input['input_ids'])
      img1_prompt_embeds = self.t5_model.get_input_embeddings()(img1_prompts_input['input_ids'])
      img2_prompt_embeds = self.t5_model.get_input_embeddings()(img2_prompts_input['input_ids'])

      inputs_embeds = torch.cat((questions_embeds,
                                 caption1_embeds, img1_prompt_embeds, adapted_img1_embeds,
                                 caption2_embeds, img2_prompt_embeds, adapted_img2_embeds), dim=1)

      attention_mask = torch.hstack((questions_input['attention_mask'],
                                     captions1_input['attention_mask'],
                                     img1_prompts_input['attention_mask'],
                                     att_ie1,
                                     captions2_input['attention_mask'],
                                     img2_prompts_input['attention_mask'],
                                     att_ie2))

      if not generate:
        labels = answers_input['input_ids']
        labels[labels == self.t5_tokenizer.pad_token_id] = -100
        output = self.t5_model(inputs_embeds = inputs_embeds, attention_mask = attention_mask, labels = labels)
      else:
        output = self.t5_model.generate(inputs_embeds = inputs_embeds, attention_mask = attention_mask, max_length=40)

      return output

    def load_model_weights(self, model_path):
       self.load_state_dict(torch.load(model_path))

    def generate_answers_batch(self, data_json_file_name, batch_size, device):

        # read validation data
        with open(data_json_file_name, 'r') as file:
           data = json.load(file)

        # instantiate training and validation dataset objects
        dataset = Idea1Dataset(data)

        # create dataloaders
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn = dataset.collate_fn)

        pred = {}
        with torch.no_grad():
            for batch in data_loader:
               guids = batch['guid']
               batch = {k:v.to(device) for k,v in batch.items() if k != 'guid' }
               question_prompts = self.t5_tokenizer.batch_decode(batch['questions_input']['input_ids'], skip_special_tokens=True)
               #batch['answers_input']['input_ids'][batch['answers_input']['input_ids'] == -100] = self.t5_tokenizer.pad_token_id
               #answers = self.t5_tokenizer.batch_decode(batch['answers_input']['input_ids'], skip_special_tokens=True)
               batch['generate'] = True
               gen_output =  self.forward(**batch)
               pred_out =self.t5_tokenizer.batch_decode(gen_output, skip_special_tokens=True)
               
               for i, q in enumerate(question_prompts):
                  pred[guids[i]] = {'gen_ans': pred_out[i]}
               
        return pred

# dataset class for training idea 1 model
class Idea1Dataset(Dataset):
    def __init__(self, records):
       self.records = records
       self.t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
       self.load_arrays()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
       record = self.records[idx]
       
       guid = record['guid']
       question = record['question']

       answer = None
       if 'answer' in record.keys():
          answer = record['answer']

       captions = record['captions']
       image_embeddings = self.image_embeddings_dict[guid]
       image_attentions = self.image_attentions_dict[guid]

       return {'guid': guid, 'question': question, 'answer': answer, 'captions': captions, 'ie': image_embeddings, 'att_ie': image_attentions}
    
    def load_arrays(self):
       self.image_embeddings_dict = {} 
       self.image_attentions_dict = {}
       
       for record in tqdm(self.records, desc='Processing arrays'):
          guid = record['guid']
          npzfile = np.load('./Data/Arrays/'+guid+'.npz')
          self.image_embeddings_dict[guid] = npzfile['img_embeds']
          self.image_attentions_dict[guid] = npzfile['img_att']
          

    def collate_fn(self, batch):
       question_prompt = [f"Question: {item['question']} Answer this question using the following images and captions." for item in batch]
       
       caption1_prompt = [f"Caption 1: {item['captions'][0]}" for item in batch]
       caption2_prompt = [f"Caption 2: {item['captions'][1]}" for item in batch]

       img1_prompt = ["Image 1: "]*len(batch)
       img2_prompt = ["Image 2: "]*len(batch)

       ie1 = [torch.tensor(item['ie'][0], dtype=torch.float32) for item in batch]
       ie2 = [torch.tensor(item['ie'][1], dtype=torch.float32) for item in batch]

       att_ie1 = torch.tensor([item['att_ie'][0] for item in batch], dtype=torch.long)
       att_ie1 = att_ie1.reshape(len(att_ie1), 1)

       att_ie2 = torch.tensor([item['att_ie'][1] for item in batch], dtype=torch.long)
       att_ie2 = att_ie2.reshape(len(att_ie2), 1)

       guid = [item['guid'] for item in batch]
       answers = [item['answer'] for item in batch]

       questions_input = self.t5_tokenizer(question_prompt, return_tensors="pt", padding=True, truncation=True)
       captions1_input = self.t5_tokenizer(caption1_prompt, return_tensors="pt", padding=True, truncation=True)
       captions2_input = self.t5_tokenizer(caption2_prompt, return_tensors="pt", padding=True, truncation=True)
       answers_input = self.t5_tokenizer(answers, return_tensors="pt", padding=True, truncation=True)
       img1_prompts_input = self.t5_tokenizer(img1_prompt, return_tensors="pt", padding=True, truncation=True)
       img2_prompts_input = self.t5_tokenizer(img2_prompt, return_tensors="pt", padding=True, truncation=True)

       collated_batch = {}
       collated_batch['guid'] = guid
       collated_batch['questions_input'] = questions_input
       collated_batch['captions1_input'] = captions1_input
       collated_batch['captions2_input'] = captions2_input
       collated_batch['img1_prompts_input'] = img1_prompts_input
       collated_batch['img2_prompts_input'] = img2_prompts_input
       collated_batch['ie1'] = torch.vstack(ie1)
       collated_batch['ie2'] = torch.vstack(ie2)
       collated_batch['att_ie1'] = att_ie1
       collated_batch['att_ie2'] = att_ie2
       collated_batch['answers_input'] = answers_input

       return collated_batch
    
def train_idea1_model(root, train_data_json_file, validation_data_json_file, number_epochs, batch_size, lr):

    if not os.path.exists(root):
        print('root provided does not exist')
        return
    
    # name of directory where model weights and figures will be saved
    output_folder = root+'/Trained Models/nepch_' + str(number_epochs) + '_bs_' + str(batch_size) + '_lr_' + str(lr)

    # create output directory if it already does not exist
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # read training data
    with open(train_data_json_file, 'r') as file:
        training_data = json.load(file)
    
    # read validation data
    with open(validation_data_json_file, 'r') as file:
        val_data = json.load(file)

    # instantiate training and validation dataset objects
    trainDataset = Idea1Dataset(training_data)
    valDataset = Idea1Dataset(val_data)

    # create dataloaders
    train_data_loader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True, collate_fn = trainDataset.collate_fn)
    val_data_loader = DataLoader(valDataset, batch_size=batch_size, shuffle=False, collate_fn = valDataset.collate_fn)

    # instantiate idea1 model object
    model = Idea1Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    number_batches_train = len(train_data_loader)
    number_batches_val = len(val_data_loader)

    best_val_loss = np.inf
    train_losses = []
    val_losses = []
    for epoch in range(number_epochs):
        model.train()
        epoch_loss = 0.0

        for batch in train_data_loader:
            batch = {k:v.to(device) for k,v in batch.items() if k != 'guid'}

            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(**batch)

            loss = outputs.loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= number_batches_train

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_data_loader:
                batch = {k:v.to(device) for k,v in batch.items() if k != 'guid'}
                outputs = model(**batch)
                loss = outputs.loss
                val_loss += loss.item()


        average_val_loss = val_loss / number_batches_val

        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            torch.save(model.state_dict(), output_folder+'/best_model1_weights.pth')

        print(f"Epoch {epoch + 1}, Training Loss: {epoch_loss:.4f}, Validation Loss: {average_val_loss:.4f}")

        train_losses.append(epoch_loss)
        val_losses.append(average_val_loss)
    
    # plot loss function
    plt.clf()
    plt.plot(range(number_epochs), train_losses, label='Training Loss')
    plt.plot(range(number_epochs), val_losses, label='Validation Loss')

    # add labels and title
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')

    # add legend
    plt.legend()

    # save the plot as a PNG file
    plt.savefig(output_folder+'/training_validation_plot.png')

#/content/drive/My Drive/11-777 Multimodal/Final/Idea1
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', dest='root', type=str, default='/content/drive/My Drive/11-777 Multimodal/Final/Idea1/', help='Project root directory')
    parser.add_argument('--train-data-file', dest="train_data_file", type=str, default='Data/idea1_train_data.json', help='Name of json file with training annotated data.')
    parser.add_argument('--val-data-file', dest="val_data_file", type=str, default='Data/idea1_val_data.json', help='Name of json file with validation annotated data.')
    parser.add_argument('--number-epochs', dest="number_epochs", type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--batch-size', dest="batch_size", type=int, default = 16, help="Batch size")
    parser.add_argument('--lr', dest="lr", type=float, default = 5e-5, help='Learning rate')    
    
    return parser.parse_args()

if __name__ == '__main__':
    
    args = parse_arguments()

    print(f"training data file: {args.train_data_file}, validation data file: {args.val_data_file}, number of epochs:{args.number_epochs}")
    print(f"batch size: {args.batch_size}, lr: {args.lr}")

    train_idea1_model(root= args.root,
                      train_data_json_file=args.train_data_file,
                      validation_data_json_file=args.val_data_file,
                      number_epochs=args.number_epochs,
                      batch_size=args.batch_size,
                      lr=args.lr)   