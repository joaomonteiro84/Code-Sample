
import unittest
import pickle as pk

from numpy.testing import assert_allclose
import torch

import transformer
from transformer import seed_everything

SEED = 10707

TOLERANCE = 1e-4

with open("tests.pk", "rb") as f: TESTS = pk.load(f)
TEST_MODELS = torch.load('test_models.pth')

class TestPositionalEncoding(unittest.TestCase):

    def setUp(self):
        seed_everything(SEED)
        self.layer = transformer.PositionalEncodingLayer(32)
        self.ans_key = "positional_encoding"

    def test(self):
        x = torch.arange(3*16*32, dtype=torch.float).reshape((3, 16, 32))
        output = self.layer(x).detach().numpy()
        assert_allclose(output, TESTS[self.ans_key], atol=TOLERANCE)

class TestSelfAttention(unittest.TestCase):

    def setUp(self):
        seed_everything(SEED)
        self.layer = transformer.SelfAttentionLayer(12, 30)
        self.ans_key = "self_attention"

    def test(self):
        qx = torch.arange(3*8*12, dtype=torch.float).reshape((3, 8, 12))
        kx = torch.arange(3*10*12, dtype=torch.float).reshape((3, 10, 12)) + 0.5
        vx = torch.arange(3*10*12, dtype=torch.float).reshape((3, 10, 12)) - 0.75

        out, weights = self.layer(qx, kx, vx)

        assert_allclose(out.detach().numpy(), TESTS[self.ans_key][0], atol=TOLERANCE)
        assert_allclose(weights.detach().numpy(), TESTS[self.ans_key][1], atol=TOLERANCE)

class TestSelfAttentionWithMask(unittest.TestCase):

    def setUp(self):
        seed_everything(SEED)
        self.layer = transformer.SelfAttentionLayer(12, 30)
        self.ans_key = "self_attention_masked"

    def test(self):
        qx = torch.arange(3*8*12, dtype=torch.float).reshape((3, 8, 12))
        kx = torch.arange(3*10*12, dtype=torch.float).reshape((3, 10, 12)) + 0.5
        vx = torch.arange(3*10*12, dtype=torch.float).reshape((3, 10, 12)) - 0.75

        mask = torch.ones((3, 8, 10), dtype=torch.float)
        mask[0, 4:, :] = 0.
        mask[1, :, 7:] = 0.
        mask[2, 3:, 6:] = 0.

        out, weights = self.layer(qx, kx, vx, mask=mask)

        assert_allclose(out.detach().numpy(), TESTS[self.ans_key][0], atol=TOLERANCE)
        assert_allclose(weights.detach().numpy(), TESTS[self.ans_key][1], atol=TOLERANCE)

class TestMask(unittest.TestCase):

    def setUp(self):
        seed_everything(SEED)
        self.layer = transformer.Decoder(5, 5, 5, 5)
        self.ans_key = "lookahead_mask"

    def test(self):
        assert_allclose(self.layer._lookahead_mask(10), TESTS[self.ans_key], atol=TOLERANCE)

class TestEncoder(unittest.TestCase):

    def setUp(self):
        seed_everything(SEED)
        self.layer = transformer.Encoder(30, 30, 5, 3)
        self.layer.load_state_dict(TEST_MODELS['encoder_state_dict'])
        self.layer.eval()
        self.ans_key = "encoder_forward"

    def test(self):
        seed_everything(SEED)
        encoded_input = TESTS['encoder_input']
        en_output = self.layer.forward(encoded_input)
        assert_allclose(en_output[0].detach().numpy(), TESTS[self.ans_key][0], atol=TOLERANCE)

class TestDecoder(unittest.TestCase):

    def setUp(self):
        seed_everything(SEED)
        self.layer = transformer.Decoder(35, 30, 5, 3)
        self.layer.load_state_dict(TEST_MODELS['decoder_state_dict'])
        self.layer.eval()
        self.ans_key = "decoder_forward"

    def test(self):
        seed_everything(SEED)
        de_input = TESTS['decoder_inputs']
        de_output = self.layer.forward(
            TESTS['decoder_inputs']['source'],
            TESTS['decoder_inputs']['source_padding'],
            TESTS['decoder_inputs']['target']
        )[0]
        assert_allclose(de_output.detach().numpy(), TESTS[self.ans_key], atol=1e-3)

class TestBeamSearch(unittest.TestCase):

    def setUp(self):
        self.source_vocab_size = 5080
        self.target_vocab_size = 7835
        self.transformer = transformer.Transformer(self.source_vocab_size, self.target_vocab_size , 256, 2, 2, 3)

        self.ans_key = "transformer"
        self.transformer.load_state_dict(TESTS[self.ans_key][0])

    def test(self):

        x = [2, 25, 26, 3193, 233, 132, 16, 1337, 5, 3, self.source_vocab_size, self.source_vocab_size]
        for b in range(1, 6):
            out, ll = self.transformer.predict(x, beam_size=b)
            assert_allclose(out, TESTS[self.ans_key][2*b-1], atol=TOLERANCE)
            assert_allclose(ll, TESTS[self.ans_key][2*b], atol=TOLERANCE)

        x = [2, 5, 4, 6, 8, 10, 12, 14, 3, self.source_vocab_size, self.source_vocab_size, self.source_vocab_size]

        for b in range(1, 6):
            out, ll = self.transformer.predict(x, beam_size=b)
            assert_allclose(out, TESTS[self.ans_key][9+2*b], atol=TOLERANCE)
            assert_allclose(ll, TESTS[self.ans_key][10+2*b], atol=TOLERANCE)

        x = [2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 3, self.source_vocab_size]

        for b in range(1, 6):
            out, ll = self.transformer.predict(x, beam_size=b, max_length=6)
            assert_allclose(out, TESTS[self.ans_key][19+2*b], atol=TOLERANCE)
            assert_allclose(ll, TESTS[self.ans_key][20+2*b], atol=TOLERANCE)

class TestBleuScore(unittest.TestCase):

    def setUp(self):
        self.ans_key = "bleu_score"

    def test(self):
        target = [2, 10, 11 , 4, 5, 6, 7, 8, 9, 3, 3796, 3796]
        predicted = [2, 10, 11, 4, 5, 6, 13, 12, 3, 3796, 3796]

        for n in range(1, 5):
            assert_allclose(transformer.bleu_score(predicted, target, N=n), TESTS[self.ans_key][n-1], atol=TOLERANCE)

        predicted = [2, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3796, 3796]
        assert_allclose(transformer.bleu_score(predicted, target, N=1), TESTS[self.ans_key][4], atol=TOLERANCE)

        predicted = [2, 5, 15, 4, 10, 6, 3, 3796, 3796, 3796, 3796, 3796]
        for n in range(1, 5):
            assert_allclose(transformer.bleu_score(predicted, target, N=n), TESTS[self.ans_key][n+4], atol=TOLERANCE)

        predicted = [2, 4, 88, 4, 5, 6, 10, 11, 12, 7, 8, 3]
        for n in range(1, 5):
            assert_allclose(transformer.bleu_score(predicted, target, N=n), TESTS[self.ans_key][n+8], atol=TOLERANCE)

        target = [2, 99, 92, 6, 4, 4, 4, 5, 5, 5, 5, 3]
        predicted = [2, 6, 6, 6, 6, 4, 5, 3]

        for n in range(1, 5):
            assert_allclose(transformer.bleu_score(predicted, target, N=n), TESTS[self.ans_key][n+12], atol=TOLERANCE)

        predicted = [2, 11, 4, 3]
        assert_allclose(transformer.bleu_score(predicted, target, N=4), TESTS[self.ans_key][17], atol=TOLERANCE)

        predicted = [2, 12, 13, 4, 5, 6, 7, 3]
        target = [2,10, 311, 3]
        assert_allclose(transformer.bleu_score(predicted, target, N=4), TESTS[self.ans_key][18], atol=TOLERANCE)
