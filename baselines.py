import os
import random
from typing import Callable, Optional

from gensim.summarization.summarizer import summarize
from gensim.summarization.textcleaner import split_sentences
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
import torch

from loader import json_iterator
from preprocessing import BasicHtmlPreprocessor


class GensimSummarizer:
    def __init__(self, model_params, file_path, pred_file_name, ref_file_name,
                 preprocessor: Optional[BasicHtmlPreprocessor]=None):
        self.preprocessor = preprocessor
        self.pred_file_name = pred_file_name
        self.ref_file_name = ref_file_name
        self.file_path = file_path
        self.model_params = model_params

    def open_save_files(self):
        self.out_pred = open(self.pred_file_name, 'w')
        self.out_ref = open(self.ref_file_name, 'w')

    def close_save_file(self):
        self.out_pred.close()
        self.out_ref.close()

    def summarize_text(self, text: str):
        if len(split_sentences(text)) > 1:
            try:
                pred: str = summarize(text, **self.model_params)
            except ValueError:
                pred = text
        else:
            pred: str = text
        if not pred:
            pred = 'none'

        return pred

    def summarize(self):
        self.open_save_files()

        for jsn in tqdm(json_iterator(self.file_path), 'Predicting'):
            reference: str = jsn['title']

            text = jsn['text']
            if self.preprocessor is not None:
                text = self.preprocessor.transform(text)

            pred = self.summarize_text(text)

            self.out_pred.write(pred.replace('\n', '\\n') + '\n')
            self.out_ref.write(reference + '\n')

        self.close_save_file()


class ExtractFirstFullSentence(GensimSummarizer):

    def summarize_text(self, text: str):
        sentences = split_sentences(text)
        if 'риа новости' in sentences[0]:
            return sentences[1]
        else:
            return sentences[0]


class Encoder(nn.Module):

    def __init__(self, embedding: nn.Embedding, lstm_n_layers=4, lstm_hidden_size=512, embedding_dim=300,
                 lstm_batch_first=False, lstm_bidirectional=True, lstm_dropout=0.5, embed_dropout=0.5):
        super().__init__()
        self.hidden_size = lstm_hidden_size
        self.n_layers = lstm_n_layers
        self.embedding_dim = embedding_dim
        self.embedding = embedding
        self.lstm_bidirectional = lstm_bidirectional

        self.dropout = nn.Dropout(embed_dropout)
        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_size,
                            num_layers=self.n_layers, batch_first=lstm_batch_first,
                            bidirectional=lstm_bidirectional, dropout=lstm_dropout)

    def forward(self, input, input_lengths):

        embed = self.dropout(self.embedding(input))
        embed = pack_padded_sequence(embed, input_lengths, batch_first=True)
        # Input size (seq_len, batch, input_size)
        self.lstm.flatten_parameters()
        outputs, (h, c) = self.lstm(embed)
        outputs, output_lens = pad_packed_sequence(outputs, batch_first=True)
        if self.lstm_bidirectional:
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]

        return outputs, (h, c)


class Decoder(nn.Module):

    def __init__(self, embedding: nn.Embedding, vocab_size, lstm_n_layers=4, lstm_hidden_size=512, embedding_dim=300,
                 lstm_batch_first=False, lstm_bidirectional=True, lstm_dropout=0.5, embed_dropout=0.5):
        super().__init__()
        self.hidden_size = lstm_hidden_size
        self.n_layers = lstm_n_layers
        self.embedding_dim = embedding_dim
        self.embedding = embedding

        self.dropout = nn.Dropout(embed_dropout)
        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_size,
                            num_layers=self.n_layers, batch_first=lstm_batch_first,
                            bidirectional=lstm_bidirectional, dropout=lstm_dropout)

        self.vocab_size = vocab_size
        self.out = nn.Linear(lstm_hidden_size * (2 if lstm_bidirectional else 1), vocab_size)
        self.softmax = nn.Softmax()

    def forward(self, input, hidden):
        embed = self.dropout(self.embedding(input.unsqueeze(1)))

        self.lstm.flatten_parameters()
        out, hidden = self.lstm(embed, hidden)
        pred = self.softmax(self.out(out.squeeze(1)))

        return pred, hidden


class Seq2SeqSummarizer(nn.Module):
    """Implementation of http://anlp.jp/proceedings/annual_meeting/2018/pdf_dir/A1-2.pdf with bpe"""
    def __init__(self, encoder: Encoder, decoder: Decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert self.encoder.hidden_size == self.decoder.hidden_size
        assert self.encoder.n_layers == self.decoder.n_layers

    def forward(self, input_batch: torch.LongTensor, ground_truth: torch.LongTensor, input_lenght, output_length,
                teacher_forcing_ratio=0.5):
        batch_size = input_batch.shape[0]
        max_len = ground_truth.shape[1]
        out_size = self.decoder.vocab_size

        outputs = torch.zeros(max_len, batch_size, out_size).to(self.device)

        # Start tokens
        inp = ground_truth[:, 0]

        encoder_output, hidden = self.encoder(input_batch, input_lenght)

        for i in range(1, max_len):
            output, hidden = self.decoder(inp, hidden)
            outputs[i] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            inp = ground_truth[:, i] if teacher_force else top1

        return outputs


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()


class AttentionEncoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, n_heads, pf_dim, encoder_layer, self_attention,
                 positionwise_feedforward, dropout, device):
        super().__init__()

        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.encoder_layer = encoder_layer
        self.self_attention = self_attention
        self.positionwise_feedforward = positionwise_feedforward
        self.dropout = dropout
        self.device = device

        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(30000, hid_dim)

        self.layers = nn.ModuleList(
            [encoder_layer(hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device)
             for _ in range(n_layers)])

        self.do = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, src, src_mask):
        # src = [batch size, src sent len]
        # src_mask = [batch size, src sent len]

        pos = torch.arange(0, src.shape[1]).unsqueeze(0).repeat(src.shape[0], 1).to(self.device)

        src = self.do((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))

        # src = [batch size, src sent len, hid dim]

        for layer in self.layers:
            src = layer(src, src_mask)

        return src


class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device):
        super().__init__()

        self.ln = nn.LayerNorm(hid_dim)
        self.sa = self_attention(hid_dim, n_heads, dropout, device)
        self.pf = positionwise_feedforward(hid_dim, pf_dim, dropout)
        self.do = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # src = [batch size, src sent len, hid dim]
        # src_mask = [batch size, src sent len]

        src = self.ln(src + self.do(self.sa(src, src, src, src_mask)))

        src = self.ln(src + self.do(self.pf(src)))

        return src


class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)

        self.do = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)

    def forward(self, query, key, value, mask=None):
        bsz = query.shape[0]

        # query = key = value [batch size, sent len, hid dim]

        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        # Q, K, V = [batch size, sent len, hid dim]

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)

        # Q, K, V = [batch size, n heads, sent len, hid dim // n heads]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # energy = [batch size, n heads, sent len, sent len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.do(F.softmax(energy, dim=-1))

        # attention = [batch size, n heads, sent len, sent len]

        x = torch.matmul(attention, V)

        # x = [batch size, n heads, sent len, hid dim // n heads]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, sent len, n heads, hid dim // n heads]

        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))

        # x = [batch size, src sent len, hid dim]

        x = self.fc(x)

        # x = [batch size, sent len, hid dim]

        return x


class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.pf_dim = pf_dim

        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)

        self.do = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch size, sent len, hid dim]

        x = x.permute(0, 2, 1)

        # x = [batch size, hid dim, sent len]

        x = self.do(F.relu(self.fc_1(x)))

        # x = [batch size, ff dim, sent len]

        x = self.fc_2(x)

        # x = [batch size, hid dim, sent len]

        x = x.permute(0, 2, 1)

        # x = [batch size, sent len, hid dim]

        return x


class AttentionDecoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, n_heads, pf_dim, decoder_layer, self_attention,
                 positionwise_feedforward, dropout, device):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.decoder_layer = decoder_layer
        self.self_attention = self_attention
        self.positionwise_feedforward = positionwise_feedforward
        self.dropout = dropout
        self.device = device

        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(30000, hid_dim)

        self.layers = nn.ModuleList(
            [decoder_layer(hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device)
             for _ in range(n_layers)])

        self.fc = nn.Linear(hid_dim, output_dim)

        self.do = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, trg, src, trg_mask, src_mask):
        # trg = [batch_size, trg sent len]
        # src = [batch_size, src sent len]
        # trg_mask = [batch size, trg sent len]
        # src_mask = [batch size, src sent len]

        pos = torch.arange(0, trg.shape[1]).unsqueeze(0).repeat(trg.shape[0], 1).to(self.device)

        trg = self.do((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))

        # trg = [batch size, trg sent len, hid dim]

        for layer in self.layers:
            trg = layer(trg, src, trg_mask, src_mask)

        return self.fc(trg)


class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device):
        super().__init__()

        self.ln = nn.LayerNorm(hid_dim)
        self.sa = self_attention(hid_dim, n_heads, dropout, device)
        self.ea = self_attention(hid_dim, n_heads, dropout, device)
        self.pf = positionwise_feedforward(hid_dim, pf_dim, dropout)
        self.do = nn.Dropout(dropout)

    def forward(self, trg, src, trg_mask, src_mask):
        # trg = [batch size, trg sent len, hid dim]
        # src = [batch size, src sent len, hid dim]
        # trg_mask = [batch size, trg sent len]
        # src_mask = [batch size, src sent len]

        trg = self.ln(trg + self.do(self.sa(trg, trg, trg, trg_mask)))

        trg = self.ln(trg + self.do(self.ea(trg, src, src, src_mask)))

        trg = self.ln(trg + self.do(self.pf(trg)))

        return trg


class AttentionSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, pad_idx, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        self.device = device

    def make_masks(self, src, trg):
        # src = [batch size, src sent len]
        # trg = [batch size, trg sent len]

        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)

        trg_pad_mask = (trg != self.pad_idx).unsqueeze(1).unsqueeze(3)

        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), dtype=torch.uint8, device=self.device))

        trg_mask = trg_pad_mask & trg_sub_mask

        return src_mask, trg_mask

    def forward(self, src, trg):
        # src = [batch size, src sent len]
        # trg = [batch size, trg sent len]

        src_mask, trg_mask = self.make_masks(src, trg)

        enc_src = self.encoder(src, src_mask)

        # enc_src = [batch size, src sent len, hid dim]

        out = self.decoder(trg, enc_src, trg_mask, src_mask)

        # out = [batch size, trg sent len, output dim]

        return out
