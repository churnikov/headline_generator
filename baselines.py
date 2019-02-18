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
