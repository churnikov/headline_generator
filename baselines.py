import os
from typing import Callable, Optional

from gensim.summarization.summarizer import summarize
from gensim.summarization.textcleaner import split_sentences
from tqdm import tqdm
from torch import nn

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
                 lstm_batch_first=True, lstm_bidirectional=True, lstm_dropout=0.5, embed_dropout=0.5):
        super().__init__()
        self.hidden_size = lstm_hidden_size
        self.n_layers = lstm_n_layers
        self.embedding_dim = embedding_dim
        self.embedding = embedding

        self.dropout = nn.Dropout(embed_dropout)
        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_size,
                            num_layers=self.n_layers, batch_first=lstm_batch_first,
                            bidirectional=lstm_bidirectional, dropout=lstm_dropout)

    def forward(self, input):
        embed = self.dropout(self.embedding(input))

        _, (h, c) = self.lstm(embed)

        return (h, c)


class Decoder(nn.Module):

    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.embedding = embedding


class Seq2SeqSummarizer(nn.Module):
    """Implementation of http://anlp.jp/proceedings/annual_meeting/2018/pdf_dir/A1-2.pdf with bpe"""
    def __init__(self, encoder, decoder):
        super().__init__()
        self.decoder = decoder
        self.encoder = encoder
