import csv
import json
import os
import sys
from collections import Counter
import argparse
from typing import Dict, Union, Optional

import torch
from torch import nn
from torch import optim
from bpemb import BPEmb
from torch.nn import CrossEntropyLoss
from torchtext.data import Field, Example, Dataset, BucketIterator
from torchtext.vocab import Vocab
from tqdm import tqdm

from baselines import Encoder, Decoder, Seq2SeqSummarizer

csv.field_size_limit(sys.maxsize)

try:
    nn.GRU(10, 10).to('cuda')
except Exception:
    pass

parser = argparse.ArgumentParser()
parser.add_argument('--config')
args = parser.parse_args()
config_path = args.config

with open(config_path, 'r') as f:
    config = json.load(f)

MODEL_NAME = config['model']['name']
EMB_DIM = config['model']['embedding']['params']['dim']
VOCAB_SIZE = config['model']['embedding']['params']['vocab_size']
BATCH_SIZE = config['training_params']['batch_size']
TRAIN_DATA_PATH = config['data']['train_file_path']
TEST_DATA_PATH = config['data']['test_file_path']
SAVE_MODEL_PATH = os.path.join(config['results']['output_dir'], MODEL_NAME)
if not os.path.exists(SAVE_MODEL_PATH):
    os.makedirs(SAVE_MODEL_PATH)
SAVE_NAME = os.path.join(SAVE_MODEL_PATH, config['results']['model_save_name'])

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'
PAD_TOKEN = '<pad>'

if config['model']['embedding']['name'] == 'bpe':
    bpe = BPEmb(lang='ru', vs=VOCAB_SIZE, dim=EMB_DIM, add_pad_emb=True)

    text_field = Field(init_token=SOS_TOKEN, eos_token=EOS_TOKEN, tokenize=bpe.encode, pad_token=PAD_TOKEN)
    text_field.vocab = Vocab(Counter(bpe.words))

    embedding = nn.Embedding.from_pretrained(torch.tensor(bpe.vectors, dtype=torch.float32))
    embedding.to(DEVICE)
else:
    raise NotImplementedError(f"Embedding {config['model']['embedding']['name']} not supported")

if config['model']['encoder']['name'] == 'lstm_encoder':
    encoder_params: Dict[str, Union[int, float, bool, Optional[str]]] = config['model']['encoder']['params']
    encoder_device: Optional[str] = encoder_params.pop('device') or DEVICE

    encoder = Encoder(embedding=embedding,
                      lstm_n_layers=encoder_params['lstm_n_layers'],
                      lstm_hidden_size=encoder_params['lstm_hidden_size'],
                      embedding_dim=EMB_DIM,
                      lstm_batch_first=encoder_params['lstm_batch_first'],
                      lstm_bidirectional=encoder_params['lstm_bidirectional'],
                      lstm_dropout=encoder_params['lstm_dropout'],
                      embed_dropout=encoder_params['embed_dropout'])
    encoder.to(encoder_device)
else:
    raise NotImplementedError(f"Encoder {config['model']['encoder']['name']} not supported")

if config['model']['decoder']['name'] == 'lstm_decoder':
    decoder_params: Dict[str, Union[int, float, bool, Optional[str]]] = config['model']['decoder']['params']
    decoder_device: Optional[str] = decoder_params.pop('device') or DEVICE

    decoder = Decoder(embedding=embedding, vocab_size=VOCAB_SIZE,
                      lstm_n_layers=decoder_params['lstm_n_layers'],
                      lstm_hidden_size=decoder_params['lstm_hidden_size'],
                      embedding_dim=EMB_DIM,
                      lstm_batch_first=decoder_params['lstm_batch_first'],
                      lstm_bidirectional=decoder_params['lstm_bidirectional'],
                      lstm_dropout=decoder_params['lstm_dropout'],
                      embed_dropout=decoder_params['embed_dropout'])
    decoder.to(decoder_device)
else:
    raise NotImplementedError(f"Decoder {config['model']['decoder']['name']} not supported")

if config['model']['name'] == 'Seq2SeqSummarizer':
    model = Seq2SeqSummarizer(encoder, decoder, device=DEVICE).to(DEVICE)
else:
    raise NotImplementedError(f"Model {config['model']['name']} not supported")

if config['training_params']['criterion']['name'] == 'CrossEntropyLoss':
    loss = CrossEntropyLoss(ignore_index=text_field.vocab.stoi[PAD_TOKEN])
else:
    raise NotImplementedError(f"Loss {config['training_params']['criterion']['name']} not supported")

if config['training_params']['optimizer']['name'] == 'Adam':
    optimizer = optim.Adam(model.parameters())
else:
    raise NotImplementedError(f"Loss {config['training_params']['criterion']['name']} not supported")


def train(model, train_data, optimizer, criterion, clip, teacher_forcing_ratio):
    model.train()

    epoch_loss = 0
    i = 0

    with tqdm(bar_format='{postfix[0]} {postfix[3][iter]}/{postfix[2]} {postfix[1]}: {postfix[1][loss]}',
              postfix=['Training iter:', 'Loss', dict(loss=0, iter=0)]) as t:
        for i, data in enumerate(train_data):
            x_train, y_train = data.text, data.title

            optimizer.zero_grad()

            output = model.forward(x_train, y_train, teacher_forcing_ratio)

            y_true = y_train[1:].view(-1)
            loss = criterion(output[1:].view(-1, output.shape[2]), y_true)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            optimizer.step()

            epoch_loss += loss.item()

            t.postfix[2]['loss'] = loss.item()
            t.postfix[2]['iter'] = i
            t.update()

            if DEVICE == 'cuda':
                del x_train
                del y_train
                torch.cuda.empty_cache()

    return epoch_loss / (i or 1)


def evaluate(model, validation_data, criterion):
    model.eval()

    epoch_loss = 0
    i = 1

    with torch.no_grad():
        for i, data in tqdm(enumerate(validation_data), desc='Validating'):
            x_val, y_val = data.text, data.title

            y_true = y_val[1:].view(-1)

            output = model.forward(x_val, y_val, 0)

            loss = criterion(output[1:].view(-1, output.shape[2]), y_true)

            epoch_loss += loss.item()

            if DEVICE == 'cuda':
                del x_val
                del y_val
                torch.cuda.empty_cache()

    return epoch_loss / (i or 1)


def predict(model, x, max_len, end_symbol, id2word):
    model.eval()

    with torch.no_grad():
        _, hidden = model.encoder(x)
        inp = x[0, :]

        symbol = ''
        output = []

        while symbol != end_symbol and len(output) < max_len:
            out, hidden = model.decoder(inp, hidden)

            idx = out.max(1)[1]
            symbol = id2word[idx.item()]

            output.append(symbol)
            inp = idx

    return ''.join(output)


def lazy_examples(csv_source):
    with open(csv_source) as f:
        reader = csv.reader(f)
        next(reader)
        for text, title in reader:
            yield Example.fromlist([text, title], [('text', text_field), ('title', text_field)])


best_valid_loss = float('inf')

for epoch in range(10):
    train_dataset = Dataset(lazy_examples(TRAIN_DATA_PATH), [('text', text_field), ('title', text_field)])
    train_iterator = BucketIterator(train_dataset, batch_size=BATCH_SIZE,
                                    sort_key=lambda x: len(x.text), shuffle=False, device=DEVICE)
    val_dataset = Dataset(lazy_examples(TEST_DATA_PATH), [('text', text_field), ('title', text_field)])
    val_iterator = BucketIterator(val_dataset, batch_size=BATCH_SIZE,
                                  sort_key=lambda x: len(x.text), shuffle=False, device=DEVICE)

    train_loss = train(model, train_iterator, optimizer, loss, 1, 0.5)
    valid_loss = evaluate(model, val_iterator, loss)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
    torch.save(model.state_dict(), SAVE_NAME)

    print(
        f'| Epoch: {epoch+1:03} | Train Loss: {train_loss:.3f} | Valid Loss {valid_loss:.3f}')
