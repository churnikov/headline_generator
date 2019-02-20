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

from baselines import Encoder, Decoder, Seq2SeqSummarizer, AttentionEncoder, AttentionSeq2Seq, \
    AttentionDecoder, NoamOpt, EncoderLayer, DecoderLayer, SelfAttention, PositionwiseFeedforward

csv.field_size_limit(sys.maxsize)

try:
    nn.GRU(10, 10).to('cuda')
except Exception:
    pass

parser = argparse.ArgumentParser()
parser.add_argument('--config')
parser.add_argument('--train_path')
parser.add_argument('--test_path')
args = parser.parse_args()
config_path = args.config

with open(config_path, 'r') as f:
    config = json.load(f)

MODEL_NAME = config['model']['name']
EMB_DIM = config['model']['embedding']['params']['dim']
VOCAB_SIZE = config['model']['embedding']['params']['vocab_size']
BATCH_SIZE = config['training_params']['batch_size']
if args.train_path:
    TRAIN_DATA_PATH = args.train_path
elif 'data' in config and 'train_file_path' in config['data']:
    TRAIN_DATA_PATH = config['data']['train_file_path']
else:
    raise FileNotFoundError('Train file is not provided. '
                            'Use --train_path or set it in config file `data.train_file_path`')
if args.test_path:
    TEST_DATA_PATH = args.test_path
elif 'data' in config and 'test_file_path' in config['data']:
    TEST_DATA_PATH = config['data']['test_file_path']
else:
    raise FileNotFoundError('Test file is not provided. Use --test_path or set it in config file `data.test_file_path`')
SAVE_MODEL_PATH = os.path.join(config['results']['output_dir'], MODEL_NAME)
if not os.path.exists(SAVE_MODEL_PATH):
    os.makedirs(SAVE_MODEL_PATH)
SAVE_NAME = os.path.join(SAVE_MODEL_PATH, config['results']['model_save_name'])

if config['model']['params']['device']:
    DEVICE = torch.device(config['model']['params']['device'])
else:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'
PAD_TOKEN = '<pad>'

if config['model']['embedding']['name'] == 'bpe':
    bpe = BPEmb(lang='ru', vs=VOCAB_SIZE-1, dim=EMB_DIM, add_pad_emb=True)

    text_field = Field(init_token=SOS_TOKEN, eos_token=EOS_TOKEN, tokenize=bpe.encode, pad_token=PAD_TOKEN,
                       include_lengths=True, batch_first=True)
    text_field.vocab = Vocab(Counter(bpe.words))

    embedding = nn.Embedding.from_pretrained(torch.tensor(bpe.vectors, dtype=torch.float32))
    embedding.to(DEVICE)
else:
    raise NotImplementedError(f"Embedding {config['model']['embedding']['name']} not supported")

if 'encoder' in config['model']:
    encoder_name = config['model']['encoder']['name']
    encoder_params: Dict[str, Union[int, float, bool, Optional[str]]] = config['model']['encoder']['params']
    encoder_device_name = encoder_params.pop('device')
    encoder_device = torch.device(encoder_device_name) if encoder_device_name else DEVICE
    if encoder_name == 'lstm_encoder':
        encoder = Encoder(embedding=embedding,
                          lstm_n_layers=encoder_params['lstm_n_layers'],
                          lstm_hidden_size=encoder_params['lstm_hidden_size'],
                          embedding_dim=EMB_DIM,
                          lstm_batch_first=encoder_params['lstm_batch_first'],
                          lstm_bidirectional=encoder_params['lstm_bidirectional'],
                          lstm_dropout=encoder_params['lstm_dropout'],
                          embed_dropout=encoder_params['embed_dropout'])
        encoder.to(encoder_device)
    elif encoder_name == 'attention_encoder':
        encoder = AttentionEncoder(input_dim=len(text_field.vocab),
                                   hid_dim=encoder_params['hidden_dim'],
                                   n_layers=encoder_params['n_layers'],
                                   n_heads=encoder_params['n_heads'],
                                   pf_dim=encoder_params['pf_dim'],
                                   dropout=encoder_params['dropout'],
                                   device=encoder_device,
                                   encoder_layer=EncoderLayer,
                                   self_attention=SelfAttention,
                                   positionwise_feedforward=PositionwiseFeedforward)

    else:
        raise NotImplementedError(f"Encoder {config['model']['encoder']['name']} not supported")
else:
    raise KeyError('`model` config should contain information about encoder')


if 'decoder' in config['model']:
    decoder_name = config['model']['decoder']['name']
    decoder_params: Dict[str, Union[int, float, bool, Optional[str]]] = config['model']['decoder']['params']
    decoder_device_name = decoder_params.pop('device')
    decoder_device = torch.device(decoder_device_name) if decoder_device_name else DEVICE

    if decoder_name == 'lstm_decoder':


        decoder = Decoder(embedding=embedding, vocab_size=VOCAB_SIZE,
                          lstm_n_layers=decoder_params['lstm_n_layers'],
                          lstm_hidden_size=decoder_params['lstm_hidden_size'],
                          embedding_dim=EMB_DIM,
                          lstm_batch_first=decoder_params['lstm_batch_first'],
                          lstm_bidirectional=decoder_params['lstm_bidirectional'],
                          lstm_dropout=decoder_params['lstm_dropout'],
                          embed_dropout=decoder_params['embed_dropout'])
        decoder.to(decoder_device)
    elif decoder_name == 'attention_decoder':
        decoder = AttentionDecoder(output_dim=len(text_field.vocab),
                                   hid_dim=decoder_params['hidden_dim'],
                                   n_layers=decoder_params['n_layers'],
                                   n_heads=decoder_params['n_heads'],
                                   pf_dim=decoder_params['pf_dim'],
                                   dropout=decoder_params['dropout'],
                                   device=decoder_device,
                                   decoder_layer=DecoderLayer,
                                   self_attention=SelfAttention,
                                   positionwise_feedforward=PositionwiseFeedforward)

    else:
        raise NotImplementedError(f"Decoder {config['model']['decoder']['name']} not supported")
else:
    raise KeyError('`model` config should contain information about decoder')

if config['model']['name'] == 'Seq2SeqSummarizer':
    model = Seq2SeqSummarizer(encoder, decoder, device=DEVICE).to(DEVICE)
elif config['model']['name'] == 'AttentionSeq2Seq':
    model = AttentionSeq2Seq(encoder, decoder, text_field.vocab.stoi[PAD_TOKEN], DEVICE).to(DEVICE)
else:
    raise NotImplementedError(f"Model {config['model']['name']} not supported")

if config['training_params']['criterion']['name'] == 'CrossEntropyLoss':
    loss = CrossEntropyLoss(ignore_index=text_field.vocab.stoi[PAD_TOKEN])
else:
    raise NotImplementedError(f"Loss {config['training_params']['criterion']['name']} not supported")

if config['training_params']['optimizer']['name'] == 'Adam':
    optimizer = optim.Adam(model.parameters(),
                           lr=config['training_params']['optimizer']['params']['lr'])
elif config['training_params']['optimizer']['name'] == 'NoamOpt':
    opt_params = config['training_params']['optimizer']['params']
    sub_optimizer_name =  opt_params['optimizer']
    if sub_optimizer_name == 'Adam':
        sub_optimizer = optim.Adam(model.parameters(), lr=opt_params['params']['lr'],
                                   betas=tuple(opt_params['params']['betas']),
                                   eps=opt_params['params']['eps'])
    else:
        raise NotImplementedError(f'Suboptimizer {sub_optimizer_name} is not supported')
    optimizer = NoamOpt(decoder_params['hidden_dim'], factor=opt_params['factor'], warmup=opt_params['warmup'],
                        optimizer=sub_optimizer)

else:
    raise NotImplementedError(f"Loss {config['training_params']['criterion']['name']} not supported")


def train(model, train_data, optimizer, criterion, clip, teacher_forcing_ratio):
    model.train()

    epoch_loss = 0
    i = 0

    with tqdm(bar_format='{postfix[0]} {postfix[2][iter]}} {postfix[1]}: {postfix[2][loss]}',
              postfix=['Training iter:', 'Loss', dict(loss=0, iter=0)]) as t:
        for i, data in enumerate(train_data):
            try:
                (x_train, x_len), (y_train, y_len) = data.text, data.title

                x_len, x_idx = x_len.sort(0, descending=True)
                x_train = x_train[x_idx, :]
                y_len = y_len[x_idx]
                y_train = y_train[x_idx, :]

                optimizer.zero_grad()


                if config['model']['name'] == 'Seq2SeqSummarizer':
                    output = model.forward(x_train, y_train, teacher_forcing_ratio=teacher_forcing_ratio, input_lenght=x_len,
                                           output_length=y_len)
                else:
                    output = model.forward(x_train, y_train)

                y_true = y_train[1:, :].contiguous().view(-1)
                y_pred = output[1:].view(-1, output.shape[2])
                loss = criterion(y_pred, y_true)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

                optimizer.step()

                epoch_loss += loss.item()

                t.postfix[2]['loss'] = loss.item()
            except RuntimeError as e:
                print(e)
            t.postfix[2]['iter'] = i
            t.update()

            if DEVICE.type == 'cuda':
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
            (x_val, x_len), (y_val, y_len) = data.text, data.title
            x_len, x_idx = x_len.sort(0, descending=True)
            x_val = x_val[x_idx, :]
            y_len = y_len[x_idx]
            y_val = y_val[x_idx, :]

            if config['model']['name'] == 'Seq2SeqSummarizer':
                output = model.forward(x_val, y_val, teacher_forcing_ratio=0,
                                       input_lenght=x_len,
                                       output_length=y_len)
            else:
                output = model.forward(x_val, y_val)

            y_true = y_val[1:, :].contiguous().view(-1)
            y_pred = output[1:].view(-1, output.shape[2])
            loss = criterion(y_pred, y_true)

            epoch_loss += loss.item()

            if DEVICE.type == 'cuda':
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
