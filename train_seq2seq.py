import csv
import math
import sys
from collections import Counter

import torch
from torch import nn
from torch import optim
from bpemb import BPEmb
from torch.nn import CrossEntropyLoss
from torchtext.data import Field, Example, Dataset, BucketIterator
from torchtext.vocab import Vocab
from tqdm import tqdm

from baselines import Encoder, Decoder, Seq2SeqSummarizer


def train(model, train_data, optimizer, criterion, clip, device, teacher_forcing_ratio):
    model.train()

    epoch_loss = 0
    i = 1

    with tqdm(bar_format='{postfix[0]} {postfix[3][iter]}/{postfix[2]} {postfix[1]}: {postfix[1][loss]}',
              postfix=['Training iter:', 'Loss', dict(loss=0, iter=0)]) as t:
        for i, (x_train, y_train) in enumerate(train_data):
            optimizer.zero_grad()

            output = model.forward(x_train, y_train, teacher_forcing_ratio)

            y_true = y_train[1:].view(-1)
            loss = criterion(output[1:].view(-1, output.shape[2]), y_true)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            optimizer.step()

            epoch_loss += loss.item()

            t.postfix[3]['loss'] = loss.item()
            t.postfix[3]['iter'] = i
            t.update()

    return epoch_loss / i


def evaluate(model, validation_data, criterion, device):
    model.eval()

    epoch_loss = 0
    i = 1

    with torch.no_grad():
        for i, (x_val, y_val) in tqdm(enumerate(validation_data), desc='Validating'):

            y_val, y_val = y_val.to(device), y_val.to(device)

            y_true = y_val[1:].view(-1)

            output = model.forward(x_val, y_val, 0)

            loss = criterion(output[1:].view(-1, output.shape[2]), y_true)

            epoch_loss += loss.item()

    return epoch_loss / i


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


csv.field_size_limit(sys.maxsize)

EMB_DIM = 300
VOCAB_SIZE = 100001
BATCH_SIZE = 256

bpe = BPEmb(lang='ru', vs=VOCAB_SIZE-1, dim=EMB_DIM, add_pad_emb=True)
SOS_TOKEN = bpe.BOS_str
EOS_TOKEN = bpe.EOS_str
PAD_TOKEN = '<pad>'

TRAIN_DATA_PATH = 'data/dataset/ria_prep_train.csv'
TEST_DATA_PATH = 'data/dataset/ria_prep_test.csv'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

text_field = Field(init_token=SOS_TOKEN, eos_token=EOS_TOKEN, tokenize=bpe.encode, pad_token=PAD_TOKEN)
text_field.vocab = Vocab(Counter(bpe.words))


def lazy_examples(csv_source):
    with open(csv_source) as f:
        reader = csv.reader(f)
        next(reader)
        for text, title in reader:
            yield Example.fromlist([text, title], [('text', text_field), ('title', text_field)])


train_dataset = Dataset(lazy_examples(TRAIN_DATA_PATH), [('text', text_field), ('title', text_field)])
iterator = BucketIterator(
    train_dataset, batch_size=BATCH_SIZE, sort_key=lambda x: len(x.text), shuffle=False,
    device=DEVICE)

embedding = nn.Embedding(VOCAB_SIZE, EMB_DIM)

encoder = Encoder(embedding=embedding, **{
    "lstm_n_layers": 4,
    "lstm_hidden_size": 512,
    "lstm_batch_first": False,
    "lstm_bidirectional": True,
    "lstm_dropout": 0.5,
    "embed_dropout": 0.5
})

decoder = Decoder(embedding=embedding, vocab_size=VOCAB_SIZE, **{
    "lstm_n_layers": 4,
    "lstm_hidden_size": 512,
    "lstm_batch_first": False,
    "lstm_bidirectional": True,
    "lstm_dropout": 0.5,
    "embed_dropout": 0.5
})

embedding.to(DEVICE)
encoder.to(DEVICE)
decoder.to(DEVICE)
summarizer = Seq2SeqSummarizer(encoder, decoder, device=DEVICE).to(DEVICE)

loss = CrossEntropyLoss(ignore_index=text_field.vocab.stoi[PAD_TOKEN])
optimizer = optim.Adam(summarizer.parameters())


# best_valid_loss = float('inf')

for epoch in range(10):

    train_loss = train(summarizer, iterator, optimizer, loss, 1, DEVICE, 0.5)
    # valid_loss = evaluate(summarizer, VALIDATION_DATA, criterion)

    # for t in TEST_DATA:
    #     ex = t.x[:, 0].unsqueeze(0)
    #     break
    #
    # res = predict(model, ex, MAX_LEN)
    # print(''.join([id2word[idx] for idx in ex.detach().numpy()]), res)

    # if valid_loss < best_valid_loss:
    #     best_valid_loss = valid_loss
    torch.save(summarizer.state_dict(), 'model.pt')

    print(
        f'| Epoch: {epoch+1:03} | Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
