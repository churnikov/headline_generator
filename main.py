import argparse
import json
import os

import rouge
import torch

from baselines import *
from preprocessing import *


def train(model, train_data, optimizer, criterion, clip, device, teacher_forcing_ratio):
    model.train()

    epoch_loss = 0
    i = 1

    with tqdm(bar_format='{postfix[0]} {postfix[3][iter]}/{postfix[2]} {postfix[1]}: {postfix[1][loss]}',
              postfix=['Training iter:', 'Loss', dict(loss=0, iter=0)]) as t:
        for i, (x_train, y_train) in enumerate(train_data):
            optimizer.zero_grad()

            x_train, y_train = x_train.to(device), y_train.to(device)

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


parser = argparse.ArgumentParser()
parser.add_argument('--config_path')

args = parser.parse_args()

with open(args.config_path, 'r') as f:
    config = json.load(f)

model_name = config['model']['name']
model_params = config['model']['params']

file_to_process = config['input']['file_path']

out_dir = config['results']['output_dir']
experiment_number = config['results']['experiment_number']
pred_file_name_suffix = config['results']['pred_file_name_suffix']
ref_file_name_suffix = config['results']['ref_file_name_suffix']

prep_name = config['preprocessing']['name']
if prep_name == 'BasicHtmlPreprocessor':
    preprocessor = BasicHtmlPreprocessor()
else:
    raise NotImplementedError(f'Preprocessor {prep_name} not supported')

save_path = os.path.join(out_dir, model_name, f'{model_params}', f'experiment_{experiment_number}')
if not os.path.exists(save_path):
    os.makedirs(save_path)

pred_test_file_name = os.path.join(save_path, f'test_{pred_file_name_suffix}.txt')
ref_test_file_name = os.path.join(save_path, f'test_{ref_file_name_suffix}.txt')

if model_name == 'textrank':
    summarizer = GensimSummarizer(model_params, file_to_process, pred_test_file_name, ref_test_file_name, preprocessor)
    summarizer.summarize()
if model_name == 'first_sentence':
    summarizer = ExtractFirstFullSentence(model_params, file_to_process, pred_test_file_name, ref_test_file_name,
                                          preprocessor)
    summarizer.summarize()
if model_name == 'Seq2SeqSummarizer':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if config['model']['embedding']['name'] == 'bpe':
        bpe_tokenizer = BPETokenizer(**config['model']['embedding']['params'])
        embedding = nn.Embedding.from_pretrained(torch.tensor(bpe_tokenizer.bpe.vectors))
    else:
        raise NotImplementedError(f'Embedding {config["model"]["embedding"]["name"]} not yet implemented')

    if config['model']['encoder']['name'] == 'lstm_encoder':
        encoder = Encoder(embedding=embedding, **config['model']['encoder'])
    else:
        raise NotImplementedError(f'Encoder {config["model"]["encoder"]["name"]} not yet implemented')

    if config['model']['decoder']['name'] == 'lstm_decoder':
        decoder = Decoder(embedding=embedding, vocab_size=config['model']['embedding']['params']['vocab_size'],
                          **config['model']['decoder'])
    else:
        raise NotImplementedError(f'Encoder {config["model"]["encoder"]["name"]} not yet implemented')

    embedding.to(device)
    encoder.to(device)
    decoder.to(device)
    summarizer = Seq2SeqSummarizer(encoder, decoder, device=device).to(device)


else:
    raise NotImplementedError(f'Model {model_name} not yet implemented')

r = rouge.FilesRouge(pred_test_file_name, ref_test_file_name)

scores = r.get_scores(avg=True)

print(scores)
with open(os.path.join(save_path, 'scores.json'), 'w') as f:
    json.dump(scores, f, indent=2)

with open(os.path.join(save_path, 'config.json'), 'w') as f:
    json.dump(config, f, indent=2)

