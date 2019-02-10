import argparse
import json
import os

import rouge
import torch

from baselines import *
from preprocessing import *

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

    summarizer = Seq2SeqSummarizer(encoder, decoder, device=device)


else:
    raise NotImplementedError(f'Model {model_name} not yet implemented')

r = rouge.FilesRouge(pred_test_file_name, ref_test_file_name)

scores = r.get_scores(avg=True)

print(scores)
with open(os.path.join(save_path, 'scores.json'), 'w') as f:
    json.dump(scores, f, indent=2)

with open(os.path.join(save_path, 'config.json'), 'w') as f:
    json.dump(config, f, indent=2)

