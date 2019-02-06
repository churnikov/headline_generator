import os

from gensim.summarization.summarizer import summarize
from tqdm import tqdm

from loader import json_iterator

def summarize_and_save(model_params, file_path, pred_file_name, ref_file_name):
    out_pred = open(pred_file_name, 'w')
    out_ref = open(ref_file_name, 'w')
    for jsn in tqdm(json_iterator(file_path), 'Predicting'):
        reference: str = jsn['title']
        pred: str = summarize(jsn['text'], **model_params)
        if not pred:
            pred = 'none'
        out_pred.write(pred.replace('\n', '\\n') + '\n')
        out_ref.write(reference + '\n')
