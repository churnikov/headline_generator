import os
from typing import Callable, Optional

from gensim.summarization.summarizer import summarize
from gensim.summarization.textcleaner import split_sentences
from tqdm import tqdm

from loader import json_iterator

def summarize_and_save(model_params, file_path, pred_file_name, ref_file_name, preprocessor: Optional[Callable]=None):
    out_pred = open(pred_file_name, 'w')
    out_ref = open(ref_file_name, 'w')
    for jsn in tqdm(json_iterator(file_path), 'Predicting'):
        reference: str = jsn['title']

        text = jsn['text']
        if preprocessor is not None:
            text = preprocessor(text)

        if len(split_sentences(text)) > 1:
            try:
                pred: str = summarize(text, **model_params)
            except ValueError:
                print(f'could not processes {jsn}')
        else:
            pred: str = text
        if not pred:
            pred = 'none'
        out_pred.write(pred.replace('\n', '\\n') + '\n')
        out_ref.write(reference + '\n')
