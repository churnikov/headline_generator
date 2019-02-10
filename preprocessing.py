import re
from typing import Union, List

from bpemb import BPEmb


class BasicHtmlPreprocessor:
    """replace all html tags end entities with space"""

    def __init__(self):
        self.regexp = re.compile(r'&[\w\d];+|<([^>]+)>')

    def transform(self, text: Union[str, List[str]]):
        if type(text) == str:
            new_text = text
            for r in self.regexp.finditer(text):
                new_text = new_text.replace(r[0], ' ')

            new_text = ' '.join([' '.join([t for t in sent.split(' ') if t])
                                 for sent in new_text.lower().strip().split('\n')])

            return new_text
        elif type(text) == list:
            return [self.transform(t) for t in text]
        else:
            raise TypeError(f'Type {type(text)} is not supported. `text` should be `list` or `str`')


class BPETokenizer:
    """Use byte pair encoding to transform text"""

    def __init__(self, lang='ru', pretrained=True, vocab_size=100000, dim=300):
        self.lang = lang
        self.pretrained = pretrained
        self.bpe = BPEmb(lang=self.lang, vs=vocab_size, dim=dim, vs_fallback=True)

    def fit(self, text):
        raise NotImplementedError('fit is not supported')

    def transform(self, text: Union[str, List[str]], get_ids=True):
        if get_ids:
            return self.bpe.encode_ids_with_bos_eos(text)
        else:
            return self.bpe.encode_with_bos_eos(text)
