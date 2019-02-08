import re
from typing import Union, List


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



