# TODO licence from easse

from typing import List
from typing import Union

import sacrebleu
import sacremoses


def normalize(sentence, lowercase: bool = True, tokenizer: str = '13a', return_str: bool = True):
    if lowercase:
        sentence = sentence.lower()

    if tokenizer in ['13a', 'intl']:
        normalized_sent = sacrebleu.TOKENIZERS[tokenizer]()(sentence)
    elif tokenizer == 'moses':
        normalized_sent = sacremoses.MosesTokenizer().tokenize(sentence, return_str=True, escape=False)
    elif tokenizer == 'penn':
        normalized_sent = sacremoses.MosesTokenizer().penn_tokenize(sentence, return_str=True)
    else:
        normalized_sent = sentence

    if not return_str:
        normalized_sent = normalized_sent.split()

    return normalized_sent


def all_norms(sentences: Union[str, List[str]]):
    tokenizers = ['13a']
    if type(sentences) == str:
        for x in tokenizers:
            sentences = normalize(sentences, tokenizer=x, lowercase=False)

    else:
        output = []
        for sent in sentences:
            for x in tokenizers:
                sent = normalize(sent, tokenizer=x, lowercase=False)
            output.append(sent)
        sentences = output

    return sentences
