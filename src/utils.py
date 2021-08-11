from __future__ import unicode_literals, print_function, division
import time
import math
import numpy as np
from io import open
import unicodedata
import string
import pickle
import re
import random
# from config import model_config as config
import json
import torch
import torch.nn as nn
from torch.utils import data
from torch import optim
import torch.nn.functional as F
from pytorch_models.BiRNN import BiRNN
# from nltk.parse.corenlp import CoreNLPServer
from nltk.parse.corenlp import CoreNLPParser
import os
from model.FKGL import sentence_fre, sentence_fkgl
import spacy
from sklearn.metrics.pairwise import cosine_similarity
# from allennlp.modules.elmo import batch_to_ids
from itertools import chain
import nltk
from nltk.corpus import wordnet as wn
import pyinflect
from pyinflect import getAllInflections, getInflection
# from tokenizers import SentencePieceBPETokenizer
import copy
from normalizing import *  # CHANGED
from types import ModuleType
from importlib import reload
from tqdm import tqdm
from pathlib import Path
from easse.sari import corpus_sari, get_corpus_sari_operation_scores


import transformers
from transformers import DebertaForSequenceClassification, Trainer, TrainingArguments, DebertaTokenizerFast
from pattern.en import lexeme
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict

conf_file = open("config.json", "r")
config = json.load(conf_file)
conf_file.close()

print(config)
'''from allennlp.modules.elmo import Elmo, batch_to_ids
options_file = ('https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json')
weight_file = ('https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5')
elmo = Elmo(options_file, weight_file, 2, dropout=0)'''
parser = CoreNLPParser('http://localhost:9000')
# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load("en_core_web_lg")
from spacy.tokenizer import Tokenizer

nlp.tokenizer = Tokenizer(nlp.vocab)
# to make sure spacy always tokenizes just on space and not other tokens
# if not done then number of tokens will be different which would result in issues
# when using pos and dep tags for language model
if config['lexical_simplification']:
    import gensim
    import gensim.downloader as api
    from gensim.models import Word2Vec

    print('loading glove')
    glove_model300 = api.load('glove-wiki-gigaword-300')
    print('loading word2vec')
    word2vec = api.load('word2vec-google-news-300')
    our_word2vec = Word2Vec.load(config['dataset'] + '/Word2vec/word2vec.model')  # word2vec_src

device = torch.device("cuda:" + str(config['gpu']) if torch.cuda.is_available() else "cpu")

root_comp_simp = "/home/m25dehgh/simplification/complex-classifier"
model_comp_simp = "newsela-auto-high-quality"
path_comp_simp = root_comp_simp + '/results' + '/' + model_comp_simp + "/whole-high-quality/checkpoint-44361/"
comp_simp_class_model = DebertaForSequenceClassification.from_pretrained(path_comp_simp)

root_grammar_checker = "/home/m25dehgh/simplification/grammar-checker"
model_name_grammar_checker = "deberta-base-cola"
path = root_grammar_checker + '/results' + '/' + model_name_grammar_checker + "/checkpoint-716"
model_grammar_checker = DebertaForSequenceClassification.from_pretrained(path)
tokenizer_deberta = DebertaTokenizerFast.from_pretrained('microsoft/deberta-base')

SOS_token = 1
EOS_token = 2
PAD_token = 0
UNK_token = 3


# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# special_tokens_dict = {'sep_token': '< >'}
# num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
# model = GPT2LMHeadModel.from_pretrained('gpt2')
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS", 3: "UNK"}
        self.n_words = 4  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addVocab(self, vocab):
        for word in vocab:
            self.addWord(word)

    def addWord(self, word):
        # should check the count and say if it is less than 3 then should be converted to UNK
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    # Turn a Unicode string to plain ASCII, thanks to
    # https://stackoverflow.com/a/518232/2809427
    def unicodeToAscii(s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    # Lowercase, trim, and remove non-letter characters

    def normalizeString(s):
        s = Lang.unicodeToAscii(s)
        # s = Lang.unicodeToAscii(s.lower().strip())
        # s = re.sub(r"([.!?])", r" \1", s)
        # s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s

    def getSentences(src, reverse):
        sent = {}
        x_unique = []
        x = []
        for i in range(len(src) - 1):
            if reverse:
                x.append(reverse_sent(Lang.normalizeString(src[i])))
            else:
                x.append(Lang.normalizeString(src[i]))
            if src[i] not in sent:
                sent[src[i]] = 1
                if reverse:
                    x_unique.append(reverse_sent(Lang.normalizeString(src[i])))
                else:
                    x_unique.append(Lang.normalizeString(src[i]))
        return x, x_unique

    def readLangs(dataset):
        print("Reading lines...")
        train_src = []
        valid_src = []
        test_src = []
        train_dst = []
        valid_dst = []
        test_dst = []
        # Read train file

        if dataset == 'Wikilarge':
            print('loading Wikilarge data')
            train_src = open('/home/m25dehgh/simplification/datasets/wikilarge/data-simplification/wikilarge/wiki'
                             '.full.aner.ori.train.src', encoding='utf-8').read().split('\n')
            train_dst = open('/home/m25dehgh/simplification/datasets/wikilarge/data-simplification/wikilarge/wiki'
                             '.full.aner.ori.train.dst', encoding='utf-8').read().split('\n')
            # valid_src = open('/home/m25dehgh/simplification/datasets/wikilarge/data-simplification/wikilarge/wiki'
            #                  '.full.aner.ori.valid.src', encoding='utf-8').read().split('\n')
            # valid_dst = open('/home/m25dehgh/simplification/datasets/wikilarge/data-simplification/wikilarge/wiki'
            #                  '.full.aner.ori.valid.dst', encoding='utf-8').read().split('\n')
            # test_src = open('/home/m25dehgh/simplification/datasets/wikilarge/data-simplification/wikilarge/wiki.full'
            #                 '.aner.ori.test.src', encoding='utf-8').read().split('\n')
            # test_dst = open('/home/m25dehgh/simplification/datasets/wikilarge/data-simplification/wikilarge/wiki.full'
            #                 '.aner.ori.test.dst', encoding='utf-8').read().split('\n')

            # print("Loading Asset instead of Wikilarge data")  # changed
            # train_src = open('/home/m25dehgh/simplification/asset/dataset/asset.test.orig',
            #                  encoding='utf-8').read().split('\n')
            # train_dst = open('/home/m25dehgh/simplification/asset/dataset/asset.test.simp.0',
            #                  encoding='utf-8').read().split('\n')
            valid_src = open('/home/m25dehgh/simplification/datasets/asset/dataset/asset.valid.orig',
                             encoding='utf-8').read().split('\n')
            valid_dst = open('/home/m25dehgh/simplification/datasets/asset/dataset/asset.valid.simp.0',
                             encoding='utf-8').read().split('\n')
            test_src = open('/home/m25dehgh/simplification/datasets/asset/dataset/asset.test.orig',
                            encoding='utf-8').read().split('\n')
            test_dst = open('/home/m25dehgh/simplification/datasets/asset/dataset/asset.test.simp.0',
                            encoding='utf-8').read().split('\n')

            # train_src = open('/home/m25dehgh/simplification/Edit-Unsup-TS/src/turkcorpus/test.8turkers.tok.norm',
            #                  encoding='utf-8').read().split('\n')
            # train_src = open('/home/m25dehgh/simplification/datasets/wikilarge/data-simplification/wikilarge/wiki'
            #                  '.full.aner.ori.train.src', encoding='utf-8').read().split('\n')

            # train_dst = open('/home/m25dehgh/simplification/Edit-Unsup-TS/src/turkcorpus/test.8turkers.tok.turk.0',
            #                  encoding='utf-8').read().split('\n')
            # train_dst = open('/home/m25dehgh/simplification/datasets/wikilarge/data-simplification/wikilarge/wiki.'
            #                  'full.aner.ori.train.dst', encoding='utf-8').read().split('\n')
            # valid_src = open('/home/m25dehgh/simplification/Edit-Unsup-TS/src/turkcorpus/test.8turkers.tok.specified.norm', encoding='utf-8').read().split('\n')  # changed
            # valid_dst = open('/home/m25dehgh/simplification/Edit-Unsup-TS/src/turkcorpus/test.8turkers.tok.turk.0', encoding='utf-8').read().split('\n')
            # test_src = open('/home/m25dehgh/simplification/Edit-Unsup-TS/src/turkcorpus/test.8turkers.tok.specified.norm', encoding='utf-8').read().split('\n')   # changed
            # test_dst = open('/home/m25dehgh/simplification/Edit-Unsup-TS/src/turkcorpus/test.8turkers.tok.turk.0', encoding='utf-8').read().split('\n')

        elif dataset == 'Newsela':
            print('loading Newsela data')
            train_src = open(
                '/home/m25dehgh/simplification/datasets/newsela/dhruv-newsela/V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.ori.train.src',
                encoding='utf-8').read().split('\n')
            train_dst = open(
                '/home/m25dehgh/simplification/datasets/newsela/dhruv-newsela/V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.ori.train.dst',
                encoding='utf-8').read().split('\n')
            valid_src = open(
                '/home/m25dehgh/simplification/datasets/newsela/dhruv-newsela/V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.ori.valid.src',
                encoding='utf-8').read().split('\n')
            valid_dst = open(
                '/home/m25dehgh/simplification/datasets/newsela/dhruv-newsela/V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.ori.valid.dst',
                encoding='utf-8').read().split('\n')
            test_src = open(
                '/home/m25dehgh/simplification/datasets/newsela/dhruv-newsela/V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.ori.test.src',
                encoding='utf-8').read().split('\n')
            test_dst = open(
                '/home/m25dehgh/simplification/datasets/newsela/dhruv-newsela/V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.ori.test.dst',
                encoding='utf-8').read().split('\n')

            # print("Loading Asset instead of Newsela data")  # changed
            # train_src = open('/home/m25dehgh/simplification/datasets/asset/dataset/asset.test.orig',
            #                  encoding='utf-8').read().split('\n')
            # train_dst = open('/home/m25dehgh/simplification/datasets/asset/dataset/asset.test.simp.0',
            #                  encoding='utf-8').read().split('\n')
            # valid_src = open('/home/m25dehgh/simplification/datasets/asset/dataset/asset.valid.orig',
            #                  encoding='utf-8').read().split('\n')
            # valid_dst = open('/home/m25dehgh/simplification/datasets/asset/dataset/asset.valid.simp.0',
            #                  encoding='utf-8').read().split('\n')
            # test_src = open('/home/m25dehgh/simplification/datasets/asset/dataset/asset.test.orig',
            #                 encoding='utf-8').read().split('\n')
            # test_dst = open('/home/m25dehgh/simplification/datasets/asset/dataset/asset.test.simp.0',
            #                 encoding='utf-8').read().split('\n')

        # changed
        # print("normalizing")
        # train_src = [all_norms(i) for i in train_src]
        # train_dst = [all_norms(i) for i in train_dst]
        valid_src = [all_norms(i) for i in valid_src]
        valid_dst = [all_norms(i) for i in valid_dst]
        test_src = [all_norms(i) for i in test_src]
        test_dst = [all_norms(i) for i in test_dst]
        # changed
        # lexical_simplification
        # train_src = [normalize(s, lowercase=True, tokenizer='moses') for s in train_src]
        # train_dst = [normalize(s, lowercase=True, tokenizer='moses') for s in train_dst]
        # valid_src = [normalize(s, lowercase=True, tokenizer='moses') for s in valid_src]
        # valid_dst = [normalize(s, lowercase=True, tokenizer='moses') for s in valid_dst]
        # test_src = [normalize(s, lowercase=True, tokenizer='moses') for s in test_src]
        # test_dst = [normalize(s, lowercase=True, tokenizer='moses') for s in test_dst]

        train_simple, train_simple_unique = Lang.getSentences(train_dst, config['lm_backward'])
        valid_simple, valid_simple_unique = Lang.getSentences(valid_dst, config['lm_backward'])
        test_simple, test_simple_unique = Lang.getSentences(test_dst, config['lm_backward'])
        train_complex, train_complex_unique = Lang.getSentences(train_src, config['lm_backward'])
        valid_complex, valid_complex_unique = Lang.getSentences(valid_src, config['lm_backward'])
        test_complex, test_complex_unique = Lang.getSentences(test_src, config['lm_backward'])
        output_lang = Lang('simple')
        dep_lang = Lang('dep')
        tag_lang = Lang('tag')
        # this returns a pair of simple, complex
        # but as of now we dont use the complex part in our language model
        # return x_train, y_train, x_valid, y_valid, x_test, y_test, output_lang
        return train_simple, train_simple_unique, valid_simple, valid_simple_unique, test_simple, test_simple_unique, train_complex, train_complex_unique, valid_complex, valid_complex_unique, test_complex, test_complex_unique, output_lang, tag_lang, dep_lang


# this Class is taken from https://gist.github.com/wassname/7fd4c975883074a99864
# Reverse Stemming
class SnowCastleStemmer(nltk.stem.SnowballStemmer):
    """ A wrapper around snowball stemmer with a reverse lookip table """

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self._stem_memory = defaultdict(set)
        # switch stem and memstem
        self._stem = self.stem
        self.stem = self.memstem

    def memstem(self, word):
        """ Wrapper around stem that remembers """
        stemmed_word = self._stem(word)
        self._stem_memory[stemmed_word].add(word)
        return stemmed_word

    def unstem(self, stemmed_word):
        """ Reverse lookup """
        return sorted(self._stem_memory[stemmed_word], key=len)


def create_reverse_stem():
    """creates reverse stem for all words in the dictionary
    It should be called once in the start of program
    """

    stemmer = SnowCastleStemmer('english')
    dictionary = nltk.corpus.words.words("en")

    for vocab in dictionary:
        stemmer.stem(vocab)

    return stemmer


def reverse_sent(sent):
    s = sent.split(' ')
    s = s[::-1]
    s = ' '.join(s)
    return s


def load_word_embeddings(embedding_type, embedding_dim, ver):
    if embedding_type == 'glove':
        embeddings_index = dict()
        f = open('/home/m25dehgh/simplification/pretrained_models/glove/' + ver + str(embedding_dim) + 'd.txt',
                 encoding='utf-8')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        return embeddings_index
    else:
        return None


# given a text, returns a 2d matrix with its word embeddings.
# each column represents the embedding for each word
def get_embedding_matrix(embeddings_index, vocab, embedding_dim):
    oov = []
    if embeddings_index is not None:
        embedding_matrix = np.zeros((len(vocab) + 4, embedding_dim))
        embedding_matrix[0] = np.zeros((1, embedding_dim), dtype='float32')  # PAD
        embedding_matrix[1] = np.random.uniform(low=-0.05, high=0.05, size=(1, embedding_dim))  # SOS
        embedding_matrix[2] = np.random.uniform(low=-0.05, high=0.05, size=(1, embedding_dim))  # EOS
        embedding_matrix[3] = np.random.uniform(low=-0.05, high=0.05, size=(1, embedding_dim))  # UNK
        l = 4
        for key in vocab:
            embedding_vector = embeddings_index.get(key)
            if embedding_vector is not None:
                embedding_matrix[l] = embedding_vector
            else:
                oov.append(key)
                # embedding_matrix[l] = np.zeros((1, embedding_dim))
                # initializing it with zeros seems to work better
                embedding_matrix[l] = np.random.uniform(low=-0.05, high=0.05, size=(1, embedding_dim))
            l += 1
        '''print("starting saving in oov file")
        with open('oov' + file_extension + '_glove.txt', 'w') as f:
            for item in oov:
                item = item.encode('utf8')
                f.write("%s\n" % item)
        f.close()
        print('len embedding matrix should be same as vocab')
        print(len(embedding_matrix))'''
        print("OOV count")
        print(len(oov))
        return embedding_matrix
    else:
        embedding_matrix = np.zeros((len(vocab) + 4, embedding_dim))
        embedding_matrix[0] = np.zeros((1, embedding_dim), dtype='float32')  # PAD
        embedding_matrix[1] = np.random.uniform(low=-0.05, high=0.05, size=(1, embedding_dim))  # SOS
        embedding_matrix[2] = np.random.uniform(low=-0.05, high=0.05, size=(1, embedding_dim))  # EOS
        embedding_matrix[3] = np.random.uniform(low=-0.05, high=0.05, size=(1, embedding_dim))  # UNK
        l = 4
        for key in vocab:
            embedding_matrix[l] = np.random.normal(loc=0, scale=0.1, size=(1, embedding_dim))
            l += 1
        return embedding_matrix


def getvocab(pairs, min_freq_inp, min_freq_out, outputvocab, outputword2count):
    # outputvocab = []
    # outputword2count = {}
    for pair in pairs:

        for i in pair.split(' '):
            # since now sentences are not lowered. Doing that now.
            word = i.lower()
            if word not in outputword2count:
                outputword2count[word] = 1
            else:
                outputword2count[word] += 1

    for k, v in outputword2count.items():
        if v >= min_freq_out:
            outputvocab.append(k)
    return outputvocab, outputword2count


def convert_to_sent(sent):
    s = ''
    for i in range(len(sent)):
        if sent[i] not in '':
            s = s + sent[i] + ' '
    return s[:-1]


def getIDF(outputword2count, N):
    idf = {}
    for k, v in outputword2count.items():
        idf[k] = math.log2(N / v)
    return idf


def get_unigram_probability(outputword2count, N):
    unigram_prob = {}
    for k, v in outputword2count.items():
        unigram_prob[k] = v / N
    return unigram_prob


def get_idf_value(idf, word):
    if idf is None:
        return 1
    else:
        if word in idf:
            return idf[word]
        return 16


def get_unigram_prob_value(unigram_prob, word):
    if unigram_prob is None:
        return 1
    else:
        if word in unigram_prob:
            return unigram_prob[word]
        return 1


def filterPair(p):
    return len(p.split(' ')) < config['MAX_LENGTH']


# p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(embedding_dim, freq, ver, dataset, operation):
    train_simple, train_simple_unique, valid_simple, valid_simple_unique, test_simple, test_simple_unique, train_complex, train_complex_unique, valid_complex, valid_complex_unique, test_complex, test_complex_unique, output_lang, tag_lang, dep_lang = Lang.readLangs(
        dataset)
    print("Read %s train sentence pairs" % len(train_simple))
    print("Read %s valid sentence pairs" % len(valid_simple))
    print("Read %s test sentence pairs" % len(test_simple))
    print("Read %s unique train sentence pairs" % len(train_simple_unique))
    print("Read %s unique valid sentence pairs" % len(valid_simple_unique))
    print("Read %s unique test sentence pairs" % len(test_simple_unique))
    '''count = 0
	for i in range(len(train_simple)):
		if count < len(train_simple[i].split(' ')):
			count = len(train_simple[i].split(' '))
	print(count)
	for i in range(len(valid_simple)):
		if count < len(valid_simple[i].split(' ')):
			count = len(valid_simple[i].split(' '))
	print(count)
	for i in range(len(test_simple)):
		if count < len(test_simple[i].split(' ')):
			count = len(test_simple[i].split(' '))
	print(count)
	for i in range(len(train_complex)):
		if count < len(train_complex[i].split(' ')):
			count = len(train_complex[i].split(' '))
	print(count)
	for i in range(len(valid_complex)):
		if count < len(valid_complex[i].split(' ')):
			count = len(valid_complex[i].split(' '))
	print(count)
	for i in range(len(test_complex)):
		if count < len(test_complex[i].split(' ')):
			count = len(test_complex[i].split(' '))
	print(count)'''
    train_simple = filterPairs(train_simple)
    valid_simple = filterPairs(valid_simple)
    test_simple = filterPairs(test_simple)
    train_complex = filterPairs(train_complex)
    valid_complex = filterPairs(valid_complex)
    test_complex = filterPairs(test_complex)
    train_simple_unique = filterPairs(train_simple_unique)
    valid_simple_unique = filterPairs(valid_simple_unique)
    test_simple_unique = filterPairs(test_simple_unique)
    train_complex_unique = filterPairs(train_complex_unique)
    valid_complex_unique = filterPairs(valid_complex_unique)
    test_complex_unique = filterPairs(test_complex_unique)
    print("Trimmed to %s train sentence pairs" % len(train_simple))
    print("Trimmed to %s valid sentence pairs" % len(valid_simple))
    print("Trimmed to %s test sentence pairs" % len(test_simple))
    print("Trimmed to %s unique train sentence pairs" % len(train_simple_unique))
    print("Trimmed to %s unique valid sentence pairs" % len(valid_simple_unique))
    print("Trimmed to %s unique test sentence pairs" % len(test_simple_unique))
    print("Loading/Building vocabulary")
    # vaocab is always made up from all the sentences and not unique since we ahve a min frequency clause

    if os.path.isfile(config['dataset'] + '/outputword2count.npy'):
        print('outputword2count (Vocab) file present')
        outputword2count = np.load(config['dataset'] + '/outputword2count.npy', allow_pickle=True).item()
        with open(config['dataset'] + "/output_vocab.txt", "rb") as fp:  # Unpickling
            output_vocab = pickle.load(fp)
    else:
        print('outputword2count (Vocab) file not present, creating and saving it')
        output_vocab, outputword2count = getvocab(train_simple_unique, freq, freq, [], {})
        with open(config['dataset'] + "/output_vocab.txt", "wb") as fp:
            pickle.dump(output_vocab, fp)
        np.save(config['dataset'] + '/outputword2count.npy', outputword2count)

    tag_vocab = ['$', "''", ',', '-LRB-', '-RRB-', '.', ':', 'ADD', 'AFX', 'CC', 'CD', 'DT', 'EX', 'FW', 'HYPH',
                 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NFP', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS', 'PRP', 'PRP$',
                 'RB',
                 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$',
                 'WRB', 'XX', '_SP',
                 '``']
    dep_vocab = ['ROOT', 'ACL', 'ACOMP', 'ADVCL', 'ADVMOD', 'AGENT', 'AMOD', 'APPOS', 'ATTR', 'AUX', 'AUXPASS',
                 'CASE', 'CC', 'CCOMP', 'COMPOUND', 'CONJ', 'CSUBJ', 'CSUBJPASS', 'DATIVE', 'DEP', 'DET', 'DOBJ',
                 'EXPL',
                 'INTJ', 'MARK', 'META', 'NEG', 'NMOD', 'NPADVMOD', 'NSUBJ', 'NSUBJPASS', 'NUMMOD', 'OPRD', 'PARATAXIS',
                 'PCOMP', 'POBJ', 'POSS', 'PRECONJ', 'PREDET', 'PREP', 'PRT', 'PUNCT', 'QUANTMOD', 'RELCL', 'XCOMP', '',
                 'SUBTOK']
    print("Generating tf-idf file using simple sentences from the training set")
    idf = getIDF(outputword2count, len(train_simple_unique))
    print("Calculating unigram probabilities")
    unigram_prob = get_unigram_probability(outputword2count, len(train_simple_unique))
    '''full_vocab, fullword2count = getvocab(train_simple_unique, freq, freq, [], {})
	full_vocab, fullword2count = getvocab(valid_simple_unique, freq, freq, full_vocab, fullword2count)
	full_vocab, fullword2count = getvocab(test_simple_unique, freq, freq, full_vocab, fullword2count)
	full_vocab, fullword2count = getvocab(train_complex, freq, freq, full_vocab, fullword2count)
	full_vocab, fullword2count = getvocab(valid_complex, freq, freq, full_vocab, fullword2count)
	full_vocab, fullword2count = getvocab(test_complex, freq, freq, full_vocab, fullword2count)'''
    output_lang.addVocab(output_vocab)
    tag_lang.addVocab(tag_vocab)
    dep_lang.addVocab(dep_vocab)
    if not config['elmo']:
        embeddings_index = load_word_embeddings('glove', embedding_dim, ver)
        output_embedding_weights = get_embedding_matrix(embeddings_index, output_vocab, embedding_dim)
    else:
        output_embedding_weights = get_embedding_matrix(None, output_vocab, embedding_dim)
    tag_embedding_weights = get_embedding_matrix(None, tag_vocab, config['tag_dim'])
    dep_embedding_weights = get_embedding_matrix(None, dep_vocab, config['dep_dim'])

    print("Total vocabulary size:")
    print(output_lang.name, output_lang.n_words)
    # return input_lang, output_lang, train_pairs, valid_pairs, test_pairs, [], []
    if operation == 'sample' or operation == 'train_encoder':
        return idf, unigram_prob, output_lang, tag_lang, dep_lang, train_simple, valid_simple, test_simple, train_complex, valid_complex, test_complex, output_embedding_weights, tag_embedding_weights, dep_embedding_weights

    elif operation == 'train_lm':
        return idf, unigram_prob, output_lang, tag_lang, dep_lang, train_simple_unique, valid_simple_unique, test_simple_unique, train_complex_unique, valid_complex_unique, test_complex_unique, output_embedding_weights, tag_embedding_weights, dep_embedding_weights


def reverse_file(split_type):
    pos_file_backward = 'Pos' + split_type + '_backward.txt'
    dep_file_backward = 'Dep' + split_type + '_backward.txt'
    pos_file = 'Pos' + split_type + '.txt'
    dep_file = 'Dep' + split_type + '.txt'
    pos = open(pos_file, encoding='utf-8').read().split('\n')
    pos_sent = pos[:-1]
    dep = open(dep_file, encoding='utf-8').read().split('\n')
    dep_sent = dep[:-1]
    # print(len(pos_sent))
    # print(len(dep_sent))
    with open(pos_file_backward, "a") as pos:
        with open(dep_file_backward, "a") as dep:
            for i in range(len(pos_sent)):
                a = reverse_sent(pos_sent[i])
                pos.write(a + "\n")
            for i in range(len(dep_sent)):
                a = reverse_sent(dep_sent[i])
                dep.write(a + "\n")


def load_syntax_file(sentences, split_type, lm_backward):
    if lm_backward:
        pos_file = config['dataset'] + '/Pos' + split_type + '_backward.txt'
        dep_file = config['dataset'] + '/Dep' + split_type + '_backward.txt'
    else:
        pos_file = config['dataset'] + '/Pos' + split_type + '.txt'
        dep_file = config['dataset'] + '/Dep' + split_type + '.txt'
    if os.path.isfile(pos_file) and os.path.isfile(dep_file):
        print("Sytax files present, loading it...")
        pos = open(pos_file, encoding='utf-8').read().split('\n')
        pos_sent = pos[:-1]
        dep = open(dep_file, encoding='utf-8').read().split('\n')
        # print(len(dep))
        dep_sent = dep[:-1]
    # print(len(dep_sent))

    else:
        print("Sytax files absent, creating and saving it...")
        pos_sent = []
        dep_sent = []
        with open(pos_file, "a") as pos:
            with open(dep_file, "a") as dep:
                for i in range(len(sentences)):
                    doc = nlp(sentences[i])
                    a = convert_to_sent([(tok.dep_).upper() for tok in doc])
                    if lm_backward:
                        a = reverse_sent(a)
                    dep_sent.append(a)
                    dep.write(a + "\n")
                    a = convert_to_sent([(tok.tag_).upper() for tok in doc])
                    if lm_backward:
                        a = reverse_sent(a)
                    pos_sent.append(a)
                    pos.write(a + "\n")

    return pos_sent, dep_sent


def pad_sequences(x, max_len, p):
    if p:
        padded = torch.zeros((max_len, 50), dtype=torch.long, device=device)
    else:
        padded = torch.zeros((max_len), dtype=torch.long, device=device)
    if len(x) > max_len:
        padded[:] = x[:max_len]
    else:
        padded[:len(x)] = x
    return padded


def getword(lang, word):
    # print(word)
    if word in lang.word2index:
        return lang.word2index[word]
    else:
        return UNK_token  # index number of UNK


def indexesFromSentence(lang, sentence):
    # print(sentence.split(' '))
    return [getword(lang, word) for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    # indexes = [SOS_token]
    # print(indexes)
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    indexes = [SOS_token] + indexes
    return torch.tensor(indexes, dtype=torch.long, device=device)


def tensorsFromPair(pair, output_lang):
    input_tensor = tensorFromSentence(output_lang, pair)
    # target_tensor = tensorFromSentence(output_lang, pair[1])
    return input_tensor


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def get_len(itensor):
    a = []
    mask = []
    for i in itensor:
        count = 0
        temp = []
        # print(i)
        for j in i:
            if j == 0:
                break
            else:
                count += 1
        a.append(count)
    return torch.tensor(a, device=device)


def get_mask(tensor_len):
    a = torch.max(tensor_len)
    t = []
    for i in tensor_len:
        temp = []
        for k in range(i):
            temp.append(1)
        for k in range(a - i):
            temp.append(0)
        t.append(temp)
    return torch.tensor(t, device=device)


def removeUNK(sent):
    # print(sent[sent!=UNK_token])
    return sent[sent != UNK_token]


def removeUNKforall(input_tensor, tag_tensor, dep_tensor):
    # print(input_tensor)
    # print(tag_tensor)
    # print(dep_tensor)
    for i in range(len(input_tensor)):
        if input_tensor[i] == EOS_token:
            break
        elif input_tensor[i] == UNK_token:
            tag_tensor[i] = UNK_token
            dep_tensor[i] = UNK_token
        '''input_sent[0] = input_sent[0][input_sent[0]!=UNK_token]
		input_sent[0] = pad_sequences(input_sent[0], config['num_steps'])
		tag_tensor[0] = tag_tensor[0][tag_tensor[0]!=UNK_token]
		tag_tensor[0] = pad_sequences(tag_tensor[0], config['num_steps'])
		dep_tensor[0] = dep_tensor[0][dep_tensor[0]!=UNK_token]
		dep_tensor[0] = pad_sequences(dep_tensor[0], config['num_steps'])'''
    input_tensor = removeUNK(input_tensor)
    tag_tensor = removeUNK(tag_tensor)
    dep_tensor = removeUNK(dep_tensor)
    # print(input_tensor)
    # print(tag_tensor)
    # print(dep_tensor)
    return input_tensor, tag_tensor, dep_tensor


def calculateProbabilitySentence(prob, sent):
    min_val = 0.0
    # print(prob)
    worst_three = 0.0
    s = []
    l = getLength(sent[0]) - 2
    p = 0
    for i in range(l):
        s = s + [prob[i][sent[0][i + 1]]]
        p += prob[i][sent[0][i + 1]]
        worst_three = prob[i][sent[0][i - 1]] + prob[i][sent[0][i]] + prob[i][sent[0][i + 1]]
        if min_val > worst_three:
            min_val = worst_three
        # print(i)
    if p.item() < 0 and p.item() > -1:
        print(s)

    return p.item() / l, min_val


def getLength(t):
    # print(t)
    c = 0
    for i in t:
        if i.item() != 0:
            c += 1
    # print(c)
    return c


def calculateLossWord(decoder, output_tensor, lang, p):
    # calculate language model score for the standard language model
    decoder.eval()
    # print(output_tensor)
    with torch.no_grad():
        criterion = nn.NLLLoss(ignore_index=0)
        decoder_hidden = decoder.initHidden(1)
        # encoder_hidden = encoder.initHidden(1, True)
        # output_tensor = output_tensor.unsqueeze(0)
        # print(input_tensor)
        c = 0
        a = []
        for i in output_tensor[0]:
            if i.item() == 0:
                a.append(0)
            else:
                c += 1
                a.append(1)
        output_tensor_len = torch.tensor([c], device=device)
        output_tensor_mask = get_mask(output_tensor_len)
        batch_size = 1
        decoder_input = output_tensor.narrow(1, 0, 1)
        # decoder_input = torch.full((batch_size, 1), SOS_token, device=device, dtype=torch.int64)
        decoded_words = []
        loss = 0
        prob = []
        # print('output_tensor len')
        # print(output_tensor_len.item()-1)
        for di in range(output_tensor_len.item() - 1):
            # -2 because of removing the SOS and EOS tokens
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, False)
            indices = torch.tensor([di + 1], device=device)
            # print(output_tensor)
            target_t = torch.index_select(output_tensor, 1, indices)
            # print(target_t)
            # print(target_t.view(-1))
            loss += criterion(decoder_output[0], target_t.view(-1))
            prob.append(decoder_output[0])
            # a = decoder_output[0]
            '''topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])'''
            decoder_input = target_t
        # decoder_input = topi.squeeze(0).detach()
        # print(loss)
        # print(math.exp(loss))
        b = torch.Tensor(1, output_tensor_len.item() - 1, lang.n_words).to(device)
        return torch.cat(prob, out=b), loss.item() / (output_tensor_len.item() - 1)
        # -2 for SOS and EOS token
    # return decoded_words, math.exp(loss)/max_length


def calculateLoss(decoder, elmo_tensor, output_tensor, tag_tensor, dep_tensor, lang, p):
    decoder.eval()
    # print('inside calculate Loss')
    # print(output_tensor)
    with torch.no_grad():
        criterion = nn.NLLLoss(ignore_index=0)
        decoder_hidden = decoder.initHidden(1)
        # encoder_hidden = encoder.initHidden(1, True)
        # output_tensor = output_tensor.unsqueeze(0)
        # print(input_tensor)
        c = 0
        a = []
        for i in output_tensor[0]:
            if i.item() == 0:
                a.append(0)
            else:
                c += 1
                a.append(1)
        output_tensor_len = torch.tensor([c], device=device)
        output_tensor_mask = get_mask(output_tensor_len)
        batch_size = 1
        if config['elmo']:
            decoder_input = elmo_tensor.narrow(1, 0, 1)
        else:
            decoder_input = output_tensor.narrow(1, 0, 1)
        tag_input = tag_tensor.narrow(1, 0, 1)
        dep_input = dep_tensor.narrow(1, 0, 1)
        # decoder_input = torch.full((batch_size, 1), SOS_token, device=device, dtype=torch.int64)
        decoded_words = []
        loss = 0
        prob = []
        # print('output_tensor len')
        # print(output_tensor_len.item()-1)
        for di in range(output_tensor_len.item() - 1):
            # -2 because of removing the SOS and EOS tokens
            decoder_output, decoder_hidden = decoder(decoder_input, tag_input, dep_input, decoder_hidden, False)
            indices = torch.tensor([di + 1], device=device)
            # print(output_tensor)
            target_t = torch.index_select(output_tensor, 1, indices)
            # print(target_t)
            # print(target_t.view(-1))
            loss += criterion(decoder_output[0], target_t.view(-1))
            prob.append(decoder_output[0])
            # a = decoder_output[0]
            '''topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])'''
            if config['elmo']:
                decoder_input = torch.index_select(elmo_tensor, 1, indices)
            else:
                decoder_input = target_t
            # decoder_input = topi.squeeze(0).detach()
    # print(loss)
    # print(math.exp(loss))
    b = torch.Tensor(1, output_tensor_len.item() - 1, lang.n_words).to(device)
    # print(math.exp(loss.item()/(output_tensor_len.item()-1)))
    return torch.cat(prob, out=b), loss.item() / (output_tensor_len.item() - 1)
    # -2 for SOS and EOS token


# return decoded_words, math.exp(loss)/max_length


def get_sentence_probability(lm_forward, elmo_tensor, input_sent_tensor, tag_tensor, dep_tensor, input_lang, input_sent,
                             unigram_prob):
    prob, _ = calculateLoss(lm_forward, elmo_tensor, input_sent_tensor, tag_tensor, dep_tensor, input_lang, False)
    prob, worst_three = calculateProbabilitySentence(prob, input_sent_tensor)
    if config['SLOR']:
        slor = prob - calcluate_unigram_probability(input_sent, unigram_prob, input_lang)
        return 1000 * math.exp(slor)  # *1000*math.exp(worst_three)
    return 1000 * math.exp(prob)  # *1000*math.exp(worst_three)


def convert_to_blue(sent):
    s = []
    for i in sent.split(' '):
        s.append(i)
    # print(s)
    return s


def avg_embedding(sentence, idf):
    # print(sentence)
    doc = nlp(sentence)
    # print(doc)
    sp = sentence.split(' ')
    # print(sp)
    l = len(sp)
    a = torch.zeros(300)
    # print("inside avg_embedding")
    # a = torch.zeros(1, config['embedding_dim'])
    for i in range(l):
        if sp[i] != 'asfsf':
            mu = get_idf_value(idf, sp[i])
            # print(doc[i])
            a = a + torch.from_numpy(doc[i].vector * mu)
    a /= (l + 0.0001)
    return a


def avg_embedding_elmo(sentence1, sentence2, idf):
    # changed
    raise ValueError('elmo removed, causing error because you are using elmo in config')

    # sentences = [sentence1.split(' '), sentence2.split(' ')]
    # character_ids = batch_to_ids(sentences)

    # embeddings = elmo(character_ids)['elmo_representations'][0]
    # l = len(embeddings[0])
    # a = torch.zeros([2, 256])
    # # print("inside avg_embedding")
    # # a = torch.zeros(1, config['embedding_dim'])
    #
    # for k in range(2):
    #     s = sentences[k]
    #     for i in range(len(s)):
    #         mu = get_idf_value(idf, s[i])
    #         a[k] = a[k] + embeddings[k][i] * mu
    # a /= (l + 0.0001)
    # return a[0], a[1]


def tokenize_sent_special(input_sent, input_lang, tag_sent, tag_lang, dep_sent, dep_lang):
    input_tensor = tensorFromSentence(input_lang, input_sent)
    tag_tensor = tensorFromSentence(tag_lang, tag_sent)
    dep_tensor = tensorFromSentence(dep_lang, dep_sent)
    input_tensor, tag_tensor, dep_tensor = removeUNKforall(input_tensor, tag_tensor, dep_tensor)
    # print(input_tensor)
    # input_tensor = removeUNK(input_tensor)
    # print(input_tensor)
    elmo_tensor = []
    if config['elmo']:
        # changed
        raise ValueError('elmo is removed, causing error because you are using elmo in config')
        # elmo_tensor = batch_to_ids([input_sent.split(' ')])
        # elmo_tensor = pad_sequences(elmo_tensor[0], config['MAX_LENGTH'], True).unsqueeze(0)
    input_tensor = pad_sequences(input_tensor, config['num_steps'], False).unsqueeze(0)
    tag_tensor = pad_sequences(tag_tensor, config['num_steps'], False).unsqueeze(0)
    dep_tensor = pad_sequences(dep_tensor, config['num_steps'], False).unsqueeze(0)

    return elmo_tensor, input_tensor, tag_tensor, dep_tensor


def check_min_length(sent):
    if len(sent.split(' ')) < config['min_length_of_edited_sent']:
        return 0
    else:
        return 1


def get_named_entity_score(sent):
    doc = nlp(sent)
    # print("named entity score = " + str(len(doc.ents)+0.01))
    return len(doc.ents) + 0.01


def calculate_cos_value(new, old, idf):
    e1 = avg_embedding(new, idf)
    e2 = avg_embedding(old, idf)
    cos_sim = nn.CosineSimilarity(dim=0, eps=1e-6)
    cos = cos_sim(e1, e2).item()
    return cos


def cos_similarity(new, old, idf):
    # e1, e2 = avg_embedding_elmo(new, old, idf)
    # e2 = avg_embedding_elmo(old, idf)
    cos = calculate_cos_value(new, old, idf)
    OldMax = 87
    OldMin = 5
    OldRange = (OldMax - OldMin)
    NewMax = 1.6
    NewMin = 1.005
    NewRange = (NewMax - NewMin)
    OldValue = len(old.split(' '))
    terminal_value = 2 - ((((OldValue - OldMin) * NewRange) / OldRange) + NewMin)
    # terminal_value = 2/math.log(OldValue)
    # print(terminal_value)
    if cos > max(config['cos_similarity_threshold'], terminal_value):
        return 1.0
    return 0.0


# changed


def comp_extract(sent, comp_simp_class_model, tokenizer):
    """ Extracting complex tokens from input sentence
    return a dict of : complex tokens in a sorted way based on their complexity,
                       not complex tokens that the attentin of CLS token to them is lower than the threshold (sorted),
                       attention matrices,
                       tokens in original order,
                       probability of the whole sentence for being complex,
    """

    out = get_model_out(comp_simp_class_model, tokenizer, sent)
    attention = out['attention']
    tokens = out['tokens']
    prob = out["prob"]

    layer = 1
    num_top_tokens = min(len(tokens), 10)
    CLS_attended_tokens = attention[layer].sum(dim=1)[0][0].topk(num_top_tokens)
    # [tokens[i] if for i in CLS_attended_tokens[1]], attention[layer].sum(dim=1)[0][0].topk(10)[0]

    more_than_thresh = []
    less_than_thresh = []
    thresh = attention[layer].sum(dim=1)[0][0].mean() * 3 / 2

    for i in range(len(CLS_attended_tokens[0])):
        if CLS_attended_tokens[0][i] > thresh:
            more_than_thresh.append(tokens[CLS_attended_tokens[1][i]])
        else:
            less_than_thresh.append(tokens[CLS_attended_tokens[1][i]])

    extracted_comps = {"comp_toks": more_than_thresh,
                       "not_comp_toks": less_than_thresh,
                       "threshold": thresh.item(),
                       "attention": attention,
                       'tokens': tokens,
                       'prob': prob,
                       }
    return extracted_comps


def neg_consts_words(comp_toks, tokens, stemmer):
    """ returns words for negative constraints
        removes some tokens,
        preprocesses the words,
        adds new negative constraint that are very similar words to the selected negative constraints (have same root)
    """

    # maximum number of accepted negative constraints
    max_num_accepted_consts = 4
    negs = []
    special_toks = ['[SEP]', '[CLS]', '.', 'Ġ.']

    for tok in comp_toks:

        # Each token should be a starting token, not a part of a word or special token
        if tok[0] == 'Ġ' and tok not in special_toks:

            # first word is usually selected mistakably so we do not pass it to the paraphraser
            if tokens.index(tok) + 1 != len(tokens) and tokens.index(tok) != 1:

                # We want the token be single word, not just the starting part of a word
                if tokens[tokens.index(tok) + 1][0] == 'Ġ':
                    negs.append(tok[1:])

    new_neg = []
    # adding all words with similar root
    try:
        lexeme("fly")
    except:
        print("lexeme handled")

    # adding words with similar root to negative constraints
    # e.g if the initial neg constraint is the word "facilitate"
    # then the new added words are : 'facilitate', 'facilitator', 'facilitative', 'facilitation', 'facilitate',
    # 'facilitates', 'facilitating', 'facilitated'
    for word in negs[:max_num_accepted_consts]:
        words_with_same_root = stemmer.unstem(stemmer.stem(word))
        words_with_same_root.remove(word) # the initial word will be added one time in the following

        new_neg += lexeme(word)
        new_neg += words_with_same_root

    return new_neg


def const_paraph(sent, neg_const, entities):
    stp_words = nltk.corpus.stopwords.words('english')

    # sent = sent.translate(str.maketrans('', '', string.punctuation))

    # removing all occurances of empty spaces from negative constraints
    neg_const = list(filter(lambda a: a != ' ' and a != '', neg_const))
    print("neg constraints: ", neg_const)

    # neg_const = neg_const.split(" ")
    pos_const = []

    neg_const = [x for x in neg_const if x not in entities and x not in stp_words]

    # if len(neg_const) >= 5:
    #     return -1

    # new_testing

    # if rest_pos_const:
    #     for i in sent.split():
    #         if i not in neg_const and i not in stp_words:
    #             pos_const.append(i.lower())
    # else:
    #     pos_const = entities

    inp = sent + "\t" + "|".join(neg_const) + '\t' + "|".join(pos_const)

    print("input: ", inp)

    f = open("inp_par.txt", "w")
    f.write(inp)
    f.close()

    f = open("out_par.txt", "w")
    f.close()

    # TODO
    imr_dir_path = '/home/m25dehgh/simplification/improved-ParaBank-rewriter'
    bashCommand = f"{imr_dir_path}/paraphrase.sh < ./inp_par.txt > ./out_par.txt 2> ./output_error_IMR.txt"

    # print(bashCommand)
    os.system(bashCommand)
    # print("outtt:", os.popen(bashCommand).read())

    f = open("./out_par.txt", "r")
    # print("tiem testing: ", f.read())
    # print("tiem testing2: ", f.read())
    return f.read()


def paraph(sent, leaves, entities, stemmer, details_sent):
    # obtaining negative constraints from comp-simp classifier attention layers.
    # print("input sentence: ", sent)
    extracted_comp_toks = comp_extract(sent, comp_simp_class_model, tokenizer_deberta)
    neg_consts = neg_consts_words(extracted_comp_toks['comp_toks'], extracted_comp_toks['tokens'], stemmer)

    # Adding used negetavie constraints in the previous steps to this step to prevent generating deleted words
    # if details_sent[1] == 'par':
    #     neg_consts += details_sent[3]

    sent = const_paraph(sent, neg_consts, entities )

    print('new: ', sent)
    if sent != -1 and sent != 1:
        return correct(all_norms(sent))
    else:
        return sent


# changed

def delete_leaves(sent, leaves):
    s = ''
    # print(leaves)
    for i in range(len(leaves)):
        s = s + ' ' + leaves[i]
    s = s + ' '
    old = sent
    sent = sent.replace(s, ' ')
    if old == sent:
        sent = sent.replace(s[1:], ' ')
        if sent[0] == ' ':
            sent = sent[1:]
    return correct(sent)


def get_model_out(model, tokenizer, sent):
    """ returns a dict containing : attention mat for all layers, tokens of the input sent, complexity probability """

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    toks = tokenizer(text=sent, truncation=True, padding=True, max_length=100, return_tensors='pt')

    input_ids = toks['input_ids'].to(device)
    attention_mask = toks['attention_mask'].to(device)
    token_type_ids = toks['token_type_ids'].to(device)

    with torch.no_grad():
        output = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_attentions=True,
                       return_dict=True)

    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze())
    attention = output.attentions

    out = {"attention": attention, "tokens": tokens, "prob": output.logits.squeeze().softmax(dim=0)[1].item()}

    return out


def construct_sent(sent):
    s = ''
    l = len(sent)
    for i in range(l):
        if sent[i] != '':
            if i != 0:
                s = s + ' ' + sent[i]
            else:
                s = s + sent[i]
    if sent[l - 1] != '.':
        s = s + ' .'
    return correct(s)


def replace_phrase(sent, phrase, new_phrase):
    sent = sent.replace(phrase, new_phrase)
    return correct(sent)


def correct(sent):
    s = sent.split(' ')
    # print('s is:  ', s)
    if s[0] == '':
        del s[0]
    if s[0] == ',':
        del s[0]
    if len(s) > 1 and s[len(s) - 2] == ',':
        del s[len(s) - 2]
    i = 0
    while i < len(s) - 2:
        if s[i + 1] == s[i]:
            del s[i]
        i = i + 1
    return convert_to_sent(s)


def get_subphrase_mod(sent, sent_list, input_lang, idf, simplifications, entities, synonym_dict, stemmer, details_sent):
    sent = sent.replace('%', ' percent')
    tree = next(parser.raw_parse(sent))

    return generate_phrases(sent, tree, sent_list, input_lang, idf, simplifications, entities, synonym_dict, stemmer, details_sent)


def generate_phrases(sent, tree, sent_list, input_lang, idf, simplifications, entities, synonym_dict, stemmer, details_sent):
    s = []
    p = []
    used_neg_consts = []
    all_par_calls = 0
    # print(sent)
    phrase_tags = ['S', 'ADJP', 'ADVP', 'CONJP', 'FRAG', 'INTJ', 'LST', 'NAC', 'NP', 'NX', 'PP', 'PRN', 'PRT',
                   'QP', 'RRC', 'UCP', 'VP', 'WHADJP', 'WHAVP', 'WHNP', 'WHPP', 'X', 'SBAR']

    # comented for testing paraphrasing and deletion in a sequential order instead of a parallel method in beam search
    if config['constrained_paraphrasing']:
        sp = paraph(sent, "", entities, stemmer, details_sent)
        if sp not in sent_list and sp != -1:
            s.append({sp: 'par'})
            all_par_calls += 1

    # To revert to the previous for (paraphrasing and deletion working in parallel in the beam search) search for
    # paraphrased_sent and change every place it was found.
    # paraphrased_sent = paraph(sent, "", entities, stemmer, rest_pos_const=False)

    if config['lexical_simplification'] or config['delete_leaves'] or config['reorder_leaves']:

        # tree = next(parser.raw_parse(paraphrased_sent))
        pos = tree.treepositions()
        for i in range(len(pos) - 1, 1, -1):
            if not isinstance(tree[pos[i]], str):
                if tree[pos[i]].label() in phrase_tags:
                    p.append(tree[pos[i]].leaves())

        for i in range(len(p)):
            if config['lexical_simplification']:
                simple = lexical_simplification(sent, p[i], input_lang, idf, simplifications, entities, synonym_dict)
                for st in simple:
                    if st not in sent_list:
                        s.append({st: 'ls'})
            if config['delete_leaves']:
                sd = delete_leaves(sent, p[i])
                # sd = delete_leaves(paraphrased_sent, p[i])
                if sd not in sent_list:
                    s.append({sd: 'dl'})
            if config['leaves_as_sent']:
                sc = construct_sent(p[i])
                if sc not in sent_list:
                    s.append({sc: 'las'})
            if config['reorder_leaves']:
                temp = []
                reorder_leaves(sent, p, p[i], convert_to_sent(p[i]), sd, temp)
                for rl in temp:
                    s.append({rl: 'rl'})

            # if config['constrained_paraphrasing']:
            #
            #     # creat the negative constraints
            #     neg_consts = gen_neg_const(entities, p[i])
            #     if neg_consts not in used_neg_consts:
            #
            #         # add the negative constraints to the used_neg_consts to avoid
            #         # using the same negative constraints in the future
            #         used_neg_consts.append(neg_consts)
            #         sp = paraph(sent, p[i], entities, rest_pos_const=False)
            #
            #         # new_testing
            #         # if sp == 1:
            #         #     all_par_calls += 1
            #
            #         if sp not in sent_list and sp != -1:
            #             s.append({sp: 'par'})

    # new_testing
    if len(s) > 0:
        return s, all_par_calls
    return '', all_par_calls


# return s

def reorder_leaves(sent, leaves, current_leaf, sc, sd, restructres):
    # if current_leaf == ['said']:
    # Assume sent is composed of phrases [A B C D E]
    # Current phrase/leaf let's say is D
    # Here loop will iterate for other leaves i.e A B C E (denoted by i)
    # Let;s say i is B [basically whatever i is the function will put current phrase infront of it]
    # first starting_sent becomes A B
    # end becomes C D E [A B C D E - A B]
    # starting_sent then becomes A B [A B - D]
    # end sent becomes C E [C D E - D]
    # final sent becomes A B + D + C E
    # However, this means with current implementation we wont get to D A B C E, need to fix this
    # Although there are complexities all subphrases are not distinct which makes this confusing
    # print('leaves are ...')
    # print(leaves)
    # print('\n')
    for i in leaves:
        if i != current_leaf and (convert_to_sent(i) not in convert_to_sent(current_leaf)
                                  and convert_to_sent(current_leaf) not in convert_to_sent(i)):

            try:
                # print(current_leaf)
                # print(i)
                index = sent.split(' ').index(i[len(i) - 1])
                # print(index)
                # print(sent)
                starting_sent = convert_to_sent(sent.split(' ')[:(index + 1)])
                # print(starting_sent)
                end = delete_leaves(sent, starting_sent.split(' '))
                starting_sent = delete_leaves(starting_sent, current_leaf)
                # print(starting_sent)
                end = delete_leaves(end, sc.split(' '))
                # st = correct(starting_sent+' '+sc+' ' + delete_leaves(sd,starting_sent.split(' ')))
                st = correct(starting_sent + ' ' + sc + ' ' + end)
                if st not in restructres:
                    restructres.append(st)
                '''st = correct(starting_sent+' '+sc+' .')
                if st not in restructres:
                	restructres.append(st)
                #st = correct(sc+' ' + delete_leaves(sd,starting_sent.split(' ')))
                st = correct(sc+' ' + end)
                if st not in restructres:
                	restructres.append(st)'''
            except:
                print('some issue as there is a mismatch of tokenization, corenlp does not always use spaces')


def in_vocab(phrase, input_lang):
    # checks if all the words in the phrase are present in our vocabulary
    phrase = phrase.replace(' .', '')
    phrase = phrase.replace('. ', '')
    phrase = phrase.split(' ')
    for i in range(len(phrase)):
        if getword(input_lang, phrase[i]) == UNK_token:
            return False
    return True


def checks_for_word_simplification(sent, word, synonyms, input_lang, pos, dep, idf, orig_sent_words, s):
    for new_word in synonyms:
        if getword(input_lang, new_word) == UNK_token:
            # print('rejected because not in vocabulary')
            continue
        if new_word in orig_sent_words:
            continue
        # print(phrase)
        # print(new_phrase)
        if get_idf_value(idf, new_word) > get_idf_value(idf, word):
            # print('idf value greater')
            continue
        infl = []
        inflections = getAllInflections(word)
        for k, v in inflections.items():
            infl.append(v[0])
        if new_word in infl:
            # print('is an inflection')
            continue
        cos_value = calculate_cos_value(word, new_word, None)
        # print(cos_value)
        if cos_value < config['cos_value_for_synonym_acceptance']:
            # print('rejected because not similar enough')
            continue

        sent = replace_phrase(sent, word, new_word)
        doc = nlp(sent)
        for token in doc:
            if token.text == new_word:
                # print('tags of new word')
                # print(token.tag_)
                # print(token.dep_)
                if pos == token.tag_ and dep == token.dep_:
                    if sent not in s:
                        s.append(sent)
            # print('rejected because tags not similar')


def get_word_to_simplify(phrase, idf, orig_sent_words, entities, lang):
    # we only simply the words that exist in the original sentence
    idf_val = -1
    complex_word = ''
    for word in phrase:
        word = word.lower()
        # if word is not present in the original sentence or word is a entity skip it
        if word not in orig_sent_words or word in entities:
            continue
        if getword(lang, word) == UNK_token:
            # if the word is not in entities and not present in the simple vocabulary, we simplify it
            return word
        # else, we choose the word that has the highest idf value above threshold
        val = get_idf_value(idf, word)
        if (val > config['min_idf_value_for_ls']) and idf_val < val:
            complex_word = word
            idf_val = val
    return complex_word


def get_entities(sent):
    doc = nlp(sent)
    entities = []
    for ent in doc.ents:
        entities.extend(ent.text.lower().split(' '))
    # print(entities)
    return entities


def lexical_simplification(sent, phrase, input_lang, idf, orig_sent_words, entities, synonym_dict):
    # simplifications = {scientist -> reader}
    s = []
    synonyms = []
    word_to_be_replaced = get_word_to_simplify(phrase, idf, orig_sent_words, entities, input_lang)
    # word_to_be_replaced = simplifications[word_to_be_replaced]
    if word_to_be_replaced != '':
        if word_to_be_replaced in synonym_dict:
            # print('synonyms already present')
            synonyms = synonym_dict[word_to_be_replaced]
        else:
            if config['dataset'] == 'Wikilarge':
                try:
                    sim = our_word2vec.similar_by_word(word_to_be_replaced, 20)
                    for i in range(len(sim)):
                        synonyms.append(sim[i][0])
                except:
                    print('word not found in our word2vec')
            try:
                sim = glove_model300.similar_by_word(word_to_be_replaced, 20)
                for i in range(len(sim)):
                    synonyms.append(sim[i][0])
            except:
                print('word not found in glove')
            try:
                sim = word2vec.similar_by_word(word_to_be_replaced, 20)
                for i in range(len(sim)):
                    synonyms.append(sim[i][0])
            except:
                print('word not found in word2vec')
            sym = wn.synsets(word_to_be_replaced)
            synonyms.extend(list(set(chain.from_iterable([word.lemma_names() for word in sym]))))
            synonym_dict[word_to_be_replaced] = synonyms
        # print('possible synonyms are ')
        # print(synonyms)
        doc = nlp(sent)
        # print(word_to_be_replaced)
        # print(sent)
        for token in doc:
            if token.text.lower() == word_to_be_replaced:
                pos = token.tag_
                dep = token.dep_
        checks_for_word_simplification(sent, word_to_be_replaced, synonyms, input_lang, pos, dep, idf, orig_sent_words,
                                       s)
    # print('s is')
    # print(s)
    return s


def calcluate_unigram_probability(sent, unigram_prob, input_lang):
    prob = 1.0
    for i in sent.lower().split(' '):
        prob += math.log(get_unigram_prob_value(unigram_prob, i))
    return prob / (len(sent.split(' ')))


def semantic_sim(sentA, sentB):
    """returns the probability that sentA and sentB have the same meaning"""

    semantic_model = SentenceTransformer('paraphrase-mpnet-base-v2')

    # Two lists of sentences
    sentences1 = [sentA]
    sentences2 = [sentB]

    # Compute embedding for both lists
    embeddings1 = semantic_model.encode(sentences1, convert_to_tensor=True)
    embeddings2 = semantic_model.encode(sentences2, convert_to_tensor=True)

    # Compute cosine-similarities
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)

    # Output the pairs with their score
    print("similarity of the two sentences: ", cosine_scores[0][0])

    return cosine_scores[0][0]


def calculate_score(lm_forward, elmo_tensor, tensor, tag_tensor, dep_tensor, input_lang, input_sent, orig_sent,
                    embedding_weights, idf, unigram_prob, cs):
    out_simplicity = get_model_out(comp_simp_class_model, tokenizer_deberta, input_sent)
    prob_simplicity = out_simplicity["prob"]
    score_simplicity = 1 - prob_simplicity
    score_final = score_simplicity

    # if the similarity between the input sentence and the original sentence is less than threshold the score becomes
    # zero
    sim_score = semantic_sim(input_sent, orig_sent)
    if sim_score < config['sim_threshold']:  # threshold should be added to config file # TODO
        score_final = 0

    # score_grammar_candidate = get_model_out(model_grammar_checker, tokenizer_deberta, input_sent)
    # score_grammar_original = get_model_out(model_grammar_checker, tokenizer_deberta, orig_sent)
    # print("candidate sentence grammar validity probability: ", score_grammar_candidate['prob'],
    #       "\n orginal sentence grammar prob:", score_grammar_original['prob'])
    # if score_grammar_candidate["prob"] * 1.1 < score_grammar_original["prob"]: # add any "magic number" to config file TODO
    #     score_final = 0

    return score_final


# def calculate_score(lm_forward, elmo_tensor, tensor, tag_tensor, dep_tensor, input_lang, input_sent, orig_sent,
#                     embedding_weights, idf, unigram_prob, cs):
#     prob = get_sentence_probability(lm_forward, elmo_tensor, tensor, tag_tensor, dep_tensor, input_lang, input_sent,
#                                     unigram_prob) ** config['sentence_probability_power']
#     if cs:
#         prob *= cos_similarity(input_sent.lower(), orig_sent.lower(), idf)
#     prob *= (get_named_entity_score(input_sent)) ** config['named_entity_score_power']
#     if config['check_min_length']:
#         prob *= check_min_length(input_sent)
#     prob /= len(input_sent.split(' ')) ** config['len_power']
#     if config['fre']:
#         prob *= sentence_fre(input_sent.lower()) ** config['fre_power']
#     return prob


class Dataset(data.Dataset):
    # 'Characterizes a dataset for PyTorch'
    def __init__(self, x_pair, y_pair):
        # 'Initialization'
        self.x_train = x_pair
        self.y_train = y_pair

    def __len__(self):
        # 'Denotes the total number of samples'
        return len(self.x_train)

    def __getitem__(self, index):
        # 'Generates one sample of data'
        # Select sample

        # Load data and get label
        x = self.x_train[index]
        y = self.y_train[index]

        return x, y


def load_data(dataset, batch_size):
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    return dataloader


def similarity_simplicity_grammar_assess(sys_sents, orig_file_path):
    orig = open(orig_file_path, encoding='utf-8').read().split('\n')

    acu_score_similarity = 0.
    acu_score_grammar_output = 0.
    acu_score_grammar_orig = 0.
    acu_score_simplicity_output = 0.
    acu_score_simplicity_orig = 0.
    acu_len_output = 0.
    acu_len_orig = 0.

    for i in tqdm(range(len(sys_sents))):
        output_sent = sys_sents[i]
        orig_sent = orig[i]

        score_similarity = semantic_sim(output_sent, orig_sent)
        score_grammar_output = get_model_out(model_grammar_checker, tokenizer_deberta, output_sent)["prob"]
        score_grammar_orig = get_model_out(model_grammar_checker, tokenizer_deberta, orig_sent)["prob"]
        score_simplicity_output = 1 - get_model_out(comp_simp_class_model, tokenizer_deberta, output_sent)["prob"]
        score_simplicity_orig = 1 - get_model_out(comp_simp_class_model, tokenizer_deberta, orig_sent)["prob"]
        len_output = len(output_sent.split())
        len_orig = len(orig_sent.split())

        acu_score_similarity += score_similarity
        acu_score_grammar_output += score_grammar_output
        acu_score_grammar_orig += score_grammar_orig
        acu_score_simplicity_output += score_simplicity_output
        acu_score_simplicity_orig += score_simplicity_orig
        acu_len_output += len_output
        acu_len_orig += len_orig

    return {
        "similarity": acu_score_similarity / len(sys_sents),
        "gram_out": acu_score_grammar_output / len(sys_sents),
        "gram_orig": acu_score_grammar_orig / len(sys_sents),
        "simplicity_out": acu_score_simplicity_output / len(sys_sents),
        "simplicity_orig:": acu_score_simplicity_orig / len(sys_sents),
        "len_out": acu_len_output / len(sys_sents),
        "len_orig": acu_len_orig / len(sys_sents),
    }


def calculate_sari_easse(ref_folder_path, sys_sents, orig_file_path):
    orig_sents = open(orig_file_path, encoding='utf-8').read().split('\n')

    orig_sents = orig_sents[:len(sys_sents)]

    ref_sents = []

    for i, file_path in enumerate(Path(ref_folder_path).glob("*")):
        f = open(file_path).read().split('\n')
        ref_sents.append(f[:len(sys_sents)])

    add, keep, delete = get_corpus_sari_operation_scores(orig_sents=orig_sents, sys_sents=sys_sents,
                                                         refs_sents=ref_sents)
    overal_sari = (add + keep + delete) / 3

    print(f'overal sari:{overal_sari}\
    add: {add}, keep: {keep}, delete: {delete}')

    return {"overall_sari": overal_sari, "addition": add, "keep": keep, "deletion": delete}


def save_output(file_name, saving_path, sys_sents):
    with open(saving_path + '/' + file_name, "w") as file:
        for i in range(len(sys_sents)):
            file.write(sys_sents[i] + "\n")


def save_config(config_dict, saving_path="."):
    config_file = open(saving_path + "/config.json", "w")
    json.dump(config_dict, config_file)
    config_file.close()


def load_config():
    conf_file = open("config.json", "r")
    config_dict = json.load(conf_file)
    conf_file.close()
    return config_dict


