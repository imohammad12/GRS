import pandas as pd
import nltk
import numpy as np
import torch
import transformers
import spacy
import time
import utils

from transformers import DebertaForSequenceClassification, Trainer, TrainingArguments, DebertaTokenizerFast
from spacy.tokenizer import Tokenizer
from collections import defaultdict
from pattern.en import lexeme
from tqdm import tqdm
# from utils import get_model_out, getword, get_idf_value, create_reverse_stem, get_word_to_simplify
from nltk.parse.corenlp import CoreNLPParser
from nltk.corpus import wordnet as wn
from itertools import chain
from pyinflect import getAllInflections, getInflection


class ComplexComponentDetector:
    default_params = {
        'path_classifier_model': '/home/m25dehgh/simplification/complex-classifier/results/newsela-auto-high-quality'
                                 '/whole-high-quality/checkpoint-44361/',
        'tokenizer_path': 'microsoft/deberta-base',
        "thresh_coef": 1.3,
        'ccd_version': 'combined',  # possible formats : 'combined', 'cls', 'ls'
        "UNK_token": 3,
        'cls_score_coef': 0, #0.001,
        'thresh_idf_cls': 4,
        'thresh_idf_combined': 4,
        'gpu': 0
    }

    def __init__(self, **config):
        self.params = self.default_params.copy()
        self.params.update(config)
        self.nlp = spacy.load("en_core_web_lg")
        self.nlp.tokenizer = Tokenizer(self.nlp.vocab)
        self.parser = CoreNLPParser('http://localhost:9000')
        self.stemmer = utils.create_reverse_stem()
        self.device = torch.device("cuda:"+str(self.params['gpu']) if torch.cuda.is_available() else "cpu")

    @classmethod
    def ls_version(cls, idf, output_lang, **config):
        """
        This version of CCD detects complex words by statistical methods used in Lexical Simplification operation of
        Iterative Edit-Based Unsupervised Sentence Simplification paper
        :param idf:
        :param output_lang:
        :param config:
        :return:
        """
        ccd = cls(**config)
        ccd.params['ccd_version'] = 'ls'
        ccd.idf = idf
        ccd.lang = output_lang
        # config.update(ccd.params)
        return ccd

    @classmethod
    def cls_version(cls, idf, comp_simp_class_model=None, tokenizer=None, **config):
        """
        This version of CCD detects complex words by interpreting the amount of
        attention payed to different tokens by the CLS token.
        :return: ComplexComponentDetector object
        """
        ccd = cls(**config)
        if comp_simp_class_model is None:
            print('Loading Deberta classifier model')
            ccd.comp_simp_class_model = DebertaForSequenceClassification.from_pretrained(
                ccd.params['path_classifier_model'])
        else:
            ccd.comp_simp_class_model = comp_simp_class_model
        ccd.comp_simp_class_model.to(ccd.device)
        ccd.comp_simp_class_model.eval()
        if tokenizer is None:
            print('Loading Deberta tokenizer')
            ccd.tokenizer = DebertaTokenizerFast.from_pretrained(ccd.params['tokenizer_path'])
        else:
            ccd.tokenizer = tokenizer
        ccd.params['ccd_version'] = 'cls'
        ccd.idf = idf
        # config.update(ccd.params)
        return ccd

    @classmethod
    def combined_version(cls, idf, output_lang, comp_simp_class_model=None, tokenizer=None, **config):
        """
        This version of CCD detects complex words by statistical methods used in Lexical Simplification operation of
        Iterative Edit-Based Unsupervised Sentence Simplification paper and interpreting the amount of
        attention payed to different tokens by the CLS token.
        :param idf:
        :param output_lang:
        :param comp_simp_class_model:
        :param tokenizer:
        :param config:
        :return: ComplexComponentDetector object
        """
        ccd = cls(**config)
        if comp_simp_class_model is None:
            print('Loading Deberta classifier model')
            ccd.comp_simp_class_model = DebertaForSequenceClassification.from_pretrained(
                ccd.params['path_classifier_model'])
        else:
            ccd.comp_simp_class_model = comp_simp_class_model
        ccd.comp_simp_class_model.to(ccd.device)
        ccd.comp_simp_class_model.eval()
        if tokenizer is None:
            print('Loading Deberta tokenizer')
            ccd.tokenizer = DebertaTokenizerFast.from_pretrained(ccd.params['tokenizer_path'])
        else:
            ccd.tokenizer = tokenizer

        ccd.params['ccd_version'] = 'combined'
        ccd.idf = idf
        ccd.lang = output_lang
        # config.update(ccd.params)
        return ccd

    def extract_complex_words(self, sent, entities):
        complex_pred = []
        neg_roots = []

        sent = sent.replace('%', ' percent')
        sent = sent.replace('` `', '`')

        if self.params["ccd_version"] == 'combined':
            # print("-- Sentce with error before ccd: ", sent)
            orig_sent_words = [i for i in self.parser.tokenize(sent)]
            complex_pred = self.get_complex_word_single_sent(orig_sent_words, entities)

        if self.params["ccd_version"] == "ls":
            orig_sent_words = sent.lower().split(' ')
            complex_pred = self.finding_complex_words(sent, orig_sent_words, entities)

        if self.params["ccd_version"] == 'combined' or self.params["ccd_version"] == "cls":
            extracted_comp_toks = self.extract_token_cls_comp_score(sent)
            neg_roots = self.raw_complx_token_to_words(extracted_comp_toks['comp_toks'],
                                                       extracted_comp_toks['tokens'],
                                                       entities,
                                                       word_level=False
                                                       )
            scores_dict = extracted_comp_toks['comp_scores']
            complexity_score_thresh = extracted_comp_toks['threshold']
            neg_roots = [word for word in neg_roots if utils.get_idf_value(self.idf, word) > self.params['thresh_idf_cls']]

        complex_pred = list(set(complex_pred + neg_roots))
        if self.params["ccd_version"] == 'combined':
            complex_pred = [word for word in complex_pred if utils.get_idf_value(self.idf, word) > self.params['thresh_idf_combined']]
            complex_pred = [word for word in complex_pred if scores_dict[word] > complexity_score_thresh * self.params['cls_score_coef']]

        # adding all words with similar root
        try:
            lexeme("fly")
        except:
            print("lexeme handled")
        new_neg = []
        # adding words with similar root to negative constraints
        # e.g if the initial neg constraint is the word "facilitate"
        # then the new added words are : 'facilitate', 'facilitator', 'facilitative', 'facilitation', 'facilitate',
        # 'facilitates', 'facilitating', 'facilitated'
        for word in complex_pred:
            words_with_same_root = self.stemmer.unstem(self.stemmer.stem(word))
            words_with_same_root.remove(word)  # the initial word will be added one time in the following

            new_neg += lexeme(word)
            new_neg += words_with_same_root

        return new_neg, complex_pred

    def extract_token_cls_comp_score(self, sent, thresh_coef=None):
        """ Extracting complex tokens from input sentence
        return a dict of : complex tokens in a sorted way based on their complexity,
                           not complex tokens that the attentin of CLS token to them is lower than the threshold (sorted),
                           attention matrices,
                           tokens in original order,
                           probability of the whole sentence for being complex,
                           complexity score of each token
        """

        out = utils.get_model_out(self.comp_simp_class_model, self.tokenizer, sent)
        attention = out['attention']
        tokens = out['tokens']
        prob = out["prob"]
        if thresh_coef is None:
            thresh_coef = self.params['thresh_coef']

        layer = 1
        num_top_tokens = len(tokens)
        CLS_attended_tokens_sorted = attention[layer].sum(dim=1)[0][0].topk(num_top_tokens)

        more_than_thresh = []
        less_than_thresh = []
        thresh = attention[layer].sum(dim=1)[0][0].mean()

        for i in range(len(CLS_attended_tokens_sorted[0])):
            if CLS_attended_tokens_sorted[0][i] > thresh * thresh_coef:
                more_than_thresh.append(tokens[CLS_attended_tokens_sorted[1][i]])
            else:
                less_than_thresh.append(tokens[CLS_attended_tokens_sorted[1][i]])

        complexity_scores = defaultdict(int)
        CLS_attended_socres = attention[layer].sum(dim=1)[0][0]
        for i, tok in enumerate(tokens):
            complexity_scores[self.token_to_word(tok, tokens)] = max(complexity_scores[self.token_to_word(tok, tokens)],
                                                                CLS_attended_socres[i].item())

        extracted_comps = {"comp_toks": more_than_thresh,
                           "not_comp_toks": less_than_thresh,
                           "threshold": thresh.item(),
                           "attention": attention,
                           'tokens': tokens,
                           'prob': prob,
                           'comp_scores': complexity_scores,
                           }
        return extracted_comps

    def token_to_word(self, token, all_tokens):
        """
        Gets a token in a sentence and returns the complete word by
        combining the given token and the adjacent tokens.
        """
        indx = all_tokens.index(token)

        special_toks = ['[SEP]', '[CLS]', '.', 'Ġ.', ',', '!', ';', '`']

        if token in special_toks:
            return token

        # If the given token is the first token of a compound word
        if token[0] == 'Ġ':
            word = token[1:]
            for tok in all_tokens[indx + 1:]:
                if tok[0] == 'Ġ' or (tok in special_toks):
                    break
                word += tok

        # If the given token is in the middle of a compund word
        else:
            word = token
            # Concatenate previous tokens
            for i in range(len(all_tokens[:indx]) - 1, 0, -1):
                tok = all_tokens[i]
                if tok in special_toks:
                    break
                if tok[0] == 'Ġ':
                    word = tok[1:] + word
                    break
                else:
                    word = tok + word

            # Concatenate next tokens
            for tok in all_tokens[indx + 1:]:
                if tok[0] == 'Ġ' or (tok in special_toks):
                    break
                word += tok

        return word

    def raw_complx_token_to_words(self, comp_toks, tokens, entities, word_level=False):
        """
            returns words for negative constraints from CLS attention complexity scores
            removes some tokens,
            preprocesses the words,
            adds new negative constraint that are very similar words to the selected negative constraints (have same root)
        """

        # maximum number of accepted negative constraints
        #     max_num_accepted_consts = 10
        negs = []
        special_toks = ['[SEP]', '[CLS]', '.', 'Ġ.', ',']

        for tok in comp_toks:

            # first word is usually selected mistakably so we do not pass it to the paraphraser
            if tokens.index(tok) + 1 != len(tokens) and tokens.index(tok) != 1 and tok not in special_toks:

                # Each token should be a starting token, not a part of a word or special token
                if word_level and tok[0] == 'Ġ':

                    # We want the token be single word, not just the starting part of a word
                    # So the next token should start with 'G' or be a special token
                    if tokens[tokens.index(tok) + 1][0] == 'Ġ' or tokens[tokens.index(tok) + 1] in special_toks:
                        negs.append(tok[1:])

                # When word_level is False we also consider complex tokens. So if a token is
                # complex we combine the adjacent tokens to return the compund word contatinig the complex token
                elif not word_level:
                    negs.append(self.token_to_word(tok, tokens))

        stp_words = nltk.corpus.stopwords.words('english')
        stp_words += ['`', '`s', '`ing', '`ed', ',', ',s', ',ing', ',ed']

        # removing all occurances of empty spaces from negative constraints
        negs = list(filter(lambda a: a != ' ' and a != '', negs))
        negs = [x for x in negs if x not in entities and x not in stp_words]

        return negs

    def get_complex_word_single_sent(self, sent, entities):
        complex_word = []
        for word in sent:
            word = word.lower()
            if utils.getword(self.lang, word) == self.params['UNK_token']:
                #             print('unk token: ', word)
                # if the word is not in entities and not present in the simple vocabulary, we simplify it
                complex_word.append(word)
                continue
            # else, we choose the word that has the highest idf value above threshold
            val = utils.get_idf_value(self.idf, word)
            if val > self.params['min_idf_value_for_ls']:
                complex_word.append(word)

        complex_word = self.lower_words_to_original(sent, complex_word)
        complex_word = [x for x in complex_word if x not in entities]

        return complex_word

    def finding_complex_words(self, input_sent, orig_sent_words, entities):

        input_sent = input_sent.replace('%', ' percent')
        input_sent = input_sent.replace('` `', '`')
        tree = next(self.parser.raw_parse(input_sent))
        p = []
        phrase_tags = ['S', 'ADJP', 'ADVP', 'CONJP', 'FRAG', 'INTJ', 'LST', 'NAC', 'NP', 'NX', 'PP', 'PRN', 'PRT',
                       'QP', 'RRC', 'UCP', 'VP', 'WHADJP', 'WHAVP', 'WHNP', 'WHPP', 'X', 'SBAR']
        pos = tree.treepositions()
        for i in range(len(pos) - 1, 1, -1):
            if not isinstance(tree[pos[i]], str):
                if tree[pos[i]].label() in phrase_tags:
                    p.append(tree[pos[i]].leaves())

        hard_words = set()
        for i in range(len(p)):
            word_to_be_replaced = utils.get_word_to_simplify(p[i],
                                                             self.idf,
                                                             orig_sent_words,
                                                             entities,
                                                             self.lang,
                                                             self.params)

            if word_to_be_replaced != '':
                hard_words.add(word_to_be_replaced)

        hard_words = self.lower_words_to_original(orig_sent_words, list(hard_words))
        hard_words = [x for x in hard_words if x not in entities]

        return hard_words

    @staticmethod
    def lower_words_to_original(orig_sent_words, complex_words):
        """
        returns the same given complex_words with orinal case sensitivity.
        """
        lower_sent = [x.lower() for x in orig_sent_words]
        new_complex_word = []
        for word in complex_words:
            if word.lower() in lower_sent:
                new_complex_word.append(orig_sent_words[lower_sent.index(word.lower())])
        return new_complex_word
