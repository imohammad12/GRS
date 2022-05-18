import importlib
import sys
from utils import *
import json
import numpy as np

from transformers import DebertaForSequenceClassification, Trainer, TrainingArguments, DebertaTokenizerFast
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from ccd import ComplexComponentDetector
from model.structural_decoder import DecoderGRU
from tree_edits_beam import *

config = load_config()

print('Loading Deberta Tokenizer...')
tokenizer_deberta = DebertaTokenizerFast.from_pretrained('microsoft/deberta-base')

root_comp_simp = "/home/m25dehgh/simplification/complex-classifier"
model_comp_simp = "newsela-auto-high-quality"
path_comp_simp = root_comp_simp + '/results' + '/' + model_comp_simp + "/whole-high-quality/checkpoint-44361/"
comp_simp_class_model = DebertaForSequenceClassification.from_pretrained(path_comp_simp).to(config['gpu'])
comp_simp_class_model.eval()

print('Loading Grammar Checker model...')
root_grammar_checker = "/home/m25dehgh/simplification/grammar-checker"
model_name_grammar_checker = "deberta-base-cola"
path = root_grammar_checker + '/results' + '/' + model_name_grammar_checker + "/checkpoint-716"
model_grammar_checker = DebertaForSequenceClassification.from_pretrained(path)

tokenizer_paraphrasing = None
model_paraphrasing = None

if config['paraphrasing_model'] != 'imr':
    tokenizer_paraphrasing = AutoTokenizer.from_pretrained(config['paraphrasing_model'])
    model_paraphrasing = AutoModelForSeq2SeqLM.from_pretrained(config['paraphrasing_model']).to(
        config['paraphrasing_gpu'])
    model_paraphrasing.eval()

save_config(config)

idf, unigram_prob, output_lang, tag_lang, dep_lang, valid_complex, test_complex, output_embedding_weights, \
tag_embedding_weights, dep_embedding_weights = prepareData(config['embedding_dim'], config['freq'],
                                                           config['ver'], config['dataset'],
                                                           config['operation'], config)


print('Creating ccd object...')
ccd = ComplexComponentDetector.cls_version(idf,
                                           comp_simp_class_model=comp_simp_class_model,
                                           tokenizer=tokenizer_deberta,
                                           **config)

open(config['resume_file'], "w").close()
start_time = time.time()
ccd.params.update(config)
if config['set'] == 'valid':
    sample(valid_complex, output_lang, tag_lang, dep_lang, idf, start_time, load_config(), tokenizer_deberta,
           comp_simp_class_model, ccd, model_grammar_checker, tokenizer_paraphrasing, model_paraphrasing)

elif config['set'] == 'test':
    sample(test_complex, output_lang, tag_lang, dep_lang, idf, start_time, load_config(), tokenizer_deberta,
           comp_simp_class_model, ccd, model_grammar_checker, tokenizer_paraphrasing, model_paraphrasing)

open(config['resume_file'], "w").close()
