import importlib
import sys
from utils import *
import json
import numpy as np
import torch

from transformers import DebertaForSequenceClassification, Trainer, TrainingArguments, DebertaTokenizerFast
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from ccd import ComplexComponentDetector
from tree_edits_beam import *

config = load_config()

print('Loading Deberta Tokenizer...')
tokenizer_deberta = DebertaTokenizerFast.from_pretrained('microsoft/deberta-base')


print('Loading Deberta complex-simple classifier model...')
general_device = "cuda:"+str(config['gpu']) if torch.cuda.is_available() and config['gpu'] != 'cpu' else "cpu"
comp_simp_class_model = DebertaForSequenceClassification.from_pretrained(config['comp_simp_classifier_model']).to(general_device)
comp_simp_class_model.eval()

print('Loading Grammar Checker model...')
model_grammar_checker = DebertaForSequenceClassification.from_pretrained(config['grammar_model']).to(general_device)

tokenizer_paraphrasing = None
model_paraphrasing = None

if config['paraphrasing_model'] != 'imr':
    print('Loading Paraphrasing model and Tokenizer...')
    tokenizer_paraphrasing = AutoTokenizer.from_pretrained(config['paraphrasing_model'])
    paraphrasing_device = "cuda:" + str(config['paraphrasing_gpu']) if torch.cuda.is_available() and config['paraphrasing_gpu'] != 'cpu' else "cpu"
    model_paraphrasing = AutoModelForSeq2SeqLM.from_pretrained(config['paraphrasing_model']).to(paraphrasing_device)
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
