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

# importlib.reload(sys.modules['utils'])
# importlib.reload(sys.modules['config'])
# from config import model_config as config

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

asset_paths = {
    "log_directory": "/home/m25dehgh/simplification/outputs/asset/whole-dataset",
    "ref_folder_path": "/home/m25dehgh/simplification/datasets/asset-from-easse/ref-test",
    "orig_file_path": "/home/m25dehgh/simplification/datasets/asset-from-easse/asset.test.orig",
    "extra_log_directory": "/home/m25dehgh/simplification/outputs/newsela/whole-dataset",
}
newsela_paths = {
    "log_directory": "/home/m25dehgh/simplification/outputs/newsela/whole-dataset",
    "ref_folder_path": "/home/m25dehgh/simplification/datasets/newsela/dhruv-newsela/ref-test-orig",
    "orig_file_path": "/home/m25dehgh/simplification/datasets/newsela/dhruv-newsela"
                      "/V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.ori.test.src",
    "extra_log_directory": "/home/m25dehgh/simplification/outputs/asset/whole-dataset",
}
tokenizer_paraphrasing = None
model_paraphrasing = None

# config['dataset'] = 'Wikilarge'

if config['dataset'] == "Newsela":
    config.update(newsela_paths)
elif config['dataset'] == 'Wikilarge':
    config.update(asset_paths)
else:
    raise ValueError("Wrong dataset name: use Newsela or Wikilarge")

if config['paraphrasing_model'] != 'imr':
    tokenizer_paraphrasing = AutoTokenizer.from_pretrained(config['paraphrasing_model'])
    model_paraphrasing = AutoModelForSeq2SeqLM.from_pretrained(config['paraphrasing_model']).to(
        config['paraphrasing_gpu'])
    model_paraphrasing.eval()

save_config(config)

idf, unigram_prob, output_lang, tag_lang, dep_lang, train_simple, valid_simple, test_simple, train_complex, \
valid_complex, test_complex, output_embedding_weights, tag_embedding_weights, \
dep_embedding_weights = prepareData(config['embedding_dim'], config['freq'], config['ver'], config['dataset'],
                                    config['operation'], config)

lm_forward = DecoderGRU(config['hidden_size'], output_lang.n_words, tag_lang.n_words, dep_lang.n_words,
                        config['num_layers'],
                        output_embedding_weights, tag_embedding_weights, dep_embedding_weights, config['embedding_dim'],
                        config['tag_dim'], config['dep_dim'], config['dropout'],
                        config['use_structural_as_standard']).to(device)
lm_backward = DecoderGRU(config['hidden_size'], output_lang.n_words, tag_lang.n_words, dep_lang.n_words,
                         config['num_layers'],
                         output_embedding_weights, tag_embedding_weights, dep_embedding_weights,
                         config['embedding_dim'], config['tag_dim'], config['dep_dim'], config['dropout'],
                         config['use_structural_as_standard']).to(device)

print('Creating ccd object...')
# ccds = {
# "combined": ComplexComponentDetector.combined_version(idf,
#                                                               output_lang,
#                                                               comp_simp_class_model=comp_simp_class_model,
#                                                               tokenizer=tokenizer_deberta,
#                                                               **config),
#         "cls": ComplexComponentDetector.cls_version(idf,
#                                                     comp_simp_class_model=comp_simp_class_model,
#                                                     tokenizer=tokenizer_deberta,
#                                                     **config),
# "ls": ComplexComponentDetector.ls_version(idf,
#                                           output_lang,
#                                           **config.copy())}

ccd = ComplexComponentDetector.cls_version(idf,
                                           comp_simp_class_model=comp_simp_class_model,
                                           tokenizer=tokenizer_deberta,
                                           **config)

open(config['file_name'], "w").close()

# Testing multiple configurations
# for i, del_threshold in enumerate(np.arange(1.1, 1.5, 0.1)):
# for j, par_thresh in enumerate(np.arange(0.7, 1.1, 0.1)):

# for i in range(0, 2):
#     config = load_config()
#     if i == 0:
#         config["paraphrasing_model"] = "/home/m25dehgh/simplification/testing-notebooks/bart-large-mnli-finetuned-parabank2-selected/checkpoint-5500"
#     else:
#         config["paraphrasing_model"] = "imr"
#     save_config(config)


# 	config['threshold']['par'] = 0.8
# 	config['threshold']['dl'] = 2.0

#
# config['delete_leaves'] = True
# config['constrained_paraphrasing'] = True
# config['constrained_paraphrasing'] = True if i == 1 else False

# config['sim_threshold'] = np.round(simplicity_thresh, 2)

# config['delete_leaves'] = False


# importlib.reload(sys.modules['utils'])
# from utils import *


# for i in range(2):
#     config = load_config()
#     config['leaves_as_sent'] = True
#     config['delete_leaves'] = True
#     if i == 0:
#         config['constrained_paraphrasing'] = False
#     else:
#         config['constrained_paraphrasing'] = True
#
#     save_config(config)
#
#     start_time = time.time()
#     ccd.params.update(config)
#     if config['set'] == 'valid':
#         sample(valid_complex, valid_simple, output_lang, tag_lang, dep_lang, lm_forward, lm_backward,
#                output_embedding_weights, idf, unigram_prob, start_time, load_config(), tokenizer_deberta,
#                comp_simp_class_model, ccd, model_grammar_checker, tokenizer_paraphrasing, model_paraphrasing)
#
#     elif config['set'] == 'test':
#         sample(test_complex, test_simple, output_lang, tag_lang, dep_lang, lm_forward, lm_backward,
#                output_embedding_weights, idf, unigram_prob, start_time, load_config(), tokenizer_deberta,
#                comp_simp_class_model, ccd, model_grammar_checker, tokenizer_paraphrasing, model_paraphrasing)
#
#     open(config['file_name'], "w").close()

# ================================================================
#
# for j, par_thresh in enumerate(np.arange(0.5, 0.89, 0.1)):
#     config = load_config()
#     config['delete_leaves'] = False
#     config['leaves_as_sent'] = False
#     config['constrained_paraphrasing'] = True
#     if j == 3:
#         config['threshold']['par'] = 0.9
#     else:
#         config['threshold']['par'] = np.round(par_thresh, 2)
#
#     save_config(config)
#     start_time = time.time()
#     ccd.params.update(config)
#     if config['set'] == 'valid':
#         sample(valid_complex, valid_simple, output_lang, tag_lang, dep_lang, lm_forward, lm_backward,
#                output_embedding_weights, idf, unigram_prob, start_time, load_config(), tokenizer_deberta,
#                comp_simp_class_model, ccd, model_grammar_checker, tokenizer_paraphrasing, model_paraphrasing)
#
#     elif config['set'] == 'test':
#         sample(test_complex, test_simple, output_lang, tag_lang, dep_lang, lm_forward, lm_backward,
#                output_embedding_weights, idf, unigram_prob, start_time, load_config(), tokenizer_deberta,
#                comp_simp_class_model, ccd, model_grammar_checker, tokenizer_paraphrasing, model_paraphrasing)
#
#     open(config['file_name'], "w").close()
#
#
for i in range(2):
    config = load_config()
    config['delete_leaves'] = True if i == 0 or i == 1 else False
    config['leaves_as_sent'] = True if i == 0 or i == 1 else False
    config['constrained_paraphrasing'] = True if i == 1 else False
    save_config(config)

    start_time = time.time()
    ccd.params.update(config)
    if config['set'] == 'valid':
        sample(valid_complex, valid_simple, output_lang, tag_lang, dep_lang, lm_forward, lm_backward,
               output_embedding_weights, idf, unigram_prob, start_time, load_config(), tokenizer_deberta,
               comp_simp_class_model, ccd, model_grammar_checker, tokenizer_paraphrasing, model_paraphrasing)

    elif config['set'] == 'test':
        sample(test_complex, test_simple, output_lang, tag_lang, dep_lang, lm_forward, lm_backward,
               output_embedding_weights, idf, unigram_prob, start_time, load_config(), tokenizer_deberta,
               comp_simp_class_model, ccd, model_grammar_checker, tokenizer_paraphrasing, model_paraphrasing)

    open(config['file_name'], "w").close()

#
# for i, gram_thresh in enumerate(np.arange(0.0, 0.4, 0.1)):
#
#     config = load_config()
#     config['delete_leaves'] = False
#     config['leaves_as_sent'] = False
#     config['constrained_paraphrasing'] = True
#     if i == 3:
#         config['grammar_threshold'] = 0.8
#     else:
#         config['grammar_threshold'] = np.round(gram_thresh, 2)
#     save_config(config)
#
#     start_time = time.time()
#     ccd.params.update(config)
#     if config['set'] == 'valid':
#         sample(valid_complex, valid_simple, output_lang, tag_lang, dep_lang, lm_forward, lm_backward,
#                output_embedding_weights, idf, unigram_prob, start_time, load_config(), tokenizer_deberta,
#                comp_simp_class_model, ccd, model_grammar_checker, tokenizer_paraphrasing, model_paraphrasing)
#
#     elif config['set'] == 'test':
#         sample(test_complex, test_simple, output_lang, tag_lang, dep_lang, lm_forward, lm_backward,
#                output_embedding_weights, idf, unigram_prob, start_time, load_config(), tokenizer_deberta,
#                comp_simp_class_model, ccd, model_grammar_checker, tokenizer_paraphrasing, model_paraphrasing)
#
#     open(config['file_name'], "w").close()


