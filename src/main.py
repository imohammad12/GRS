import importlib
import sys
from utils import *
import json
import numpy as np

from transformers import DebertaForSequenceClassification, Trainer, TrainingArguments, DebertaTokenizerFast
from ccd import ComplexComponentDetector
from model.structural_decoder import DecoderGRU

# importlib.reload(sys.modules['utils'])
# importlib.reload(sys.modules['config'])
# from config import model_config as config

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

print('Creating ccd object...')
ccd = ComplexComponentDetector.combined_version(idf,
                                                output_lang,
                                                comp_simp_class_model=comp_simp_class_model,
                                                tokenizer=tokenizer_deberta,
                                                **config)


asset_paths = {
    "log_directory": "/home/m25dehgh/simplification/outputs/asset/whole-dataset",
    "ref_folder_path": "/home/m25dehgh/simplification/datasets/asset-from-easse/ref-test",
    "orig_file_path": "/home/m25dehgh/simplification/datasets/asset-from-easse/asset.test.orig",
}

newsela_paths = {
  "log_directory": "/home/m25dehgh/simplification/outputs/newsela/whole-dataset",
  "ref_folder_path": "/home/m25dehgh/simplification/datasets/newsela/dhruv-newsela/ref-test-orig",
  "orig_file_path": "/home/m25dehgh/simplification/datasets/newsela/dhruv-newsela/V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.ori.test.src",
}

config = load_config()
config['dataset'] = 'Wikilarge'

if config['dataset'] == "Newsela":
    config.update(newsela_paths)
elif config['dataset'] == 'Wikilarge':
    config.update(asset_paths)
else:
    raise ValueError("Wrong dataset name: use Newsela or Wikilarge")

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


open(config['file_name'], "w").close()

start_time = time.time()

from tree_edits_beam import *

# Testing multiple configurations
# for i, del_threshold in enumerate(np.arange(1.1, 1.5, 0.1)):
    # for j, par_thresh in enumerate(np.arange(0.6, 1.1, 0.1)):

config = load_config()

    # 	config['threshold']['par'] = 0.8
    # 	config['threshold']['dl'] = 2.0
    #
    # config['delete_leaves'] = True
    # config['constrained_paraphrasing'] = True

    # config['sim_threshold'] = np.round(simplicity_thresh, 2)

    # config['delete_leaves'] = False

    # config['threshold']['par'] = np.round(par_thresh, 2)
config['threshold']['par'] = 1.0
config['threshold']['dl'] = np.round(del_threshold, 2)

save_config(config)

    # importlib.reload(sys.modules['utils'])
    # from utils import *

if config['set'] == 'valid':
    sample(valid_complex, valid_simple, output_lang, tag_lang, dep_lang, lm_forward, lm_backward,
           output_embedding_weights, idf, unigram_prob, start_time, load_config(), tokenizer_deberta,
           comp_simp_class_model, ccd, model_grammar_checker)

elif config['set'] == 'test':
    sample(test_complex, test_simple, output_lang, tag_lang, dep_lang, lm_forward, lm_backward,
           output_embedding_weights, idf, unigram_prob, start_time, load_config(), tokenizer_deberta,
           comp_simp_class_model, ccd, model_grammar_checker)

open(config['file_name'], "w").close()

end = time.time()
print(f"Runtime of the program is {end - start_time}")
start_time = end
