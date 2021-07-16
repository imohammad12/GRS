model_config = {
    'clip': 50,
    'lr': 0.001,
    'num_steps': 87,
    # 'threshold': {'ls':0.8, 'dl':1.25, 'las':5.0, 'rl':1.25, 'pa': 1.25}, #Newsela -> {'ls':1.25, 'dl':1.25, 'las':1.25, 'rl':1.25, 'pa': 1.25}, Wikilarge ->{'ls':0.8, 'dl':1.25, 'las':5.0, 'rl':1.25, 'pa': 1.25}
    # changed
    # 'threshold': {'ls':1.25, 'dl':1.25, 'las':1.25, 'rl':1.25, 'pa': 1.25},  # For Newsela
    # 'threshold': {'ls': 1.25, 'dl': 1, 'las': 0.75, 'rl':1.25, 'pa': 1.25},  # For ASSET 37.3 with ls and ro
    # 'threshold': {'ls': 0.75, 'dl': 1.35, 'las': 3.0, 'rl':1.5, 'pa': 1.25},  # For ASSET 36.02 with out ls and ro
    # 'threshold': {'ls': 1.25, 'dl': 1, 'las': 0.75, 'rl':1.25, 'pa': 1.25},  # For ASSET 37.25 without ls and ro
    # 'threshold': {'ls': .8, 'dl': 1, 'las': 3.0, 'rl':1.25, 'pa': 1.25},  # For ASSET 32.60 only ls
    # 'threshold': {'ls': .8, 'dl': 1, 'las': 3.0, 'rl':1.25, 'pa': 1.25},  # For ASSET 37.36 only LS+RM
    'threshold': {'ls': .8, 'dl': 1, 'las': 3.0, 'rl':1.25, 'par': 1},  # For ASSET - all operation
    'epochs': 100,
    'set': 'test',
    # 'lm_name': 'Newsela/structured_lm_forward_300_150_0_4', #wikilarge -> Wikilarge/structured_lm_forward_300_150_0_4_freq5, newsela -> Newsela/structured_lm_forward_300_150_0_4
    # changed
    'lm_name': 'Wikilarge/structured_lm_forward_300_150_0_4_freq5',  # For wikilarge
    'use_structural_as_standard': False,
    'lm_backward': False,
    'embedding_dim': 300,
    'tag_dim': 150,
    'dep_dim': 150,
    'hidden_size': 256,
    'num_layers': 2,
    'freq':0,
    'min_length': 100,
    'dataset': 'Wikilarge',  # 'Wikilarge', #Wikilarge, Newsela  #  changed
    'ver':'glove.6B.',
    'dropout':0.4,
    'batch_size':64,
    'print_every':100,
    'MAX_LENGTH': 85,
    'double_LM': False,
    'gpu': 1,
    'awd': False,
    # 'file_name': 'Wikilarge/output/simplifications_Asset.txt',#Wikilarge/output/simplifications_Wikilarge.txt , Newsela/output/simplifications_Newsela.txt #changed
    'file_name': 'Wikilarge/output/simplifications.txt',
    # Changed
    'fre': True,
    'SLOR': True,
    'beam_size': 1,
    'elmo': False,
    'min_length_of_edited_sent': 6,
    'lexical_simplification': False,  # changed
    'constrained_paraphrasing': True, #changed added
    'delete_leaves': False,
    'leaves_as_sent': False,
    'reorder_leaves': False,  # changed
    'check_min_length': True,
    'cos_similarity_threshold': 0.5, #WIKILARGE -> 0.7
    'cos_value_for_synonym_acceptance': 0.5, #Newsela ->0.5 WIKILARGE->0.45  # changed
    'min_idf_value_for_ls': 11,  #Wikilarge -> 9, NEwsela -> 11  # changed
    'sentence_probability_power': 1.0, #Wikilarge=0.5, Newsela->1.0 # changed
    'named_entity_score_power': 1.0,
    'len_power': 0.25, #Wikilarge=0.25, Newsela -> 1.0  ASSET -> 0.5 # changed
    'fre_power': 1.0,
    'operation': 'sample' # or sample or train_lm,
}
