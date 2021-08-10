import torch
from utils import *
import os
import math
from model.SARI import calculate
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu

sf = SmoothingFunction()


def sample(complex_sentences, simple_sentences, input_lang, tag_lang, dep_lang, lm_forward, lm_backward,
           embedding_weights, idf, unigram_prob, start_time, config):
    count = 0
    sari_scorel = 0
    keepl = 0
    deletel = 0
    addl = 0
    b_scorel = 0
    p_scorel = 0
    fkgl_scorel = 0
    fre_scorel = 0
    all_par_calls = 0
    beam_calls = 0
    start_index = config['start_index']
    stats = {'ls': 0, 'dl': 0, 'las': 0, 'rl': 0, 'par': 0}
    sys_sents = []
    lm_forward.load_state_dict(torch.load(config['lm_name'] + '.pt'))
    if config['double_LM']:
        lm_backward.load_state_dict(torch.load('structured_lm_backward_300_150_0_4.pt'))
    lm_forward.eval()
    lm_backward.eval()
    for i in range(start_index, len(complex_sentences)):
        if len(complex_sentences[i].split(' ')) <= config['min_length']:
            print(f'length of complex and simple sent list: {len(complex_sentences)}, {len(simple_sentences)}')
            # new_testing
            sl, kl, dl, al, bl, pl, fkl, frl, par_calls, b_calls, out_sent = mcmc(complex_sentences[i],
                                                                                  simple_sentences[i], input_lang,
                                                                                  tag_lang, dep_lang, lm_forward,
                                                                                  lm_backward, embedding_weights, idf,
                                                                                  unigram_prob, stats, config)

            sys_sents.append(out_sent)

            # new_testing
            all_par_calls += par_calls
            beam_calls += b_calls

            print('\n')
            print("Average sentence level SARI till now for sentences")
            sari_scorel += sl
            keepl += kl
            deletel += dl
            addl += al
            p_scorel += pl
            print(sari_scorel / (count + 1))
            print(keepl / (count + 1))
            print(deletel / (count + 1))
            print(addl / (count + 1))
            print("Average sentence level BLEU till now for sentences")
            b_scorel += bl
            print(b_scorel / (count + 1))
            print("Average perplexity of sentences")
            print(p_scorel / (count + 1))
            fkgl_scorel += fkl
            fre_scorel += frl
            print('Average sentence level FKGL and FRE till now for sentences')
            print(fkgl_scorel / (count + 1))
            print(fre_scorel / (count + 1))
            print('\n')
            print(i + 1)

            end = time.time()
            print(f"Runtime of the program is {end - start_time}")
            print(f"total paraphrasing calls {all_par_calls}, total beam calls {beam_calls}")

            with open(config['file_name'], "a") as file:
                file.write("Number {}: Average Sentence Level Perplexity, Bleu, SARI \n".format(i))  # changed
                file.write(str(p_scorel / (count + 1)) + " " + str(b_scorel / (count + 1)) + " " + str(
                    sari_scorel / (count + 1)) + "\n\n")
            count += 1

    # sari_scores = calculate_sari_easse(ref_folder_path=config["ref_folder_path"], sys_sents=sys_sents,
    #                                    orig_file_path=config['orig_file_path'])
    # simil_simp_gram_scores = similarity_simplicity_grammar_assess(sys_sents=sys_sents,
    #                                      orig_file_path=config['orig_file_path'])
    #
    # all_scores = {**sari_scores, **simil_simp_gram_scores}

    # print("all scores", all_scores)
    #
    # folder_path = config['log_directory'] + "/" + config['run_number'] + "-{:.2f}".format(all_scores['overall_sari'])
    # if not os.path.exists(folder_path):
    #     os.makedirs(folder_path)
    #     save_config(config, folder_path)
    #     save_output("sys_out_" + config['run_number'], folder_path, sys_sents=sys_sents)
    #     config['run_number'] += 1
    #     save_config(config)

    print(stats)


def mcmc(input_sent, reference, input_lang, tag_lang, dep_lang, lm_forward, lm_backward, embedding_weights, idf,
         unigram_prob, stats, config):
    print(stats)
    # input_sent = "highlights 2009 from the 2009 version of 52 seconds setup for passmark 5 32 5 2nd scan time , and 7 mb memory- 7 mb memory ."
    reference = reference.lower()
    given_complex_sentence = input_sent.lower()
    # final_sent = input_sent
    orig_sent = input_sent
    # print(given_complex_sentence)
    beam = {}
    entities = get_entities(input_sent)
    perplexity = -10000
    perpf = -10000
    synonym_dict = {}
    sent_list = []
    spl = input_sent.lower().split(' ')

    # new_testing
    all_par_calls = 0
    beam_calls = 0

    # creating reverse stem for all words
    stemmer = create_reverse_stem()

    # the for loop below is just in case if the edit operations go for a very long time
    # in almost all the cases this will not be required

    for iter in range(2 * len(spl)):

        '''if len(input_sent.split(' ')) <= 3:
            print('sentence length already at min, so cannot do deletion')
            # it could be debatable where do we get 85 from, is it from aligned text
            continue'''

        # new_testing

        doc = nlp(input_sent)
        elmo_tensor, input_sent_tensor, tag_tensor, dep_tensor = tokenize_sent_special(input_sent.lower(), input_lang,
                                                                                       convert_to_sent(
                                                                                           [(tok.tag_).upper() for
                                                                                            tok in doc]), tag_lang,
                                                                                       convert_to_sent(
                                                                                           [(tok.dep_).upper() for tok
                                                                                            in doc]), dep_lang)

        prob_old = calculate_score(lm_forward, elmo_tensor, input_sent_tensor, tag_tensor, dep_tensor, input_lang,
                                   input_sent, orig_sent, embedding_weights, idf, unigram_prob, False)
        # if config['double_LM']:
        #     elmo_tensor_b, input_sent_tensor_b, tag_tensor_b, dep_tensor_b = tokenize_sent_special(reverse_sent(input_sent.lower()), input_lang, reverse_sent(convert_to_sent([(tok.tag_).upper() for
        #         tok in doc])), tag_lang, reverse_sent(convert_to_sent([(tok.dep_).upper() for tok in doc])), dep_lang)
        #     prob_old += calculate_score(lm_backward, elmo_tensor_b, input_sent_tensor_b, tag_tensor_b, dep_tensor_b, input_lang, reverse_sent(input_sent), reverse_sent(orig_sent), embedding_weights, idf, unigram_prob, False)
        #     prob_old /= 2.0
        # for the first time step the beam size is 1, just the original complex sentence
        if iter == 0:
            beam[input_sent] = [prob_old, 'original']
        print('Getting candidates for iteration: ', iter)
        # print(input_sent)
        new_beam = {}
        # intialize the candidate beam
        for key in beam:

            # new_testing
            beam_calls += 1

            # get candidate sentence through different edit operations
            candidates = get_subphrase_mod(key, sent_list, input_lang, idf, spl, entities, synonym_dict, stemmer)

            # new_testing
            all_par_calls += candidates[1]
            candidates = candidates[0]
            # print(f"per sentence accumulative all paraphrasing calls is {all_par_calls}, in beam number {beam_calls}")
            # print('candidates are ', candidates)
            '''if len(candidates) == 0:
                break'''

            for i in range(len(candidates)):
                # print(candidate)
                sent = list(candidates[i].keys())[0]
                operation = candidates[i][sent]
                doc = nlp(list(candidates[i].keys())[0])

                elmo_tensor, candidate_tensor, candidate_tag_tensor, candidate_dep_tensor = tokenize_sent_special(
                    sent.lower(), input_lang, convert_to_sent([(tok.tag_).upper() for
                                                               tok in doc]), tag_lang,
                    convert_to_sent([(tok.dep_).upper() for tok in doc]), dep_lang)

                # calculate score for each candidate sentence using the scoring function
                p = calculate_score(lm_forward, elmo_tensor, candidate_tensor, candidate_tag_tensor,
                                    candidate_dep_tensor, input_lang, sent, orig_sent, embedding_weights, idf,
                                    unigram_prob, True)
                print(f'Candidate: {sent}\nOld Prob: {prob_old}, New Sent Prob: {p} \n')

                if config['double_LM']:
                    elmo_tensor_b, candidate_tensor_b, candidate_tag_tensor_b, candidate_dep_tensor_b = tokenize_sent_special(
                        reverse_sent(sent.lower()), input_lang, reverse_sent(convert_to_sent([(tok.tag_).upper() for
                                                                                              tok in doc])), tag_lang,
                        reverse_sent(convert_to_sent([(tok.dep_).upper() for tok in doc])), dep_lang)
                    p += calculate_score(lm_backward, elmo_tensor_b, candidate_tensor_b, candidate_tag_tensor_b,
                                         candidate_dep_tensor_b, input_lang, reverse_sent(sent),
                                         reverse_sent(orig_sent), embedding_weights, idf, unigram_prob, True)
                    p /= 2.0

                # if the candidate sentence is able to increase the score by a threshold value, add it to the beam
                if p > prob_old * config['threshold'][operation]:
                    new_beam[sent] = [p, operation]
                    # record the edit operation by which the candidate sentence was created
                    stats[operation] += 1
                else:
                    # if the threshold is not crossed, add it to a list so that the sentence is not considered in the future
                    sent_list.append(sent)
        if new_beam == {}:
            # if there are no candidate sentences, exit
            break
        # print(new_beam)
        new_beam_sorted_list = sorted(new_beam.items(), key=lambda x: x[1])[-config['beam_size']:]
        # sort the created beam on the basis of scores from the scoring function
        # print(new_beam_sorted_list)
        new_beam = {}
        # top k top scoring sentences selected. In our experiments the beam size is 1
        # copying the new_beam_sorted_list into new_beam
        for key in new_beam_sorted_list:
            new_beam[key[0]] = key[1]
        # new_beam = new_beam_sorted_list.copy()
        # print(new_beam)
        # we'll get top beam_size (or <= beam size) candidates

        # get the top scoring sentence. This will act as the source sentence for the next iteartion                
        maxvalue_sent = max(new_beam.items(), key=lambda x: x[1])[0]
        perpf = new_beam[maxvalue_sent][0]
        input_sent = maxvalue_sent
        # for the next iteration
        beam = new_beam.copy()

    input_sent = input_sent.lower()
    # print(given_complex_sentence)
    # print(reference)
    print("Input complex sentence")
    print(given_complex_sentence)
    print("Reference sentence")
    print(reference)
    print("Simplified sentence")
    print(input_sent)

    scorel, keepl, deletel, addl = calculate(given_complex_sentence, input_sent.lower(), [reference])
    # print(scorel)
    # print(keepl)
    # print(deletel)
    # print(addl)
    bleul = sentence_bleu([convert_to_blue(reference)], convert_to_blue(input_sent.lower()),
                          weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=sf.method3)
    # print("Blue score")
    # print(bleul)
    # print("Perplexity")
    if (perpf == -10000):
        # print('sentence remain unchanged therefore calculating perp score for last generated sentence')
        doc = nlp(input_sent)
        elmo_tensor, best_input_tensor, best_tag_tensor, best_dep_tensor = tokenize_sent_special(input_sent.lower(),
                                                                                                 input_lang,
                                                                                                 convert_to_sent(
                                                                                                     [(tok.tag_).upper()
                                                                                                      for
                                                                                                      tok in doc]),
                                                                                                 tag_lang,
                                                                                                 convert_to_sent(
                                                                                                     [(tok.dep_).upper()
                                                                                                      for tok in doc]),
                                                                                                 dep_lang)
        perpf = calculate_score(lm_forward, elmo_tensor, best_input_tensor, best_tag_tensor, best_dep_tensor,
                                input_lang, input_sent, orig_sent, embedding_weights, idf, unigram_prob, False)
        if config['double_LM']:
            elmo_tensor_b, best_input_tensor_b, best_tag_tensor_b, best_dep_tensor_b = tokenize_sent_special(
                reverse_sent(input_sent.lower()), input_lang, reverse_sent(convert_to_sent([(tok.tag_).upper() for
                                                                                            tok in doc])), tag_lang,
                reverse_sent(convert_to_sent([(tok.dep_).upper() for tok in doc])), dep_lang)
            perpf += calculate_score(lm_backward, elmo_tensor_b, best_input_tensor_b, best_tag_tensor_b,
                                     best_dep_tensor_b, input_lang, reverse_sent(input_sent), reverse_sent(orig_sent),
                                     embedding_weights, idf, unigram_prob, False)
    # print(perpf)
    # print('fkgl and fre')
    fkgl_scorel = sentence_fkgl(input_sent)
    fre_scorel = sentence_fre(input_sent)
    # print(fkgl_scorel)
    # print(fre_scorel)
    with open(config['file_name'], "a") as file:
        file.write(given_complex_sentence + "\n")
        file.write(reference + "\n")
        # file.write(final_sent.lower() + "\n")
        # file.write(str(perplexity) + " " + str(bleu) + " " + str(score) + " " + str(keep) + " " + str(delete) + " " + str(add) + " " + str(fkgl_score) + " " + str(fre_score) + "\n")
        file.write(input_sent.lower() + "\n")
        file.write(
            str(perpf) + " " + str(bleul) + " " + str(scorel) + " " + str(keepl) + " " + str(deletel) + " " + str(
                addl) + " " + str(fkgl_scorel) + " " + str(fre_scorel) + "\n")
        file.write("\n")

    # new_testing
    return scorel, keepl, deletel, addl, bleul, perpf, fkgl_scorel, fre_scorel, all_par_calls, beam_calls, input_sent
