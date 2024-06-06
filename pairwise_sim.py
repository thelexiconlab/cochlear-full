import pandas as pd
import numpy as np

def read_extract_list(filename):
    vocab = open(filename, "r")
    vocab_list = []
    list_of_lines = filename.readlines()
    label_skipped = list_of_lines[1: ]
    for line in label_skipped:
        split_line = line.split(" ")
        vocab_list.append(split_line[0])
    return vocab_list


def calc_pairwise_sim(speech2vec_file, word2vec_file, sim_matrix):
    pairwise_sim_df = pd.DataFrame()

    #isolate speech2vec vocab
    speech2vec_vocab_list = read_extract_list(speech2vec_file)
    pairwise_sim_df["Speech2vec Vocab"] = speech2vec_vocab_list
    
    speech_sims = []

    #calculate pairwise semantic similarity for speech2vec contents and insert them into the csv
    for i in range(0,len(speech2vec_vocab_list)):
        currentwordindex = i
        if i > 0:
            prevwordindex = i-1
            speech_sims.append(sim_matrix[prevwordindex, currentwordindex])
        else:
            speech_sims.append(0.0001)

    pairwise_sim_df["Speech2vec Semantic Similarity"] = speech_sims
    
    #isolate word2vec vocab
    word2vec_vocab_list = read_extract_list(word2vec_file)
    pairwise_sim_df["Word2vec Vocab"] = word2vec_vocab_list
    word_sims = []

    #calculate pairwise semantic similarity for word2vec contents and insert them into the csv
    for i in range(0,len(word2vec_vocab_list)):
        currentwordindex = i
        if i > 0:
            prevwordindex = i-1
            word_sims.append(sim_matrix[prevwordindex, currentwordindex])
        else:
            word_sims.append(0.0001)

    pairwise_sim_df["Word2vec Semantic Similarity"] = word_sims
    csv_path = "forager/output"
    pairwise_sim_df.to_csv(csv_path, index=False)

calc_pairwise_sim("forager/data/fluency_lists/speech2vec_100.txt", "forager/data/fluency_lists/word2vec_100.txt", )