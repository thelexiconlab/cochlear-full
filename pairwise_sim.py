import pandas as pd
import numpy as np

def read_extract_list(filename):
    vocab = open(filename, "r")
    vocab_list = []
    list_of_lines = vocab.readlines()
    label_skipped = list_of_lines[1: ]
    for line in label_skipped:
        split_line = line.split(" ")
        vocab_list.append(split_line[0])
    return vocab_list

def cosine_similarity(word1, word2, filename):
    file = open(filename, "r")
    word_embeddings = {}
    list_of_lines = file.readlines()
    first_line_skipped = list_of_lines[1: ]
    
    for line in first_line_skipped:
        split_line = line.split(" ")
        key = split_line[0]
        string_list = split_line[1:-1]
        float_list = []
        for item in string_list:
            if item:
                float_list.append(float(item))
            else:
                continue
            
        word_embeddings[key] = float_list
        
    vector1 = word_embeddings[word1]
    vector2 = word_embeddings[word2]
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    cosine_sim = dot_product/(magnitude1*magnitude2)
    #print("Cosine Similarity:", cosine_sim)

#cosine_similarity("apple", "apple")

def calc_pairwise_sim(speech2vec_file, word2vec_file):
    pairwise_sim_df = pd.DataFrame()
    print("here1")

    #isolate speech2vec vocab
    speech2vec_vocab_list = read_extract_list(speech2vec_file)
    pairwise_sim_df["Speech2vec Vocab"] = speech2vec_vocab_list
    
    speech_sims = []

    print("here2")
    #calculate pairwise cosine similarity for speech2vec contents and insert them into the csv
    idx = 0
    while idx < len(speech2vec_vocab_list):
        currentwordindex = idx
        if idx > 0:
            prevwordindex = idx-1
            speech_sims.append(cosine_similarity(speech2vec_vocab_list[prevwordindex], speech2vec_vocab_list[currentwordindex], "forager/data/fluency_lists/speech2vec_100.txt"))
        else:
            speech_sims.append(0.0001)
        idx += 1

    pairwise_sim_df["Speech2vec Pairwise Cosine Similarity"] = speech_sims
    print("here3")

    #isolate word2vec vocab
    word2vec_vocab_list = read_extract_list(word2vec_file)
    pairwise_sim_df["Word2vec Vocab"] = word2vec_vocab_list
    word_sims = []

    #calculate pairwise cosine similarity for word2vec contents and insert them into the csv
    word_idx = 0
    while word_idx < len(word2vec_vocab_list):
        currentwordindex = word_idx
        if word_idx > 0:
            prevwordindex = word_idx-1
            word_sims.append(cosine_similarity(word2vec_vocab_list[prevwordindex], word2vec_vocab_list[currentwordindex], "forager/data/fluency_lists/word2vec_100.txt"))
        else:
            word_sims.append(0.0001)
        
        word_idx += 1
    print("here4")
    pairwise_sim_df["Word2vec Pairwise Cosine Similarity"] = word_sims
    csv_path = "forager/output/pairwise_cosine_sim.csv"
    pairwise_sim_df.to_csv(csv_path, index=False)

calc_pairwise_sim("forager/data/fluency_lists/speech2vec_100.txt", "forager/data/fluency_lists/word2vec_100.txt")