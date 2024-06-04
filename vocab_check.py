import pandas as pd
import difflib
import numpy as np


# def vocab_check(vocab_file, data_file):
#     speech2vec_vocab_df = pd.read_csv("forager/data/lexical_data/speech2vec_vocab.csv")
#     food_data_df = pd.read_csv("forager/data/fluency_lists/food_data - Sheet1.csv")
#     for index,row in speech2vec_vocab_df.iterrows():

def cosine_similarity(word1, word2):
    file = open("forager/data/fluency_lists/speech2vec_100.txt", "r")
    word_embeddings = {}
    new_file = file.readlines()
    first_line_skipped = new_file[1: ]
    
    for line in first_line_skipped:
        split_line = line.split(" ")
        key = split_line[0]
        string_list = split_line[1: ]
        float_list = list(map(float, string_list))
        word_embeddings[key] = float_list
        
    vector1 = word_embeddings[word1]
    vector2 = word_embeddings[word2]
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    cosine_sim = dot_product/(magnitude1*magnitude2)
    print("Cosine Similarity:", cosine_sim)

cosine_similarity("apple", "apricot")