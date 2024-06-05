import pandas as pd
import numpy as np

def cosine_similarity(word1, word2):
    file = open("forager/data/fluency_lists/speech2vec_100.txt", "r")
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
                 #float_list = list(map(float, string_list))
            else:
                continue
            
        word_embeddings[key] = float_list
    #print(word_embeddings)
        
    vector1 = word_embeddings[word1]
    vector2 = word_embeddings[word2]
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    cosine_sim = dot_product/(magnitude1*magnitude2)
    print("Cosine Similarity:", cosine_sim)

cosine_similarity("apple", "peach")