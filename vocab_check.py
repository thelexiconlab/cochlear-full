import pandas as pd
import difflib
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

#cosine_similarity("apple", "peach")

def vocab_check(vocab_file, data_file):
    speech2vec_vocab_df = pd.read_csv("forager/data/lexical_data/speech2vec_vocab.csv")
    food_data_df = pd.read_csv("forager/data/fluency_lists/food_data - Sheet1.csv")
    
    vocab = open("forager/data/fluency_lists/speech2vec_100.txt", "r")
    speech2vec_vocab_list = []
    for line in vocab.readlines():
        split_line = line.split(" ")
        speech2vec_vocab_list.append(split_line[0])
    

    data = open("forager/data/fluency_lists/food_data.txt", "r")
    food_data_list = []
    for line in data.readlines():
        split_line = line.split(" ")
        food_data_list.append(split_line[0])
    
    presence_list = []
    replacements = []
    for item in food_data_list:
        if item in speech2vec_vocab_list:
            presence_list.append("FOUND")
            replacements.append(item)
            # add column evaluation and write FOUND
        else:
            presence_list.append("NOT FOUND")
            replacements.append(difflib.get_close_matches(item, speech2vec_vocab_list))
            #add column evaluation and write NOT FOUND

    #incorporate difflib to evalaute potential replacements with get_close_matches
    food_data_df["evaluation"] = presence_list
    food_data_df["difflib"] = replacements
    
    final_csv_path = "forager/output/cochlear_food_fulldata_forager_results/speech2vec_vocab_check.csv"
    food_data_df.to_csv(final_csv_path, index=False)

vocab_check("forager/data/fluency_lists/speech2vec_100.txt", "forager/data/fluency_lists/food_data.txt")