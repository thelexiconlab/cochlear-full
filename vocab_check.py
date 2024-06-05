import pandas as pd
import difflib
import numpy as np

def vocab_check(vocab_file, data_file):
    food_data_df = pd.read_csv("forager/data/fluency_lists/food_data - Sheet1.csv")
    
    vocab = open("forager/data/fluency_lists/speech2vec_100.txt", "r")
    speech2vec_vocab_list = []
    vocab_lines = vocab.readlines()
    label_skipped = vocab_lines[1: ]
    for line in label_skipped:
        split_line = line.split(" ")
        speech2vec_vocab_list.append(split_line[0])
    
    data = open("forager/data/fluency_lists/food_data.txt", "r")
    food_data_list = []
    for item in food_data_df.loc[:, "entry"]:
        food_data_list.append(item)
    #print(food_data_list)

    presence_list = []
    replacements = []
    for item in food_data_list:
        if item in speech2vec_vocab_list:
            presence_list.append("FOUND")
            replacements.append(item)
        else:
            presence_list.append("NOT FOUND")
            replacements.append(difflib.get_close_matches(item, speech2vec_vocab_list))

    food_data_df["evaluation"] = presence_list
    food_data_df["difflib"] = replacements
    
    final_csv_path = "forager/output/speech2vec_vocab_check.csv"
    food_data_df.to_csv(final_csv_path, index=False)

vocab_check("forager/data/fluency_lists/speech2vec_100.txt", "forager/data/fluency_lists/food_data.txt")