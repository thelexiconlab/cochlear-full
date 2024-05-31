# extract unique words from large txt file
# move words into dictionary (key = word, value = array of embeddings) iteratively, if a duplicate -- then do not add

import pandas as pd

def extract_unique(file):
    file = open("forager/data/fluency_lists/speech2vec_100.txt", "r")
    #unique_words = {}
    word_list = []
    for line in file.readlines():
        split_line = line.split(" ")
        #unique_words[split_line[0]] = split_line[1: ]
        word_list.append(split_line[0])
    
    word_list_df = pd.DataFrame(word_list)
    word_list_df.to_csv(path_or_buf="forager/data/lexical_data/speech2vec_vocab.csv")

extract_unique("forager/data/fluency_lists/speech2vec_100.txt")