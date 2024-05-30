#calculating the semantic similarity between two words

import pandas as pd

def calc_semantic_sim(first_word, second_word):
    semantic_matrix = pd.read_csv("forager/data/lexical_data/USE_semantic_matrix.csv", header=None)
    vocab_df = pd.read_csv("forager/data/lexical_data/vocab.csv")
    vocab = {}
    for index, word in vocab_df.itertuples():
        vocab[word] = index
    index1 = vocab[first_word]
    index2 = vocab[second_word]
    semantic_sim = semantic_matrix.loc[index1,index2]
    return semantic_sim

print(calc_semantic_sim("pizza","popcorn"))
print(calc_semantic_sim("popcorn", "hot dog"))
print(calc_semantic_sim("hot dog","cheese"))
print(calc_semantic_sim("cheese","coffee"))
print(calc_semantic_sim("coffee","tea"))
print(calc_semantic_sim("tea","soda"))
print(calc_semantic_sim("soda","water"))
print(calc_semantic_sim("water","asparagus"))
print(calc_semantic_sim("asparagus","broccoli"))
print(calc_semantic_sim("strawberries","blueberries"))