import pandas as pd
import difflib
import numpy as np


def vocab_check(vocab_file, data_file):
    speech2vec_vocab_df = pd.read_csv("forager/data/lexical_data/speech2vec_vocab.csv")
    food_data_df = pd.read_csv("forager/data/fluency_lists/food_data - Sheet1.csv")
    for index,row in speech2vec_vocab_df.iterrows():


def cosine_similarity(word1, word2):
    vector1 = 
    vector2 = 
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    cosine_sim = dot_product/(magnitude1*magnitude2)
    print("Cosine Similarity:", cosine_sim)