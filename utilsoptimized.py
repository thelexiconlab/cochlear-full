import pandas as pd
import difflib
import numpy as np
from utils import *


def id_df(id_entries):
    id_df = pd.DataFrame()
    #creating entries common column
    id_df['Fluency_Item'] = id_entries
    #calculating pairwise cosine similarity using Speech2vec
    speech_pair_cosines = pairwise_sim("forager/data/fluency_lists/speech2vec_100.txt", id_entries)
    id_df['Speech2vec_Pairwise_Cosine_Similarity'] = speech_pair_cosines
    #print("speech pairwise complete")

    #calculating pairwise cosine similarity using Word2vec
    word_pair_cosines = pairwise_sim("forager/data/fluency_lists/word2vec_100.txt", id_entries)
    id_df['Word2vec_Pairwise_Cosine_Similarity'] = word_pair_cosines
    #print("word pairwise complete")

    #calculating pairwise phonological similarity from scratch
    pairwise_phon_stats = pairwise_phon(id_entries, "forager/data/lexical_data/vocab.csv", "forager/data/lexical_data/USE_phonological_matrix.csv")
    id_df['Phonological_Scratch'] = pairwise_phon_stats
    #print("phon pairwise complete")

    #adding in pre-existing phonological and frequency data from lexical_results.csv
    id_df['Frequency_Value'] = get_frequency(id_entries, "forager/data/lexical_data/USE_frequencies.csv")
    id_df['Previous_Word_Frequency'] = id_df['Frequency_Value'].shift(1)
    id_df['Composite_Frequency'] = id_df['Frequency_Value']*id_df['Previous_Word_Frequency']

    #print("id_df complete")
    #print(id_df)

    return id_df

'''
Args:
    (1) food_data_csv_path: the string file path to the .csv file of the food data

Returns:
    (1) pairwise_sim_df: a dataframe including pairwise cosine similarity calculations using speech2vec and
    word2vec, as well as composite frequency analysis
'''
def calc_pairwise_sim(food_data_csv_path):
    print("running")
    pairwise_sim_df = pd.DataFrame()
    food_csv_df = pd.read_csv(food_data_csv_path)
    subjects = food_csv_df['id'].tolist()
    entries = food_csv_df['entry'].tolist()

    pairwise_sim_df['Subject'] = subjects
    pairwise_sim_df['Fluency_Item'] = entries

    grouped = food_csv_df.groupby('id')

    full_df = pd.DataFrame()
    for name,group in grouped:
        id_list = group['entry'].to_list()
        stats_df = id_df(id_list)
        stats_df["ID"]= name
        full_df = pd.concat([full_df, stats_df], ignore_index = True)
    
    csv_path = "forager/output/optimized_pairwise_cosine_sim.csv"
    full_df.to_csv(csv_path, index=False)
    print("done")

calc_pairwise_sim("forager/data/fluency_lists/test_data.csv")