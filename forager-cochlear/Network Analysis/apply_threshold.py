import pandas as pd
import itertools

import csv
sim_matrix = pd.read_csv(r"C:\Users\c.hambric\OneDrive - Bowdoin College\Documents\cochlear-full\forager-cochlear\data\lexical_data\animals\word2vec\USE_semantic_matrix.csv",header=None)
def replace_values(df):
    df[df < 0.5] = 0
    df[df > 0.5] = 1
    return df

thresh_matrix = replace_values(sim_matrix)

thresh_matrix.to_csv(r"C:\Users\c.hambric\OneDrive - Bowdoin College\Documents\cochlear-full\forager-cochlear\0_8_thresholdapp.csv", index=False)

print(f"threshold matrix has been written to file")