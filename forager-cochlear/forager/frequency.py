
import urllib
import requests
import pandas as pd
from tqdm import tqdm
import numpy as np
import re
from wordfreq import zipf_frequency

def get_frequencies(path_for_lexical_data):
    '''
        Description:
            Creates a CSV file of frequencies for each word in the vocabulary, obtained from the wordfreq package.
            The first column is the word, the second is the log count, and the third is the raw count.
            The resulting file is saved in the data/lexical data folder.
        Args:
            (1) embeddings: path to vocab
    '''
    data = pd.read_csv(path_for_lexical_data + 'vocab.csv') 
    items = data['word'].values.tolist()

    items_and_counts = []
    for item in tqdm(items):
        count = zipf_frequency(item, 'en')
        items_and_counts.append((item, count))
        
    item_counts_df = pd.DataFrame(items_and_counts, columns=['item','log_count'])
    
    item_counts_df.loc[item_counts_df['log_count'] == 0, 'log_count'] = .0001
    
    item_counts_df.to_csv(path_for_lexical_data + '/frequencies.csv', index=False, header=None)
    return None

#get_frequencies('../data/lexical_data/animals/')