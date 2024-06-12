import pandas as pd
import difflib
import numpy as np

'''
Args: 
    (1) df1: the base dataframe for data to be merged to
    (2) df2: the dataframe containing the data desired to be merged/added to df1
    (3) column_title: the string title of the column or index level to on (must be found in both dataframes)

Returns:
    (1) merged_df: a merged dataframe containing data from both df1 and df2
'''
def merge_results_hearing_status(df1, df2, column_title):
    df1 = pd.read_csv("forager/output/cochlear_food_fulldata_forager_results/individual_descriptive_stats.csv")
    #print(df1)
    df2 = pd.read_csv("forager/data/fluency_lists/cochlear_status_data.csv")
    #print(df2)
    results_path = 'forager/output/cochlear_food_fulldata_forager_results/merge_results.csv'
    merged_df = pd.merge(df1,df2, on=column_title, how='left')
    return merged_df

'''
Args:
    (1) file: the string file path to the file, which will be split on spaces

Returns:
    (1) speech2vec_vocab.csv: a file in the lexical_data folder that conatins a list of all unique words 
    extracted from the above file
'''
def extract_unique(file):
    open_file = open(file, "r")
    #unique_words = {}
    word_list = []
    for line in open_file.readlines():
        split_line = line.split(" ")
        #unique_words[split_line[0]] = split_line[1: ]
        word_list.append(split_line[0])
    
    word_list_df = pd.DataFrame(word_list)
    word_list_df.to_csv(path_or_buf="forager/data/lexical_data/speech2vec_vocab.csv")

'''
Args:
    (1) first_word: a string word to be compared to another
    (2) second_word: a string word to be compared to another

Returns:
    (1) semantic_sim: a float, representating the semantic similarity between first_word and second_word
'''
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

'''
Args:
    (1) vocab_file: the string file path to the vocab file
    (2) data_file: the string file path to the data file, to be evaluated against the larger vocab file
    (3) data_csv: the data file in .csv form
    (4) column_title: the string title of the column in the data file, from which one extracts data to be 
    compared to the vocab file

Returns:
    (1) speech2vec_vocab_check.csv: a file in the output folder containing the data entries, whether or not 
    they were present in the vocab, and the difflib get_close_matches from the vocab 
'''
def vocab_check(vocab_file, data_file, data_csv, column_title):
    food_data_df = pd.read_csv(data_csv)
    
    vocab = open(vocab_file, "r")
    speech2vec_vocab_list = []
    vocab_lines = vocab.readlines()
    label_skipped = vocab_lines[1: ]
    for line in label_skipped:
        split_line = line.split(" ")
        speech2vec_vocab_list.append(split_line[0])
    
    data = open(data_file, "r")
    food_data_list = []
    for item in food_data_df.loc[:, (column_title)]:
        food_data_list.append(item)

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

'''
Args:
    (1) csv_file: the string file path to the .csv file containing the data to be unduplicated
    (2) column_title: the string title of the column in the .csv file, from which one extracts data

Returns:
    (1) food_data_no_dups.csv: a file in the fluency_lists folder containing the data from the 
    specified column of the .csv file, with no duplicate entries
'''
def unduplicate(csv_file, column_title):
    csv_df = pd.read_csv(csv_file)
    vocab = open(csv_file, "r")
    word_set = set([])
    unduped_df = pd.DataFrame()
    for item in csv_df.loc[:, (column_title)]:
        word_set.add(item)
    unduped_df["unduplicated entries"] = list(word_set)
    csv_path = "forager/data/fluency_lists/food_data_no_dups.csv"
    unduped_df.to_csv(csv_path, index=False)

'''
Args:
    (1) filename: the string file path to the file from which vocab data will be extracted

Returns:
    (2) vocab_list: a list containing all individual data entries from filename, minus the first row 
    (often the column label)
'''
def read_extract_list(filename):
    vocab = open(filename, "r")
    vocab_list = []
    list_of_lines = vocab.readlines()
    label_skipped = list_of_lines[1: ]
    for line in label_skipped:
        split_line = line.split(" ")
        vocab_list.append(split_line[0])
    return vocab_list

'''
Args:
    (1) word1: a string word to be compared to another
    (2) word2: a string word to be compared to another
    (3) filename: the string file path to the file from which a word (key): list of float embeddings (value)
    dictionary will be constructed and utilized to calculate the cosine similarity between word1 and word2

Returns:
    (1) cosine_sim: the float represntation of the cosine similarity between word1 and word2, formatted in 
    this template - "Cosine Similarity: x"
'''
def cosine_similarity(word1, word2, filename):
    file = open(filename, "r")
    word_embeddings = {}
    list_of_lines = file.readlines()
    first_line_skipped = list_of_lines[1: ]
    
    for line in first_line_skipped:
        split_line = line.split(" ")
        key = split_line[0]
        if key in word_embeddings.keys():
            continue
        else:
            string_list = split_line[1:-1]
            float_list = []
            for item in string_list:
                if item:
                    float_list.append(float(item))
                else:
                    continue    
        word_embeddings[key] = float_list
        
    vector1 = word_embeddings[word1]
    vector2 = word_embeddings[word2]
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    cosine_sim = dot_product/(magnitude1*magnitude2)
    return cosine_sim

'''
Args:
    (1) food_data_csv: the string file path to the .csv file of the food data
    (2) subject_column_title: the string title of the column in the .csv file, from which one extracts 
    subject data
    (3) entry_column_title: the string title of the column in the .csv file, from which one extracts 
    entry data
    (4) lexical_results_csv: the string file path to a .csv file of the lexical analysis of the food_data_csv

Returns:
    (1) pairwise_sim_df: a dataframe including pairwise cosine similarity calculations using speech2vec and
    word2vec, as well as composite frequency analysis
'''
def calc_pairwise_sim(food_data_csv):
    print("running")
    pairwise_sim_df = pd.DataFrame()
    subjects = extract_subject(food_data_csv, "id")
    entries = extract_entries(food_data_csv, "entry")

    pairwise_sim_df['Subject'] = subjects
    pairwise_sim_df['Fluency_Item'] = entries
    
    speech_pair_cosines = pairwise_sim(food_data_csv, "forager/data/fluency_lists/speech2vec_100.txt")
    pairwise_sim_df['Speech2vec_Pairwise_Cosine_Similarity'] = speech_pair_cosines

    word_pair_cosines = pairwise_sim(food_data_csv, "forager/data/fluency_lists/word2vec_100.txt")
    pairwise_sim_df['Word2vec_Pairwise_Cosine_Similarity'] = word_pair_cosines

    #adding in pre-existing phonological and frequency data from lexical_results.csv
    adding_phon_freq(pairwise_sim_df, "forager/output/cochlear_food_fulldata_forager_results/lexical_results.csv")

    csv_path = "forager/output/pairwise_cosine_sim.csv"
    pairwise_sim_df.to_csv(csv_path, index=False)
    print("done")

'''
Args:
    (1) lexical_results_csv: the string file path to the .csv file of the lexical analysis of a given dataset

Returns:
    (1) composite_freq: a list of the pairwise composite frequencies of the entries in the dataset embedded in
    lexical_rewsults_csv
'''
def composite_freq(lexical_results_csv):
    lexical_df = pd.read_csv(lexical_results_csv)
    freq_list = []
    for item in lexical_df.loc[:,('Frequency_Value')]:
        freq_list.append(item)
    
    composite_freq = []
    idx = 0
    while idx < (len(freq_list)):
        #now the lists are different lengths, what are the last elements pairwise calculations?
        if idx > 0:
            currentfreqindex = idx
            prevfreqindex = idx-1
            product = freq_list[currentfreqindex]*freq_list[prevfreqindex]
            composite_freq.append(product)
            currentfreqindex = idx
        else:
            composite_freq.append(0.0001)
        idx += 1
    # print(composite_freq)
    # print(len(composite_freq))
    return composite_freq

'''
Args:
    (1) existing_df: an existing dataframe where phonological and frequency data will be added

Returns:
    (1) existing_df: the dataframe with additional phonological similarity, frequency value, and composite
    frequency data
'''
def adding_phon_freq(existing_df, lexical_results_csv):
    #adding phonologcial information
    lexical_results_df = pd.read_csv(lexical_results_csv)

    phon_list = []
    for item in lexical_results_df.loc[:, ('Phonological_Similarity')]:
        phon_list.append(item)
    existing_df['Phonological_Similarity'] = phon_list

    #adding frequency and composite frequency
    freq_list = []
    for item in lexical_results_df.loc[:, ('Frequency_Value')]:
        freq_list.append(item)
    existing_df['Frequency_Value'] = freq_list
    existing_df['Composite_Frequency'] = composite_freq(lexical_results_csv)

    return existing_df

'''
Args:
    (1) food_data_csv: the string file path to the .csv file of the food data
    (2) subject_column_title: the string title of the column in the .csv file, from which one extracts 
    subject data

Returns:
    (1) subject_list: a list of each individual subject extracted from the food_data_csv 
'''
def extract_subject(food_data_csv, subject_column_title):
    food_csv_df = pd.read_csv(food_data_csv)
    subject_list = []

    for item in food_csv_df.loc[:, (subject_column_title)]:
        subject_list.append(item)
    return subject_list

'''
Args:
    (1) food_data_csv: the string file path to the .csv file of the food data
    (2) entry_column_title: the string title of the column in the .csv file, from which one extracts 
    entry data

Returns:
    (1) entry_list: a list of the food fluency items extracted from the food_data_csv
'''
def extract_entries(food_data_csv, entry_column_title):
    food_csv_df = pd.read_csv(food_data_csv)
    entry_list = []

    for item in food_csv_df.loc[:, (entry_column_title)]:
        entry_list.append(item)
    return entry_list

'''
Args:
    (1) data_csv: the string file path to the .csv file of the desired dataset
    (2) dict_file: the string file to the .txt from which to extract a word (key): embeddings (value) 
    dictionary

Returns:
    (1) pair_cosines: a list of pairwise cosine similarity calculations for the data in the data_csv using
    a dictionary of embeddings extracted from the dict_file
'''
def pairwise_sim(data_csv, dict_file):
    entry_list = extract_entries(data_csv, "entry")

    pair_cosines = []
    idx = 0

    while idx < len(entry_list):
        if idx > 0:
            try:
                currentwordindex = idx
                prevwordindex = idx-1
                curr_speech_cosine = cosine_similarity(entry_list[prevwordindex], entry_list[currentwordindex], dict_file)
                pair_cosines.append(curr_speech_cosine)
            except KeyError:
                pair_cosines.append("NA")
        else:
            try:
                pair_cosines.append(0.0001)
            except KeyError:
                pair_cosines.append("NA")
        # print(idx)
        # print(pair_cosines)
        idx += 1
    # print(pair_cosines)
    return pair_cosines