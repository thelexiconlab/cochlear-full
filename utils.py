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
    (1)
Returns:
    (1)
'''
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

#extract_unique("forager/data/fluency_lists/speech2vec_100.txt")

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

# print(calc_semantic_sim("pizza","popcorn"))
# print(calc_semantic_sim("popcorn", "hot dog"))
# print(calc_semantic_sim("hot dog","cheese"))
# print(calc_semantic_sim("cheese","coffee"))
# print(calc_semantic_sim("coffee","tea"))
# print(calc_semantic_sim("tea","soda"))
# print(calc_semantic_sim("soda","water"))
# print(calc_semantic_sim("water","asparagus"))
# print(calc_semantic_sim("asparagus","broccoli"))
# print(calc_semantic_sim("strawberries","blueberries"))

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

#vocab_check("forager/data/fluency_lists/speech2vec_100.txt", "forager/data/fluency_lists/food_data.txt","forager/data/fluency_lists/food_data - Sheet1.csv", "entry")
#vocab_check("forager/data/fluency_lists/speech2vec_100.txt", "forager/data/fluency_lists/food_vocab_data.txt","forager/data/lexical_data/vocab.csv", "vocab")
#vocab_check("forager/data/fluency_lists/speech2vec_100.txt","forager/data/fluency_lists/food_data_unduped.txt", "forager/data/fluency_lists/food_data_unduped.csv", "unduplicated entries")

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

#unduplicate("forager/data/fluency_lists/food_data - Sheet1.csv", "entry")

def read_extract_list(filename):
    vocab = open(filename, "r")
    vocab_list = []
    list_of_lines = vocab.readlines()
    label_skipped = list_of_lines[1: ]
    for line in label_skipped:
        split_line = line.split(" ")
        vocab_list.append(split_line[0])
    return vocab_list

def cosine_similarity(word1, word2, filename):
    file = open(filename, "r")
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
            else:
                continue
            
        word_embeddings[key] = float_list
        
    vector1 = word_embeddings[word1]
    vector2 = word_embeddings[word2]
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    cosine_sim = dot_product/(magnitude1*magnitude2)
    #print("Cosine Similarity:", cosine_sim)

#cosine_similarity("apple", "apple")

def calc_pairwise_sim(speech2vec_file, word2vec_file):
    pairwise_sim_df = pd.DataFrame()
    print("here1")

    #isolate speech2vec vocab
    speech2vec_vocab_list = read_extract_list(speech2vec_file)
    pairwise_sim_df["Speech2vec Vocab"] = speech2vec_vocab_list
    
    speech_sims = []

    print("here2")
    #calculate pairwise cosine similarity for speech2vec contents and insert them into the csv
    idx = 0
    while idx < len(speech2vec_vocab_list):
        currentwordindex = idx
        if idx > 0:
            prevwordindex = idx-1
            speech_sims.append(cosine_similarity(speech2vec_vocab_list[prevwordindex], speech2vec_vocab_list[currentwordindex], "forager/data/fluency_lists/speech2vec_100.txt"))
        else:
            speech_sims.append(0.0001)
        idx += 1

    pairwise_sim_df["Speech2vec Pairwise Cosine Similarity"] = speech_sims
    print("here3")

    #isolate word2vec vocab
    word2vec_vocab_list = read_extract_list(word2vec_file)
    pairwise_sim_df["Word2vec Vocab"] = word2vec_vocab_list
    word_sims = []

    #calculate pairwise cosine similarity for word2vec contents and insert them into the csv
    word_idx = 0
    while word_idx < len(word2vec_vocab_list):
        currentwordindex = word_idx
        if word_idx > 0:
            prevwordindex = word_idx-1
            word_sims.append(cosine_similarity(word2vec_vocab_list[prevwordindex], word2vec_vocab_list[currentwordindex], "forager/data/fluency_lists/word2vec_100.txt"))
        else:
            word_sims.append(0.0001)
        
        word_idx += 1
    print("here4")
    pairwise_sim_df["Word2vec Pairwise Cosine Similarity"] = word_sims
    csv_path = "forager/output/pairwise_cosine_sim.csv"
    pairwise_sim_df.to_csv(csv_path, index=False)

calc_pairwise_sim("forager/data/fluency_lists/speech2vec_100.txt", "forager/data/fluency_lists/word2vec_100.txt")
