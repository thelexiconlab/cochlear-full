import pandas as pd
import difflib
import math
import numpy as np
from scipy import stats

'''
Args: 
    (1) column_title: the string title of the column or index level to on (must be found in both dataframes)

Returns:
    (1) merged_df: a merged dataframe containing data from both df1 and df2 (currently static and assigned inside
    the function itself)
'''
def merge_results_hearing_status(column_title):
    df1 = pd.read_csv("forager/output/corrected_forager_results/individual_descriptive_stats.csv")
    #print(df1)
    df2 = pd.read_csv("forager/data/fluency_lists/cochlear_status_data.csv")
    #print(df2)
    results_path = 'forager/output/corrected_forager_results/merge_results.csv'
    merged_df = pd.merge(df1,df2, on=column_title, how='left')
    merged_df.to_csv(path_or_buf=results_path)
    return merged_df


'''
Args:
    (1) file path: the string file path to the file, which will be split on spaces

Returns:
    (1) speech2vec_vocab.csv: a file in the lexical_data folder that conatins a list of all unique words 
    extracted from the above file
'''
def extract_unique(file_path):
    open_file = open(file_path, "r")
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
    (1) vocab_file_path: the string file path to the vocab file
    (2) data_file_path: the string file path to the data file, to be evaluated against the larger vocab file
    (3) data_csv_path: the data file in .csv form
    (4) column_title: the string title of the column in the data file, from which one extracts data to be 
    compared to the vocab file

Returns:
    (1) speech2vec_vocab_check.csv: a file in the output folder containing the data entries, whether or not 
    they were present in the vocab, and the difflib get_close_matches from the vocab 
'''
def vocab_check(vocab_file_path, data_file_path, data_csv_path, column_title):
    food_data_df = pd.read_csv(data_csv_path)
    
    vocab = open(vocab_file_path, "r")
    speech2vec_vocab_list = []
    vocab_lines = vocab.readlines()
    label_skipped = vocab_lines[1: ]
    for line in label_skipped:
        split_line = line.split(" ")
        speech2vec_vocab_list.append(split_line[0])
    
    data = open(data_file_path, "r")
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
    (1) csv_file_path: the string file path to the .csv file containing the data to be unduplicated
    (2) column_title: the string title of the column in the .csv file, from which one extracts data

Returns:
    (1) food_data_no_dups.csv: a file in the fluency_lists folder containing the data from the 
    specified column of the .csv file, with no duplicate entries
'''
def unduplicate(csv_file_path, column_title):
    csv_df = pd.read_csv(csv_file_path)
    vocab = open(csv_file_path, "r")
    word_set = set([])
    unduped_df = pd.DataFrame()
    for item in csv_df.loc[:, (column_title)]:
        word_set.add(item)
    unduped_df["unduplicated entries"] = list(word_set)
    csv_path = "forager/data/fluency_lists/food_data_no_dups.csv"
    unduped_df.to_csv(csv_path, index=False)

'''
Args:
    (1) filename_path: the string file path to the file from which vocab data will be extracted

Returns:
    (2) vocab_list: a list containing all individual data entries from filename, minus the first row 
    (often the column label)
'''
def read_extract_list(filename_path):
    vocab = open(filename_path, "r")
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
    (3) filename_path: the string file path to the file from which a word (key): list of float embeddings (value)
    dictionary will be constructed and utilized to calculate the cosine similarity between word1 and word2

Returns:
    (1) cosine_sim: the float represntation of the cosine similarity between word1 and word2, formatted in 
    this template - "Cosine Similarity: x"
'''
def cosine_similarity(word1, word2, filename_path):
    file = open(filename_path, "r")
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
    (1) food_data_csv_path: the string file path to the .csv file of the food data

Returns:
    (1) full_df: a dataframe including pairwise cosine similarity calculations using speech2vec and
    word2vec, as well as composite frequency analysis
    (2) optimized_pairwise_cosine_sim.csv: a .csv version of the full_df, routed to the output of forager
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
    return full_df

'''
Args:
    (1) id_entries: a list of the entries and fluency items generated, specific to a single individual (ID)

Returns:
    (1) id_df: a dataframe containing the pairiwse cosine similairty calculated from speech2vec and word2vec
    dictionaries, phonological similairty calculations from scratch, and composite frequency analysis for that 
    specific individual (ID)

'''
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
    (2) subject_column_title: the string title of the column in the .csv file, from which one extracts 
    subject data

Returns:
    (1) subject_list: a list of each individual subject extracted from the food_data_csv 
'''
def extract_subject(food_data_csv_path, subject_column_title):
    food_csv_df = pd.read_csv(food_data_csv_path)
    subject_list = []

    for item in food_csv_df.loc[:, (subject_column_title)]:
        subject_list.append(item)
    return subject_list

'''
Args:
    (1) food_data_csv_path: the string file path to the .csv file of the food data
    (2) entry_column_title: the string title of the column in the .csv file, from which one extracts 
    entry data

Returns:
    (1) entry_list: a list of the food fluency items extracted from the food_data_csv
'''
def extract_entries(food_data_csv_path, entry_column_title):
    food_csv_df = pd.read_csv(food_data_csv_path)
    entry_list = []

    for item in food_csv_df.loc[:, (entry_column_title)]:
        entry_list.append(item)
    return entry_list

'''
Args:
    (1) dict_file_path: the string file to the .txt from which to extract a word (key): embeddings (value) 
    dictionary
    (2) 

Returns:
    (1) pair_cosines: a list of pairwise cosine similarity calculations for the data in the data_csv using
    a dictionary of embeddings extracted from the dict_file
    (2) entries: a list of the data that will be compared and pairwise cosine similarity will be
    calculated
'''
def pairwise_sim(dict_file_path, entries):
    #entry_list = extract_entries(data_csv, "entry")
    #print(entry_list)

    pair_cosines = []
    idx = 0

    while idx < len(entries):
        if idx > 0:
            try:
                currentwordindex = idx
                prevwordindex = idx-1
                curr_speech_cosine = cosine_similarity(entries[prevwordindex], entries[currentwordindex], dict_file_path)
                #print(f"cosine between {entry_list[currentwordindex]} and {entry_list[prevwordindex]} =", curr_speech_cosine)
                pair_cosines.append(curr_speech_cosine)
            except KeyError:
                pair_cosines.append("NA")
        else:
            try:
                pair_cosines.append(2)
            except KeyError:
                pair_cosines.append("NA")
        # print(idx)
        # print(pair_cosines)
        idx += 1
    #print(pair_cosines)
    return pair_cosines

'''
Args:
    (1) entries: a list of the data that will be compared and pairwise phonological similarity will be
    calculated
    (2) vocab_csv_path: the string file path to the vocabulary specific to this domain of data
    (3) phon_matrix_path: the string file path to the phonological matrix that is specific to this domain of data

Returns:
    (1) pairwise_phons: a list of the pairwise phonological similarity calculations
'''
def pairwise_phon(entries, vocab_csv_path, phon_matrix):
    phon_matrix_df = pd.read_csv(phon_matrix, header=None)
    vocab_df = pd.read_csv(vocab_csv_path)

    vocab = {}
    for index,word in vocab_df.itertuples():
        vocab[word] = index
    
    pairwise_phons = []

    idx = 0
    while idx < len(entries):
        curr_word = entries[idx]
        #print(curr_word)
        try:
            curr_index = vocab[curr_word]
            if idx > 0:
                prev_word = entries[idx - 1]
                try:
                    prev_index = vocab[prev_word]
                    phon_sim = phon_matrix_df.loc[curr_index, prev_index]
                    pairwise_phons.append(phon_sim)
                except KeyError:
                    pairwise_phons.append("NA")
            else:
                pairwise_phons.append(0.0001)  # default value for the first entry
        except KeyError:
            pairwise_phons.append("NA")  # handle missing entry in vocab
        idx += 1

    #print("length of pairwise_phon:", len(pairwise_phons))
    return pairwise_phons

'''
Args:
    (1) entries: list of the entries/fluency items for a specific individual
    (2) freq_csv_path: the string file path to the .csv file containing the frequency values pertaining to each 
    word existent in the vocab

Returns:
    (1) frequencies: a list of the frequency values corresponding to the specfic entries for that individual
'''
def get_frequency(entries, freq_csv_path):
    freq_df = pd.read_csv(freq_csv_path,header=None)
    freq_df.columns = ["Entry", "Frequency_Value", "Count"]
    #print(freq_df.head())
    entries_df = pd.DataFrame(entries, columns=["Entry"])
    #print(entries_df.head())

    merged_df = pd.merge(entries_df, freq_df, on=['Entry'], how='left')
    frequencies = merged_df['Frequency_Value'].tolist()

    # print(merged_df)
    # print(len(merged_df))
    # print(frequencies)
    return frequencies

'''
Hypothesis Testing Code: evaluating pairiwse summary statistics calcuated for both NHs and CIs for phonological
transitions (zero and non-zero) and frequency transitions (high and low)

Type of test: independent groups t-test
'''
#Define null and alternative hypothesis
H0 = "µ1 - µ2 = 0"
H1 = "µ1 - µ2 ≠ 0"

# highfreqspeech = np.array([0.46101395,0.78126644,0.57182969,0.60923051,0.48196248,0.76742733,0.58024541,0.75872467,0.57140667,0.53674852,
# 0.68947163,0.05923254,0.48671027,0.60923051,0.53639944,0.75060368,0.45960866,0.42754948,0.45344726,0.76165103,0.72494254,
# 0.60971805,0.46196269,0.45344726,0.60681354,0.79752049,0.72859427,0.42153462,0.55833862,0.46196269,0.24823068,0.56133895,
# 0.26600549,0.46641228,0.47936238,0.78927816,0.53639944,0.5727735,0.22270582,0.46759197,0.71854073,0.56501095,0.51810404,
# 0.58776131,0.48524187,0.42619812,0.54787422,0.62697787,0.42153462,0.70053456,0.31372119,0.46097233,0.70720228,0.45291067,
# 0.65030282,0.58379138,0.58514885,0.51810404,0.56511658,0.5938543,0.55833862,0.66765506,0.60224544,0.50315638,0.5539853,
# 0.51494506,0.4540556,0.44791648,0.71221133,0.31539222,0.66315901,0.7731759,0.62087735,0.37194249,0.60923051,0.5727735,0.53301606,
# 0.63153328,0.5578239,0.71721292,0.45291067,0.64176307,0.46328491,0.61638127,0.64996378,0.42582355,0.58940987,0.69833422,0.51251159,
# 0.59197653,0.58997361,0.45291067,0.38803912,0.73722902,0.71721292,0.67739943,0.82073738,0.41428464,0.4946309,0.40394824,
# 0.62095902,0.70053456,0.59022601,0.49142019,0.69485177,0.84435479,0.5596579,0.42153462,0.56958424,0.46097233,0.46047775,
# 0.56946303,0.21675719,0.43200698,0.55896951,0.46759197,0.5727735,1,0.54046257,0.57665843,0.67559584,0.50608772,0.42153462,
# 0.43066077,0.55627205,0.52252796,0.55062902,0.69918827,0.7216587,0.60923051,0.65743006,0.51149005,0.59256761,0.68688372,
# 0.47925742,0.60923051,0.41428464,0.14366465,0.02024647,0.35510334,0.43775224,0.48524187,0.4978939,0.5913379,0.56143249,
# 0.56946303,0.48901206,0.60971805,0.78140331,0.87116092,0.48742953,0.41448371,0.3244383,0.62751843,0.58024541,0.54282226,
# 0.45291067,0.41428464,0.54741443,0.58379138,0.31372119,0.60923051,0.28214241,0.44827834,0.45725007,0.39780324,0.43078537,
# 0.71854073,0.52325037,0.45291067,0.55835741,0.33488261,0.61693128,0.67042044,0.38658019,0.10926012,0.60923051,0.68834338,
# 0.56511658,0.42153462,0.59188825,0.76321645,0.53639944,0.51335511,0.52060404,0.56902925,0.63945005,0.57665843,0.46196269,
# 0.53639944,0.82204593,0.52988861])
# highfreqword = np.array([0.60094953,0.79391192,0.56847563,0.62051227,0.515359,0.7271302,0.64206224,0.75274081,0.58242651,0.49117792,
# 0.67694054,0.34339751,0.61212394,0.62051227,0.58136539,0.72804288,0.36945867,0.43320294,0.51929822,0.81043633,0.76805737,0.68589006,
# 0.61538483,0.51929822,0.59842771,0.80444751,0.71328276,0.49398222,0.51556235,0.61538483,0.26814916,0.55652854,0.44928927,
# 0.52165005,0.40207253,0.69036884,0.58136539,0.57199263,0.32364296,0.48733253,0.6966642,0.66324256,0.53475799,0.63592572,0.59534881,
# 0.48993183,0.58767195,0.62982573,0.49398222,0.73211272,0.27586573,0.48251478,0.70384314,0.41250202,0.72186793,0.58689981,0.57732845,
# 0.53475799,0.62934154,0.52059134,0.51556235,0.70823048,0.63434005,0.5902645,0.51572954,0.40429487,0.44432571,0.43279124,0.65819109,
# 0.3863304,0.68668677,0.78101327,0.70575666,0.5313123,0.62051227,0.57199263,0.57551978,0.6872776,0.59567617,0.67157879,0.41250202,
# 0.651382,0.48705582,0.70527177,0.63958008,0.41714831,0.55496226,0.74690783,0.55799849,0.6692154,0.597538,0.41250202,0.40433245,
# 0.71721808,0.67157879,0.59460008,0.82576904,0.34090893,0.59483595,0.40463648,0.61525143,0.73211272,0.594788,0.35943208,0.70180479,0.77605476,
# 0.66537397,0.49398222,0.58458443,0.48251478,0.45534589,0.48272523,0.2636781,0.32498346,0.65682196,0.48733253,0.57199263,1,
# 0.5598756,0.69100039,0.64182602,0.49528744,0.49398222,0.47810896,0.63350467,0.6134468,0.82135208,0.63929426,0.70698485,0.62051227,
# 0.67895959,0.54429354,0.6981299,0.64669408,0.5163242,0.62051227,0.34090893,0.3232072,0.28885835,0.4537014,0.50243514,0.59534881,
# 0.56876746,0.62537736,0.5511947,0.48272523,0.48679807,0.68589006,0.76399915,0.72517188,0.52125482,0.43963695,0.38401916,0.63260762,
# 0.64206224,0.56146111,0.41250202,0.34090893,0.57965143,0.58689981,0.27586573,0.62051227,0.40444024,0.58971541,0.60945879,
# 0.37383486,0.49773624,0.6966642,0.67932288,0.41250202,0.61157537,0.70664006,0.68749434,0.69959005,0.33435483,0.18423387,
# 0.62051227,0.60541344,0.62934154,0.49398222,0.67720272,0.76450735,0.58136539,0.53000794,0.65382138,0.66491567,0.67751375,
# 0.69100039,0.61538483,0.58136539,0.76105386,0.50591456]) 

# lowfreqspeech = np.array([
# 0.8864095,0.84146442,0.6164507,0.45548024,0.40456989,0.66567393,0.7335888,0.56757264,0.67797428,0.37881307,0.79286768,
# 0.90557549,0.83337246,0.50510211,0.64475994,0.70405504,0.78779125,0.83893252,0.80824284,0.67797428,0.88299431,0.76834881,
# 0.74900527,0.86981976,0.7690936,0.67797428,0.62096617,0.40803382,0.82785737,0.77720785,0.9164338,0.85806982,0.46875892,
# 0.50737929,0.70081368,0.76983423,0.60193855,0.68966875,0.66619498,0.65173533,0.58097155,0.74900527,0.88299431,0.88780424,
# 0.78110561,0.55535007,0.86924582,0.80824284,0.90554917,0.74035923,0.51443528,0.48293696,0.79412301,0.7840286,0.65185888])
# lowfreqword = np.array([0.79068509,0.75240787,0.56608224,0.47510692,0.54118704,0.66010553,0.71591982,0.59014091,0.63423526,
# 0.48737359,0.64981601,0.77928544,0.80994316,0.60145776,0.7463535,0.74197927,0.7871602,0.74533795,0.74677735,0.63423526,
# 0.77186269,0.65946358,0.68194436,0.82372363,0.67948178,0.63423526,0.52962973,0.39093396,0.69983835,0.65740545,0.70524805,
# 0.94446971,0.56619536,0.55478047,0.65241526,0.67518191,0.58596207,0.62769889,0.47628865,0.68944664,0.52520305,0.68194436,
# 0.77186269,0.62491384,0.84303972,0.5869747,0.69598193,0.74677735,0.72696247,0.71921712,0.65774328,0.52100715,0.76104774,
# 0.75520623,0.74020882])

# nonzerospeech = np.array([0.46101395,0.57182969,0.60923051,0.8864095,0.76742733,0.58024541,0.75872467,0.84146442,0.57140667,
# 0.60923051,0.53639944,0.75060368,0.6164507,0.42754948,0.45344726,0.60971805,0.46196269,0.45344726,0.7335888,0.42153462,
# 0.55833862,0.67797428,0.46196269,0.56133895,0.37881307,0.46641228,0.78927816,0.53639944,0.5727735,0.22270582,0.51810404,
# 0.58776131,0.54787422,0.83337246,0.42153462,0.50510211,0.65030282,0.58379138,0.58514885,0.51810404,0.70405504,0.5938543,
# 0.55833862,0.66765506,0.50315638,0.5539853,0.51494506,0.44791648,0.78779125,0.71221133,0.66315901,0.37194249,0.60923051,
# 0.5727735,0.5578239,0.80824284,0.64176307,0.46328491,0.61638127,0.67797428,0.42582355,0.74900527,0.69833422,0.51251159,
# 0.38803912,0.67797428,0.62096617,0.40803382,0.62095902,0.82785737,0.84435479,0.42153462,0.56958424,0.46047775,0.56946303,
# 0.21675719,0.55896951,0.77720785,0.85806982,0.50737929,0.5727735,1,0.70081368,0.76983423,0.42153462,0.43066077,0.55062902,
# 0.65173533,0.60923051,0.65743006,0.51149005,0.59256761,0.68688372,0.47925742,0.60923051,0.35510334,0.56946303,0.60971805,
# 0.87116092,0.48742953,0.41448371,0.74900527,0.62751843,0.58024541,0.55535007,0.86924582,0.58379138,0.60923051,0.44827834,
# 0.39780324,0.43078537,0.52325037,0.80824284,0.90554917,0.33488261,0.61693128,0.74035923,0.10926012,0.60923051,0.42153462,
# 0.76321645,0.53639944,0.51335511,0.52060404,0.56902925,0.63945005,0.51443528,0.48293696,0.46196269,0.53639944,0.79412301,
# 0.7840286,0.65185888])
# nonzeroword = np.array([0.60094953,0.56847563,0.62051227,0.79068509,0.7271302,0.64206224,0.75274081,0.75240787,0.58242651,
# 0.62051227,0.58136539,0.72804288,0.56608224,0.43320294,0.51929822,0.68589006,0.61538483,0.51929822,0.71591982,0.49398222,
# 0.51556235,0.63423526,0.61538483,0.55652854,0.48737359,0.52165005,0.69036884,0.58136539,0.57199263,0.32364296,0.53475799,
# 0.63592572,0.58767195,0.80994316,0.49398222,0.60145776,0.72186793,0.58689981,0.57732845,0.53475799,0.74197927,0.52059134,
# 0.51556235,0.70823048,0.5902645,0.51572954,0.40429487,0.43279124,0.7871602,0.65819109,0.68668677,0.5313123,0.62051227,0.57199263,
# 0.59567617,0.74677735,0.651382,0.48705582,0.70527177,0.63423526,0.41714831,0.68194436,0.74690783,0.55799849,0.40433245,
# 0.63423526,0.52962973,0.39093396,0.61525143,0.69983835,0.77605476,0.49398222,0.58458443,0.45534589,0.48272523,0.2636781,
# 0.65682196,0.65740545,0.94446971,0.55478047,0.57199263,1,0.65241526,0.67518191,0.49398222,0.47810896,0.82135208,0.68944664,
# 0.62051227,0.67895959,0.54429354,0.6981299,0.64669408,0.5163242,0.62051227,0.4537014,0.48272523,0.68589006,0.72517188,
# 0.52125482,0.43963695,0.68194436,0.63260762,0.64206224,0.5869747,0.69598193,0.58689981,0.62051227,0.58971541,0.37383486,
# 0.49773624,0.67932288,0.74677735,0.72696247,0.70664006,0.68749434,0.71921712,0.18423387,0.62051227,0.49398222,0.76450735,
# 0.58136539,0.53000794,0.65382138,0.66491567,0.67751375,0.65774328,0.52100715,
# 0.61538483,0.58136539,0.76104774,0.75520623,0.74020882])

# zerospeech = np.array([2,0.78126644,0.48196248,0.66036225,0.69547423,0.66692678,0.59747385,2,0.53674852,0.68947163,
# 0.05923254,0.48671027,0.84738136,0.72163788,0.90724256,2,0.45960866,0.54698643,0.60134014,0.45548024,0.40456989,0.76165103,
# 0.72494254,2,0.66567393,0.60681354,0.79752049,0.72859427,0.56757264,2,0.24823068,0.26600549,0.47936238,2,0.69547423,
# 0.78181446,0.79687211,0.64369987,0.24623912,0.46759197,0.71854073,2,0.56501095,2,0.48524187,0.42619812,0.09084484,
# 0.79286768,0.90557549,0.52784469,0.62697787,0.69908549,0.72630968,0.70053456,0.64475994,0.69442653,2,0.31372119,0.59420433,
# 0.60799343,0.46097233,0.70720228,0.45291067,0.43090379,0.5834551,0.85310405,0.93218221,2,0.56511658,0.60224544,2,0.4540556,
# 0.25373881,0.09084484,0.31539222,0.7731759,0.62087735,2,0.4554309,0.62773017,0.69547423,0.83893252,0.53301606,
# 0.63153328,2,0.65144664,0.65340207,0.69547423,0.77102255,0.71721292,0.45291067,0.65287447,0.09084484,2,0.32126709,
# 0.40813803,2,0.64996378,0.58940987,0.88299431,2,0.69442653,2,0.76834881,0.86981976,0.8207172,0.78044992,0.62087008,
# 0.59197653,0.58997361,0.45291067,0.7690936,0.63997063,0.73722902,0.71721292,0.67739943,0.82073738,0.36637692,2,0.41428464,
# 0.33297203,2,0.4946309,0.40394824,0.70053456,0.59022601,0.49142019,0.69485177,0.5596579,0.5109655,2,0.46097233,0.65580308,
# 0.69547423,0.43200698,0.64904957,0.8110427,2,0.28233583,0.34186219,0.58594272,0.68791136,0.90535682,0.9164338,0.59551597,
# 0.5824481,2,0.46759197,2,0.46875892,0.42979666,2,0.54046257,0.32810229,0.57665843,0.67559584,0.60193855,2,0.37245645,
# 0.68966875,0.66619498,0.50608772,0.33975339,0.32126709,0.54107726,0.61167326,0.65798532,2,0.55627205,0.40731392,
# 0.52252796,0.54698643,0.58097155,0.69918827,0.7216587,2,0.72201137,2,0.32791056,0.55521901,2,0.41428464,0.14366465,
# 0.02024647,0.43775224,0.48524187,0.4978939,0.5913379,0.56143249,2,0.66552905,0.50986104,0.72625139,0.363064,0.44501941,
# 2,0.48901206,0.78140331,0.59420433,0.69033501,0.41022718,0.3244383,0.88299431,0.83793998,0.90730788,0.88780424,2,
# 0.54282226,0.45291067,0.69105289,0.67000791,0.78110561,2,0.41428464,0.32126709,0.45532172,0.54741443,0.69547423,
# 0.69105289,0.6957063,2,0.24018376,0.37704195,0.32526052,2,0.31372119,0.28214241,0.45725007,0.71854073,2,0.68425881,
# 0.65400137,0.42868814,0.45291067,0.55835741,2,0.61745388,0.74186812,0.82110134,0.67042044,2,0.38658019,0.80536029,0.68834338,
# 2,0.56511658,0.59188825,2,2,0.57665843,2,0.82204593,2,0.66692678,0.41612514,0.52988861,2,0.67683718])
# zeroword = np.array([2,0.79391192,0.515359,0.5838648,0.64394536,0.64101077,0.56652815,2,0.49117792,0.67694054,0.34339751,
# 0.61212394,0.74564516,0.7671924,0.83245343,2,0.36945867,0.56083809,0.68239245,0.47510692,0.54118704,0.81043633,0.76805737,
# 2,0.66010553,0.59842771,0.80444751,0.71328276,0.59014091,2,0.26814916,0.44928927,0.40207253,2,0.64394536,0.55664508,0.7335092,
# 0.68861356,0.29516389,0.48733253,0.6966642,2,0.66324256,2,0.59534881,0.48993183,0.18378153,0.64981601,0.77928544,0.66321267,
# 0.62982573,0.46461255,0.51640651,0.73211272,0.7463535,0.58228701,2,0.27586573,0.61284327,0.57380586,0.48251478,0.70384314,
# 0.41250202,0.29664992,0.48795307,0.80127798,0.79562024,2,0.62934154,0.63434005,2,0.44432571,0.36927062,0.18378153,0.3863304,
# 0.78101327,0.70575666,2,0.44073423,0.63311693,0.64394536,0.74533795,0.57551978,0.6872776,2,0.64837053,0.71256805,0.64394536,
# 0.63557539,0.67157879,0.41250202,0.61526211,0.18378153,2,0.24386125,0.39624279,2,0.63958008,0.55496226,0.77186269,2,
# 0.58228701,2,0.65946358,0.82372363,0.75195885,0.77814229,0.55317667,0.6692154,0.597538,0.41250202,0.67948178,0.52565571,
# 0.71721808,0.67157879,0.59460008,0.82576904,0.30327722,2,0.34090893,0.41453794,2,0.59483595,0.40463648,0.73211272,0.594788,
# 0.35943208,0.70180479,0.66537397,0.51650018,2,0.48251478,0.62343671,0.64394536,0.32498346,0.6348638,0.77003642,2,
# 0.33323186,0.25221369,0.53228734,0.6821789,0.83331259,0.70524805,0.53579871,0.56119236,2,0.48733253,2,0.56619536,0.50146183,
# 2,0.54598756,0.41545371,0.69100039,0.64182602,0.58596207,2,0.39548026,0.62769889,0.47628865,0.49528744,0.29992182,0.24386125,
# 0.67120564,0.67057055,0.64598591,2,0.63350467,0.46054605,0.6134468,0.56083809,0.52520305,0.63929426,0.70698485,2,0.71668221,2,
# 0.37440031,0.59137583,2,0.34090893,0.3232072,0.28885835,0.50243514,0.59534881,0.56876746,0.62537736,0.5511947,2,0.54352194,0.27994529,
# 0.5802749,0.51248252,0.43567563,2,0.48679807,0.76399915,0.61284327,0.62453225,0.23005762,0.38401916,0.77186269,0.71401927,0.67484454,
# 0.62491384,2,0.56146111,0.41250202,0.68921114,0.68460237,0.84303972,2,0.34090893,0.24386125,0.40159381,0.57965143,0.64394536,
# 0.68921114,0.6186046,2,0.14573531,0.37885719,0.31734006,2,0.27586573,0.40444024,0.60945879,0.6966642,2,0.62688332,0.74094827,
# 0.46484518,0.41250202,0.61157537,2,0.6310249,0.677289,0.82806998,0.69959005,2,0.33435483,0.70480168,0.60541344,2,0.62934154,
# 0.67720272,2,2,0.69100039,2,0.76105386,2,0.64101077,0.45494416,0.50591456,2,0.72658966,])

'''
Data used is commented out above.

Hypothesis Testing Code: evaluating pairiwse summary statistics calcuated for both NHs and CIs for phonological
transitions (zero and non-zero) and frequency transitions (high and low)

Type of test: independent groups t-test

Args:
    (1) dataset1: a dataset of the dependent variable of one group
    (2) dataset1: a dataset of the dependent variable of a second group

Returns:
    (1) A decision conerning the null hypothesis based on the alpha level 0.5 and the t-statistic calculated 
    between groups.
'''

#Define null and alternative hypothesis
H0 = "µ1 - µ2 = 0"
H1 = "µ1 - µ2 ≠ 0"

#t_stat, p_value = stats.ttest_ind(zerospeech,zeroword)

# alpha = 0.05
# print(len(zerospeech))
# print(len(zeroword))
# if p_value < alpha:
#     print(t_stat)
#     print(p_value)
#     print("Reject null hypothesis: There is a significant difference between the groups.")
# else:
#     print(t_stat)
#     print(p_value)
#     print("Fail to reject null hypothesis: There is no significant difference between the groups.")

'''
Args:
    (1) food_data_csv_path: the string file path to the .csv file containing the food data in its entirety
    (2) replacements_csv_path: the string file path to the .csv file containing the food data exclusions and the 
    corresponding speech2vec replacements

Returns:
    (1) merged_df: a dataframe containing participant ids, initial entries, necessary speech2vec replacements,
    and pairiwse analysis of potential repeptitions between entries after replacements have been made
    (2) repetitions_after_relacement.csv: a .csv version of the merged_df, routed to the output folder of forager
'''
def repetitions_after_replacement(food_data_csv_path,replacements_csv_path):
    print("running")
    data_df = pd.read_csv(food_data_csv_path)
    replacement_df = pd.read_csv(replacements_csv_path)
    merged_df = pd.merge(data_df,replacement_df,on='entry',how='left')

    replacements_list = merged_df['Speech2vec_Replacement'].tolist()
    entry_list = data_df['entry'].tolist()
    merge_list = []
    
    idx = 0
    for value in replacements_list:
        if isinstance(value,str) == False:
            merge_list.append(entry_list[idx])
        else:
            merge_list.append(value)
        idx += 1
    merged_df['Replacements_Made'] = merge_list

    merged_df['Previous_Replacement'] = merged_df['Replacements_Made'].shift(1)
    merged_df['Replacement_Comparison'] = (merged_df['Speech2vec_Replacement'] == merged_df['Previous_Replacement']).astype(int)

    #print(merged_df)
    results_path = "forager/output/repetitions_after_relacement.csv"
    merged_df.to_csv(path_or_buf=results_path)
    print("done")

    return merged_df
