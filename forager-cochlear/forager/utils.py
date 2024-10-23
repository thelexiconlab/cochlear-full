import numpy as np
from scipy import stats
import pandas as pd
import os
import difflib
import nltk
import zipfile

def trunc(word, df):
    # function to truncate fluency list at word

    i = df[df['entry'] == word].index.values[0]
    sid = df.iloc[i]['SID']
    if 'timepoint' in df.columns:
        tp = df.iloc[i]['timepoint']
        list_rows = df[(df['SID'] == sid) & (df['timepoint'] == tp)].index.values
    else:
        list_rows = df[df['SID'] == sid].index.values
    j = list_rows[-1] + 1
    df.drop(df.index[i:j], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return None

def exclude(word,df):
    # function to exclude all instances of word from df
    df.drop(df[df['entry'] == word].index, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return None

# takes in a path to a data file to be read as a CSV, the first row will be assumed as a header 
# accepted delimiters include commas, tabs, semicolons, pipes, and spaces

def prepareDataWithCorrections(path, domain):
    ### LOAD BEHAVIORAL DATA ###
    df = pd.read_csv(path, header=0, engine='python', sep=None, encoding='utf-8-sig')
    
    df.columns = ['SID', 'entry']

    # load vocab labels
    labels = pd.read_csv('data/lexical_data/' + domain + '/vocab.csv', names=['word'], header=None) 

    # make corrections based on corrections file

    processed_df = df.copy()

    corrections = pd.read_excel('data/lexical_data/' + domain + '/corrections.xlsx')

    # set all replacements to actual word for all words in labels as the default
    replacements = {word: word for word in labels['word'].values}

    # get values from df 
    values = processed_df['entry'].values
    SID = processed_df['SID'].values
    # loop through values to find which ones are not in file and store a tuple of the SID and the word
    #oov = [(SID[i], w) for i, w in enumerate(values) if w not in labels['word'].values]
    oov = [w for w in values if w not in labels['word'].values]
    #print(oov)
    if len(oov) > 0:
        
        print("We did not find exact matches for " + str(len(oov)) + " items in our vocabulary. Any items for which we find a reasonable match will be automatically replaced. Other items will be excluded.")
        for word in oov:
            evaluation_df = processed_df.copy()
            #print("word: " + str(word))
            # get closest match in vocab and check edit distance
            closest_word = difflib.get_close_matches(word, labels['word'].values,1)

            if len(closest_word)>0 and nltk.edit_distance(word, closest_word[0]) <= 2:
                replacements[word] = closest_word[0]
            # if there is no close match in vocab, then look through corrections file for a replacement with difflib
            else:
                # find closest match in corrections file
                closest_word = difflib.get_close_matches(word, corrections['entry'].values,1)
                if len(closest_word)>0 and nltk.edit_distance(word, closest_word[0]) <= 2:
                    # if there is a close match in the corrections file, replace with the final word
                    replacements[word] = corrections[corrections['entry'] == closest_word[0]]['replacement'].values[0]
                else:
                    # exclude this word from the list if not found in corrections file or vocab
                    exclude(word, processed_df)
                    replacements[word] = "EXCLUDE"

        processed_df.replace(replacements, inplace=True)

        # remove consecutive duplicates from the corrected file for the same SID and entry
        # for example, if zebra is repeated twice consecutively, remove the second instance
        # but if zebra is repeated twice with another word in between, keep both instances
        processed_df = processed_df.loc[(processed_df['SID'] != processed_df['SID'].shift()) | (processed_df['entry'] != processed_df['entry'].shift())]
    
        ## creating the evaluation_results file
        ## needs to contain SID, entry, evaluation decision, replacement

        # add an extra column to orig_df with the replacement word

        evaluation_df['evaluation'] = evaluation_df['entry'].map(replacements)
        # create a new column 'replacement' that is a copy of 'evaluation'
        evaluation_df['replacement'] = evaluation_df['evaluation']
        # now replace all instances in evalution where the entry doesn't match the replacement AND isn't within
        # ['UNK', 'EXCLUDE', 'TRUNCATE'] with 'REPLACE'
        evaluation_df.loc[(evaluation_df['entry'] != evaluation_df['evaluation']) & (~evaluation_df['evaluation'].isin(['UNK', 'EXCLUDE', 'TRUNCATE'])), 'evaluation'] = 'REPLACE'
        # also for the column 'evaluation', if entry matches evaluation, replace with 'found'
        evaluation_df.loc[(evaluation_df['entry'] == evaluation_df['evaluation']), 'evaluation'] = 'FOUND'

        # now I want to create a corrections_df that contains all the replacements that were made EXCLUDING the consecutive duplicates

        # generate a corrected_df where any corrections that were over 2 edit distance are recorded for later use
        # compute edit distance for each entry in replacement_df
        corrected_df = evaluation_df.copy()
        # remove any rows where there are consecutive duplicates in the replacement column
        corrected_df = corrected_df.loc[(corrected_df['replacement'] != corrected_df['replacement'].shift()) | (corrected_df['SID'] != corrected_df['SID'].shift())]
        # filter out any rows where the edit distance is less than 2
        # to account for minor spelling errors/corrected words
        
        corrected_df['edit_distance'] = corrected_df.apply(lambda x: nltk.edit_distance(x['entry'], x['replacement']), axis=1)
        
        corrected_df = corrected_df[corrected_df['edit_distance'] > 2]

        # remove duplicates from the corrected_df
        corrected_df = corrected_df.drop_duplicates(subset=['SID', 'entry', 'replacement', 'evaluation'], keep='first')

        exclude_count = (evaluation_df["evaluation"] == "EXCLUDE").sum()
        unk_count = (evaluation_df["evaluation"] == "UNK").sum()
        trunc_count = (evaluation_df["evaluation"] == "TRUNCATE").sum()
        replacement_count = (evaluation_df["evaluation"] == "REPLACE").sum()
        print("We have found reasonable replacements for " + str(replacement_count)+ " item(s) in your data. \n\n Consecutive duplicates will be removed.")
        if exclude_count>0:
            print(str(exclude_count) + " items were excluded across all lists.\n")
        elif unk_count>0: 
            print(str(unk_count) + " items were assigned a random vector across all lists.\n")
        elif trunc_count>0:
            print("Lists were truncated at " + str(trunc_count) + " items across all lists.\n")
        
        # Stratify data into fluency lists
        data = []
        lists = processed_df.groupby("SID")
        
        for sub, frame in lists:
            list = frame["entry"].values.tolist()
            subj_data = (sub, list)
            data.append(subj_data)

        return data, evaluation_df, processed_df, corrected_df
    
    else:
        print("Success! We have found exact matches for all items in your data. \n\n")
        replacement_df = processed_df.copy()
        replacement_df['evaluation'] = "FOUND"
        # Add the column corresponding to the replacement column , set it all to the same value   
        data = []
        lists = processed_df.groupby("SID")
        
        for sub, frame in lists:
            list = frame["entry"].values.tolist()
            subj_data = (sub, list)
            data.append(subj_data)

        # create an empty dataframe for the corrected_df that has the same columns as replacement_df
        corrected_df = pd.DataFrame(columns=replacement_df.columns)
        
        return data, replacement_df, processed_df, corrected_df


