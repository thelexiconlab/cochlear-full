import argparse
from scipy.optimize import fmin
from forager.foraging import forage
from forager.switch import *
from forager.cues import create_history_variables
from forager.utils import *
import pandas as pd
import numpy as np
from scipy.optimize import fmin, minimize
import os, sys
from tqdm import tqdm
import warnings 
import zipfile

warnings.simplefilter('ignore')

"""
Workflow: 
1. Evaluate data
    a. "Prepare Data" 
        - takes path of fluency list ; 

2. Select use case (lexical, switches, models) via --pipeline
    a. Lexical: returns similarity & frequency values for each word in fluency list
    b. Switches: returns switch values for each word in fluency list + lexical values
    c. Models: returns model outputs for each word in fluency list + lexical values + switch values
"""

# Global Variables
models = ['static','dynamic','pstatic','pdynamic','all']
switch_methods = ['simdrop','multimodal','norms_associative','norms_categorical', 'delta', 'multimodaldelta', 'all']

#Methods
def retrieve_data(path, domain):
    """
    1. Verify that data path exists

    """
    if os.path.exists(path) == False:
        ex_str = "Provided path to data \"{path}\" does not exist. Please specify a proper path".format(path=path)
        raise Exception(ex_str)
    data = prepareDataWithCorrections(path, domain)
    return data

def get_lexical_data(domain, speech, dimension, alpha=None):
    print(domain, speech, dimension, alpha)

    animalnormspath =  'data/norms/animals_snafu_scheme_vocab.csv'
    foodnormspath =  'data/norms/foods_snafu_scheme_vocab.csv'
    if alpha is None:
        similaritypath =  'data/lexical_data/' + domain + '/' + speech + '/' + dimension + '/semantic_matrix.csv'
    else:
        similaritypath =  'data/lexical_data/' + domain + '/' + speech + '/' + dimension + '/' + str(alpha) + '/semantic_matrix.csv'

    frequencypath =  'data/lexical_data/' + domain + '/frequencies.csv'
    phonpath = 'data/lexical_data/' + domain + '/phonological_matrix.csv'

    animalnorms = pd.read_csv(animalnormspath, encoding="unicode-escape")
    foodnorms = pd.read_csv(foodnormspath, encoding="unicode-escape")
    norms = [animalnorms, foodnorms]
    similarity_matrix = np.loadtxt(similaritypath,delimiter=',')
    frequency_list = np.array(pd.read_csv(frequencypath,header=None,encoding="unicode-escape")[1])
    phon_matrix = np.loadtxt(phonpath,delimiter=',')
    labels = pd.read_csv(frequencypath,header=None)[0].values.tolist()
    
    return norms, similarity_matrix, phon_matrix, frequency_list,labels

def calculate_model(model, history_vars, switch_names, switch_vecs):
    """
    1. Check if specified model is valid
    2. Return a set of model functions to pass
    """
    model_name = []
    model_results = []
    if model not in models:
        ex_str = "Specified model is invalid. Model must be one of the following: {models}".format(models=models)
        raise Exception(ex_str)
    if model == models[0] or model == models[4]:
        r1 = np.random.rand()
        r2 = np.random.rand()

        v = minimize(forage.model_static, [r1,r2], args=(history_vars[2], history_vars[3], history_vars[0], history_vars[1])).x
        beta_df = float(v[0]) # Optimized weight for frequency cue
        beta_ds = float(v[1]) # Optimized weight for similarity cue
        
        nll, nll_vec = forage.model_static_report([beta_df, beta_ds], history_vars[2], history_vars[3], history_vars[0], history_vars[1])
        model_name.append('forage_static')
        model_results.append((beta_df, beta_ds, nll, nll_vec))
    if model == models[1] or model == models[4]:
        for i, switch_vec in enumerate(switch_vecs):
            r1 = np.random.rand()
            r2 = np.random.rand()

            v = minimize(forage.model_dynamic, [r1,r2], args=(history_vars[2], history_vars[3], history_vars[0], history_vars[1], switch_vec)).x
            beta_df = float(v[0]) # Optimized weight for frequency cue
            beta_ds = float(v[1]) # Optimized weight for similarity cue
            
            nll, nll_vec = forage.model_dynamic_report([beta_df, beta_ds], history_vars[2], history_vars[3], history_vars[0], history_vars[1],switch_vec)
            model_name.append('forage_dynamic_' + switch_names[i])
            model_results.append((beta_df, beta_ds, nll, nll_vec))
    if model == models[2] or model == models[4]:
        r1 = np.random.rand()
        r2 = np.random.rand()
        r3 = np.random.rand()
        v = minimize(forage.model_static_phon, [r1,r2,r3], args=(history_vars[2], history_vars[3], history_vars[0], history_vars[1], history_vars[4],history_vars[5])).x
        beta_df = float(v[0]) # Optimized weight for frequency cue
        beta_ds = float(v[1]) # Optimized weight for similarity cue
        beta_dp = float(v[2]) # Optimized weight for phonological cue

        nll, nll_vec = forage.model_static_phon_report([beta_df, beta_ds, beta_dp], history_vars[2], history_vars[3], history_vars[0], history_vars[1],history_vars[4],history_vars[5])
        model_name.append('forage_phonologicalstatic')
        model_results.append((beta_df, beta_ds, beta_dp, nll, nll_vec))
    if model == models[3] or model == models[4]:
        for i, switch_vec in enumerate(switch_vecs):
            # Global Dynamic Phonological Model
            r1 = np.random.rand()
            r2 = np.random.rand()
            r3 = np.random.rand()
            v = minimize(forage.model_dynamic_phon, [r1,r2,r3], args=(history_vars[2], history_vars[3], history_vars[0], history_vars[1],history_vars[4],history_vars[5], switch_vec,'global')).x
            beta_df = float(v[0]) # Optimized weight for frequency cue
            beta_ds = float(v[1]) # Optimized weight for similarity cue
            beta_dp = float(v[2]) # Optimized weight for phonological cue
            
            nll, nll_vec = forage.model_dynamic_phon_report([beta_df, beta_ds,beta_dp], history_vars[2], history_vars[3], history_vars[0], history_vars[1],history_vars[4],history_vars[5],switch_vec,'global')
            model_name.append('forage_phonologicaldynamicglobal_' + switch_names[i])
            model_results.append((beta_df, beta_ds, beta_dp, nll, nll_vec))
    
            # Local Dynamic Phonological Model
            r1 = np.random.rand()
            r2 = np.random.rand()
            r3 = np.random.rand()
            v = minimize(forage.model_dynamic_phon, [r1,r2,r3], args=(history_vars[2], history_vars[3], history_vars[0], history_vars[1],history_vars[4],history_vars[5], switch_vec,'local')).x
            beta_df = float(v[0]) # Optimized weight for frequency cue
            beta_ds = float(v[1]) # Optimized weight for similarity cue
            beta_dp = float(v[2]) # Optimized weight for phonological cue
            
            nll, nll_vec = forage.model_dynamic_phon_report([beta_df, beta_ds,beta_dp], history_vars[2], history_vars[3], history_vars[0], history_vars[1],history_vars[4],history_vars[5],switch_vec,'local')
            model_name.append('forage_phonologicaldynamiclocal_' + switch_names[i])
            model_results.append((beta_df, beta_ds, beta_dp, nll, nll_vec))
    
            # Switch Dynamic Phonological Model
            r1 = np.random.rand()
            r2 = np.random.rand()
            r3 = np.random.rand()
            v = minimize(forage.model_dynamic_phon, [r1,r2,r3], args=(history_vars[2], history_vars[3], history_vars[0], history_vars[1],history_vars[4],history_vars[5], switch_vec,'switch')).x
            beta_df = float(v[0]) # Optimized weight for frequency cue
            beta_ds = float(v[1]) # Optimized weight for similarity cue
            beta_dp = float(v[2]) # Optimized weight for phonological cue
            
            nll, nll_vec = forage.model_dynamic_phon_report([beta_df, beta_ds,beta_dp], history_vars[2], history_vars[3], history_vars[0], history_vars[1],history_vars[4],history_vars[5],switch_vec,'switch')
            model_name.append('forage_phonologicaldynamicswitch_' + switch_names[i])

            model_results.append((beta_df, beta_ds, beta_dp, nll, nll_vec))
    
    # Unoptimized Model
    model_name.append('forage_random_baseline')
    nll_baseline, nll_baseline_vec = forage.model_static_report(beta = [0,0], freql = history_vars[2], freqh = history_vars[3], siml = history_vars[0], simh = history_vars[1])
    model_results.append((0, 0, nll_baseline, nll_baseline_vec))
    return model_name, model_results

def calculate_switch(switch, fluency_list, semantic_similarity, phon_similarity, norms,domain, alpha = np.arange(0, 1.1, 0.1), rise = np.arange(0, 1.25, 0.25), fall = np.arange(0, 1.25, 0.25)):
    '''
    1. Check if specified switch model is valid
    2. Return set of switches, including parameter value, if required

    switch_methods = ['simdrop','multimodal','norms_associative', 'norms_categorical', 'delta','multimodaldelta','all']
    '''
    switch_names = []
    switch_vecs = []

    if switch not in switch_methods:
        ex_str = "Specified switch method is invalid. Switch method must be one of the following: {switch}".format(switch=switch_methods)
        raise Exception(ex_str)

    if switch == switch_methods[0] or switch == switch_methods[6]:
        switch_names.append(switch_methods[0])
        switch_vecs.append(switch_simdrop(fluency_list, semantic_similarity))

    if switch == switch_methods[1] or switch == switch_methods[6]:
        for i, a in enumerate(alpha):
            a = round(a, 1)
            switch_names.append('multimodal_alpha={alpha}'.format(alpha=a))
            switch_vecs.append(switch_multimodal(fluency_list, semantic_similarity, phon_similarity, a))

    if (switch == switch_methods[2] or switch == switch_methods[6]) and domain in ['animals','foods']:
        
        if domain == 'animals':
            switch_names.append("norms_associative")
            switch_vecs.append(switch_norms_associative(fluency_list,norms[0]))
        else:
            switch_names.append("norms_associative")
            switch_vecs.append(switch_norms_associative(fluency_list,norms[1]))
    
    if switch == switch_methods[3] or switch == switch_methods[6] and domain in ['animals','foods']:
        
        if domain == 'animals':
            switch_names.append("norms_categorical")
            switch_vecs.append(switch_norms_categorical(fluency_list,norms[0]))
        else:
            switch_names.append("norms_categorical")
            switch_vecs.append(switch_norms_categorical(fluency_list,norms[1]))

    if switch == switch_methods[4] or switch == switch_methods[6]:
        for i, r in enumerate(rise):
            for j, f in enumerate(fall):
                r = round(r, 1)
                f = round(f, 1)
                switch_names.append("delta_rise={rise}_fall={fall}".format(rise=r,fall=f))
                switch_vecs.append(switch_delta(fluency_list, semantic_similarity, r, f))
    
    if switch == switch_methods[5] or switch == switch_methods[6]:
        for i, a in enumerate(alpha):
            for i, r in enumerate(rise):
                for j, f in enumerate(fall):
                    # round a, r, f to 1 decimal places
                    a = round(a, 1)
                    r = round(r, 1)
                    f = round(f, 1)
                    switch_names.append("multimodaldelta_alpha={alpha}_rise={rise}_fall={fall}".format(alpha=a,rise=r,fall=f))
                    switch_vecs.append(switch_multimodaldelta(fluency_list, semantic_similarity, phon_similarity, r, f, a))

    return switch_names, switch_vecs

def run_model(data, model_type, switch_type, domain, speech, dimension, history_vars_list, switch_names_list, switch_vecs_list, alpha=None):
    # Get Lexical Data needed for executing methods
    norms, similarity_matrix, phon_matrix, frequency_list, labels =get_lexical_data(domain, speech, dimension, alpha)
    forager_results = []
    # Run through each fluency list in dataset
    for i, (subj, fl_list) in enumerate(tqdm(data)):
        print("\nRunning Model for Subject {subj}".format(subj=subj))
        import time
        start_time = time.time()
        # Get History Variables 
        #history_vars = create_history_variables(fl_list, labels, similarity_matrix, frequency_list, phon_matrix)
        history_vars = history_vars_list[i]
        
        # Calculate Switch Vector(s)
        #switch_names, switch_vecs = calculate_switch(switch_type, fl_list, history_vars[0],   history_vars[4], norms, domain)
        switch_names = switch_names_list[i]
        switch_vecs = switch_vecs_list[i]

        #Execute Individual Model(s) and get result(s)
        model_names, model_results = calculate_model(model_type,history_vars, switch_names, switch_vecs)

        #Create Model Output Results DataFrame
        for i, model in enumerate(model_names):
            model_dict = dict()
            model_dict['Subject'] = subj
            model_dict['Model'] = model
            model_dict['Beta_Frequency'] = model_results[i][0]
            model_dict['Beta_Semantic'] = model_results[i][1]
            # print(results[i])
            # sys.exit()
            if len(model_results[i]) == 4:
                model_dict['Beta_Phonological'] = None
                model_dict['Negative_Log_Likelihood_Optimized'] = model_results[i][2]
            if len(model_results[i]) == 5:
                model_dict['Beta_Phonological'] = model_results[i][2]
                model_dict['Negative_Log_Likelihood_Optimized'] = model_results[i][3]
            forager_results.append(model_dict)
    forager_results = pd.DataFrame(forager_results)
        
    return forager_results
    
def run_lexical(data, domain,speech, dimension, corrected_df, alpha=None):
    # Get Lexical Data needed for executing methods
    norms, similarity_matrix, phon_matrix, frequency_list, labels = get_lexical_data(domain, speech, dimension, alpha)
    lexical_results = []
    history_vars_list = []
    for i, (subj, fl_list) in enumerate(tqdm(data)):
        history_vars = create_history_variables(fl_list, subj, corrected_df, labels, similarity_matrix, frequency_list, phon_matrix)
        history_vars_list.append(history_vars)
        lexical_df = pd.DataFrame()
        lexical_df['Subject'] = len(fl_list) * [subj]
        lexical_df['Fluency_Item'] = fl_list
        lexical_df['Semantic_Similarity'] = history_vars[0]
        lexical_df['Frequency_Value'] = history_vars[2]
        lexical_df['Phonological_Similarity'] = history_vars[4]
        lexical_results.append(lexical_df)
    lexical_results = pd.concat(lexical_results,ignore_index=True)
    return lexical_results, history_vars_list

def run_switches(data,switch_type, domain, speech, dimension, history_vars_list,alpha=None):
    norms, similarity_matrix, phon_matrix, frequency_list, labels = get_lexical_data(domain, speech, dimension, alpha)
    switch_results = []
    switch_names_list = []
    switch_vecs_list = []
    for i, (subj, fl_list) in enumerate(tqdm(data)):
        #history_vars = create_history_variables(fl_list, labels, similarity_matrix, frequency_list, phon_matrix)
        history_vars = history_vars_list[i]
        switch_names, switch_vecs = calculate_switch(switch_type, fl_list, history_vars[0], history_vars[4], norms, domain)
        switch_vecs_list.append(switch_vecs)
        switch_names_list.append(switch_names)
    
        switch_df = []
        for j, switch in enumerate(switch_vecs):
            df = pd.DataFrame()
            df['Subject'] = len(switch) * [subj]
            df['Fluency_Item'] = fl_list
            df['Switch_Value'] = switch
            df['Switch_Method'] = switch_names[j]
            switch_df.append(df)
    
        switch_df = pd.concat(switch_df, ignore_index=True)
        switch_results.append(switch_df)
    switch_results = pd.concat(switch_results, ignore_index=True)
    return switch_results, switch_names_list, switch_vecs_list


def indiv_desc_stats(lexical_results, switch_results = None):
    metrics = lexical_results[['Subject', 'Semantic_Similarity', 'Frequency_Value', 'Phonological_Similarity']]
    # replace first row of each subject with NaN for Semantic_Similarity and Phonological_Similarity
    metrics.loc[metrics.groupby('Subject').head(1).index, ['Semantic_Similarity', 'Phonological_Similarity']] = np.nan
    # ungroup the DataFrame
    metrics = metrics.reset_index(drop=True)
    # group the DataFrame by Subject and calculate the mean and standard deviation of each column
    grouped = metrics.groupby('Subject').agg(['mean', 'std'])
    grouped.columns = ['{}_{}'.format(col[0], col[1]) for col in grouped.columns]
    grouped.reset_index(inplace=True)
    num_items = lexical_results.groupby('Subject')['Fluency_Item'].size()
    grouped['#_of_Items'] = num_items[grouped['Subject']].values
    # create column for each switch method per subject and get number of switches, mean cluster size, and sd of cluster size for each switch method
    if switch_results is not None:
        # count the number of unique values in the Switch_Method column of the switch_results DataFrame
        n_rows = len(switch_results['Switch_Method'].unique())
        new_df = pd.DataFrame(np.nan, index=np.arange(len(grouped) * (n_rows)), columns=grouped.columns)

        # Insert the original DataFrame into the new DataFrame but repeat the value in 'Subject' column n_rows-1 times

        new_df.iloc[(slice(None, None, n_rows)), :] = grouped
        new_df['Subject'] = new_df['Subject'].ffill()

        switch_methods = []
        num_switches_arr = []
        cluster_size_mean = []
        cluster_size_sd = []
        for sub, fl_list in switch_results.groupby(["Subject", "Switch_Method"]):
            switch_method = sub[1]
            cluster_lengths = []
            num_switches = 0
            ct = 0
            for x in fl_list['Switch_Value'].values:
                ct += 1
                if x == 1:
                    num_switches += 1
                    ct = ct-1
                    cluster_lengths.append(ct)
                    ct = 1
            if ct != 0:
                cluster_lengths.append(ct)
            
            avg = sum(cluster_lengths) / len(cluster_lengths)
            sd = np.std(cluster_lengths)
            switch_methods.append(switch_method)
            num_switches_arr.append(num_switches)
            cluster_size_mean.append(avg)
            cluster_size_sd.append(sd)


        new_df['Switch_Method'] = switch_methods
        new_df['Number_of_Switches'] = num_switches_arr
        new_df['Cluster_Size_mean'] = cluster_size_mean
        new_df['Cluster_Size_std'] = cluster_size_sd
        grouped = new_df
        
    return grouped

def agg_desc_stats(switch_results, model_results=None):
    agg_df = pd.DataFrame()
    # get number of switches per subject for each switch method
    switches_per_method = {}
    for sub, fl_list in switch_results.groupby(["Subject", "Switch_Method"]):
        method = sub[1]
        if method not in switches_per_method:
            switches_per_method[method] = []
        if 1 in fl_list['Switch_Value'].values:
            switches_per_method[method].append(fl_list['Switch_Value'].value_counts()[1])
        else: 
            switches_per_method[method].append(0)
    agg_df['Switch_Method'] = switches_per_method.keys()
    agg_df['Switches_per_Subj_mean'] = [np.average(switches_per_method[k]) for k in switches_per_method.keys()]
    agg_df['Switches_per_Subj_SD'] = [np.std(switches_per_method[k]) for k in switches_per_method.keys()]
    
    if model_results is not None:
        betas = model_results.drop(columns=['Subject', 'Negative_Log_Likelihood_Optimized'])
        betas.drop(betas[betas['Model'] == 'forage_random_baseline'].index, inplace=True)
        grouped = betas.groupby('Model').agg(['mean', 'std'])
        grouped.columns = ['{}_{}'.format(col[0], col[1]) for col in grouped.columns]
        grouped.reset_index(inplace=True)

        # add a column to the grouped dataframe that contains the switch method used for each model
        grouped.loc[grouped['Model'].str.contains('static'), 'Model'] += ' none'
        # if the model name starts with 'forage_dynamic_', ''forage_phonologicaldynamicglobal_', 'forage_phonologicaldynamiclocal_', or 'forage_phonologicaldynamicswitch_', replace the second underscore with a space
        switch_models = ['forage_dynamic_', 'forage_phonologicaldynamicglobal_', 'forage_phonologicaldynamiclocal_', 'forage_phonologicaldynamicswitch_']
        for model in switch_models:
            # replace only the second underscore with a space
            grouped.loc[grouped['Model'].str.contains(model), 'Model'] = grouped.loc[grouped['Model'].str.contains(model), 'Model'].str.replace('_', ' ', 2)
            grouped.loc[grouped['Model'].str.contains("forage "), 'Model'] = grouped.loc[grouped['Model'].str.contains("forage "), 'Model'].str.replace(' ', '_', 1)
        
        # split the Model column on the space
        grouped[['Model', 'Switch_Method']] = grouped['Model'].str.rsplit(' ', n=1, expand=True)

        # merge the two dataframes on the Switch_Method column 
        agg_df = pd.merge(agg_df, grouped, how='outer', on='Switch_Method')


    return agg_df
 
parser = argparse.ArgumentParser(description='Execute Semantic Foraging Code.')
parser.add_argument('--data', type=str,  help='specifies path to fluency lists')
parser.add_argument('--pipeline',type=str, help='specifies which part of pipeline (lexical, switches, models) to execute')
parser.add_argument('--model', type=str, help='specifies foraging model to use')
parser.add_argument('--switch', type=str, help='specifies switch model to use')
parser.add_argument('--domain', type=str, help='specifies domain to use')
parser.add_argument('--speech', type=str, help='specifies whether to use speech2vec or word2vec or blended')
parser.add_argument('--dimension', type=str, help='specifies which dimension embedding to use: 50/100/200/300')
parser.add_argument('--alpha', type=float, help='specifies which alpha to use for blended models (0 to 1)')


args = parser.parse_args()

if os.path.exists('output') == False:
    os.mkdir('output')

if args.data == None:
    parser.error("Please specify a data file for which you would like to run the forager pipeline for")

if args.pipeline == None:
    parser.error("Please specify which part of the forager pipeline you would like to execute for your data (e.g. \'lexical\', \'switches\',\'model\')")

args.data = os.path.join(os.getcwd(),args.data)
#oname = 'output/' + args.data.split('/')[-1].split('.')[0] + '_forager_results.zip'

output_dir = 'output'
file_name = 'forager_results.zip'

alpha_str = f"{args.alpha:.1f}" #convert float to str
# Construct the output path
oname = os.path.join(output_dir, args.domain, args.speech, args.dimension, alpha_str)

# Create the directory if it does not exist
os.makedirs(oname, exist_ok=True)

# Complete path with file name
file_path = os.path.join(oname, file_name)



if args.pipeline == 'evaluate_data':
    data, replacement_df, processed_df, corrected_df = retrieve_data(args.data, args.domain)
    replacement_df.to_csv(oname + '/evaluation_results.csv', index=False)
    print(f"File 'evaluation_results.csv' detailing the changes made to the dataset has been saved in '{oname}'")

    processed_df.to_csv(oname + '/processed_data.csv', index=False)
    print(f"File 'processed_data.csv' containing the processed dataset used in the forager pipeline saved in '{oname}'")

    corrected_df.to_csv(oname + '/corrected_df.csv', index=False)
    print(f"File 'corrected_df.csv' containing the corrections dataset used in the forager pipeline saved in '{oname}'")

    vocabpath = 'data/lexical_data/' + args.domain + '/vocab.csv'
    vocab = pd.read_csv(vocabpath, encoding="unicode-escape")
    vocab.to_csv(oname + '/forager_vocab.csv', index=False)
    print(f"File 'forager_vocab.csv' containing the full vocabulary used by forager saved in '{oname}'")

elif args.pipeline == 'lexical':
    # Retrieve the Data for Getting Lexical Info
    data, replacement_df, processed_df, corrected_df = retrieve_data(args.data, args.domain)
    replacement_df.to_csv(oname + '/evaluation_results.csv', index=False)
    print(f"File 'evaluation_results.csv' detailing the changes made to the dataset has been saved in '{oname}'")

    processed_df.to_csv(oname + '/processed_data.csv', index=False)
    print(f"File 'processed_data.csv' containing the processed dataset used in the forager pipeline saved in '{oname}'")

    corrected_df.to_csv(oname + '/corrected_df.csv', index=False)
    print(f"File 'corrected_df.csv' containing the corrections dataset used in the forager pipeline saved in '{oname}'")

    vocabpath = 'data/lexical_data/' + args.domain + '/vocab.csv'
    vocab = pd.read_csv(vocabpath, encoding="unicode-escape")
    vocab.to_csv(oname + '/forager_vocab.csv', index=False)
    print(f"File 'forager_vocab.csv' containing the full vocabulary used by forager saved in '{oname}'")
    # Run subroutine for getting strictly the similarity & frequency values 
    lexical_results, history_vars_list = run_lexical(data, args.domain, args.speech, args.dimension, corrected_df, args.alpha)
    lexical_results.to_csv(oname + '/lexical_results.csv', index=False) 
    print(f"File 'lexical_results.csv' containing similarity and frequency values of fluency list data saved in '{oname}'")

    ind_stats = indiv_desc_stats(lexical_results)
    ind_stats.to_csv(oname + '/individual_descriptive_stats.csv', index=False)
    print(f"File 'individual_descriptive_stats.csv' containing individual-level statistics saved in '{oname}'")
    
        
elif args.pipeline == 'switches':
    # Check if switches, then there is a switch method specified
    if args.switch == None:
        parser.error(f"Please specify a switch method (e.g. {switch_methods})")
    if args.switch not in switch_methods:
        parser.error(f"Please specify a proper switch method (e.g. {switch_methods})")
    # Run subroutine for getting strictly switch outputs 
    # Run subroutine for getting model outputs
    print("Checking Data ...")
    data, replacement_df, processed_df, corrected_df = retrieve_data(args.data, args.domain)
    replacement_df.to_csv(oname + '/evaluation_results.csv', index=False)
    print(f"File 'evaluation_results.csv' detailing the changes made to the dataset has been saved in '{oname}'")

    processed_df.to_csv(oname + '/processed_data.csv', index=False)
    print(f"File 'processed_data.csv' containing the processed dataset used in the forager pipeline saved in '{oname}'")

    corrected_df.to_csv(oname + '/corrected_df.csv', index=False)
    print(f"File 'corrected_df.csv' containing the corrections dataset used in the forager pipeline saved in '{oname}'")

    vocabpath = 'data/lexical_data/' + args.domain + '/vocab.csv'
    vocab = pd.read_csv(vocabpath, encoding="unicode-escape")
    vocab.to_csv(oname + '/forager_vocab.csv', index=False)
    print(f"File 'forager_vocab.csv' containing the full vocabulary used by forager saved in '{oname}'")

    print("Retrieving Lexical Data ...")
    lexical_results, history_vars_list = run_lexical(data, args.domain, args.speech, args.dimension, corrected_df, args.alpha)
    lexical_results.to_csv(oname + '/lexical_results.csv', index=False) 
    print(f"File 'lexical_results.csv' containing similarity and frequency values of fluency list data saved in '{oname}'")

    print("Obtaining Switch Designations ...")
    switch_results, switch_names_list, switch_vecs_list = run_switches(data,args.switch, args.domain, args.speech,args.dimension, history_vars_list, args.alpha)
    switch_results.to_csv(oname + '/switch_results.csv', index=False) 
    print(f"File 'switch_results.csv' containing designated switch methods and switch values of fluency list data saved in '{oname}'")

    ind_stats = indiv_desc_stats(lexical_results, switch_results)
    ind_stats.to_csv(oname + '/individual_descriptive_stats.csv', index=False)
    print(f"File 'individual_descriptive_stats.csv' containing individual-level statistics saved in '{oname}'")
    
    agg_stats = agg_desc_stats(switch_results)
    agg_stats.to_csv(oname + '/aggregate_descriptive_stats.csv', index=False)  
    print(f"File 'aggregate_descriptive_stats.csv' containing the overall group-level statistics saved in '{oname}'")

elif args.pipeline == 'models':
    # Check for model and switch parameters
    if args.model == None:
        parser.error(f"Please specify a forager model (e.g. {models})")
    if args.model not in models:
        parser.error(f"Please specify a proper forager model (e.g. {models})")
    if args.switch == None:
        parser.error(f"Please specify a switch method (e.g. {switch_methods})")
    if args.switch not in switch_methods:
        parser.error(f"Please specify a proper switch method (e.g. {switch_methods})")
    # Run subroutine for getting model outputs
    print("Checking Data ...")
    data, replacement_df, processed_df, corrected_df = retrieve_data(args.data, args.domain)
    # save to oname folder
    replacement_df.to_csv(oname + '/evaluation_results.csv', index=False)
    print(f"File 'evaluation_results.csv' detailing the changes made to the dataset has been saved in '{oname}'")

    processed_df.to_csv(oname + '/processed_data.csv', index=False)
    print(f"File 'processed_data.csv' containing the processed dataset used in the forager pipeline saved in '{oname}'")

    corrected_df.to_csv(oname + '/corrected_df.csv', index=False)
    print(f"File 'corrected_df.csv' containing the corrections dataset used in the forager pipeline saved in '{oname}'")

    vocabpath = 'data/lexical_data/' + args.domain + '/vocab.csv'
    vocab = pd.read_csv(vocabpath, encoding="unicode-escape")
    vocab.to_csv(oname + '/forager_vocab.csv', index=False)
    print(f"File 'forager_vocab.csv' containing the full vocabulary used by forager saved in '{oname}'")

    print("Retrieving Lexical Data ...")
    lexical_results, history_vars_list = run_lexical(data, args.domain, args.speech, args.dimension, corrected_df, args.alpha)
    lexical_results.to_csv(oname + '/lexical_results.csv', index=False) 
    print(f"File 'lexical_results.csv' containing similarity and frequency values of fluency list data saved in '{oname}'")

    print("Obtaining Switch Designations ...")
    switch_results, switch_names_list, switch_vecs_list = run_switches(data,args.switch, args.domain, args.speech, args.dimension, history_vars_list, args.alpha)
    switch_results.to_csv(oname + '/switch_results.csv', index=False) 
    print(f"File 'switch_results.csv' containing designated switch methods and switch values of fluency list data saved in '{oname}'")

    print("Running Forager Models...")
    model_results = run_model(data, args.model, args.switch, args.domain, args.speech, args.dimension, history_vars_list, switch_names_list, switch_vecs_list,args.alpha)
    model_results.to_csv(oname + '/model_results.csv', index=False) 
    print(f"File 'model_results.csv' containing model level NLL results of provided fluency data saved in '{oname}'")

    ind_stats = indiv_desc_stats(lexical_results, switch_results)
    ind_stats.to_csv(oname + '/individual_descriptive_stats.csv', index=False)
    print(f"File 'individual_descriptive_stats.csv' containing individual-level statistics saved in '{oname}'")
    
    agg_stats = agg_desc_stats(switch_results, model_results)
    agg_stats.to_csv(oname + '/aggregate_descriptive_stats.csv', index=False)  
    print(f"File 'aggregate_descriptive_stats.csv' containing the overall group-level statistics saved in '{oname}'")

else:
    parser.error("Please specify a proper pipeline option (e.g. \'evaluate_data\', \'lexical\', \'switches\',\'models\')")


 
#### SAMPLE RUN CODE ####
## regular models
# python3 run_foraging.py --data data/fluency_lists/samples/coch-foods-sample.txt --pipeline models --model all --switch all  --domain foods --speech speech2vec

## blended models
# python3 run_foraging.py --data data/fluency_lists/samples/coch-no-corrections.txt --pipeline models --model all --switch all  --domain foods --speech blended --dimension 50 --alpha 0.2
# python3 run_foraging.py --data data/fluency_lists/samples/coch-no-corrections.txt --pipeline models --model dynamic --switch simdrop  --domain foods --speech blended --dimension 50 --alpha 0.2
