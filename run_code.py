from utils import *

'''merge_results_hearing_status'''
#print(merge_results_hearing_status(df1,df2,"Subject"))
#merge_results_hearing_status(df1,df2).to_csv(path_or_buf=results_path)

'''extract_unique'''
#extract_unique("forager/data/fluency_lists/speech2vec_100.txt")

'''calc_semantic_sim'''
# print(calc_semantic_sim("pizza","popcorn"))
# print(calc_semantic_sim("popcorn", "hot dog"))
# print(calc_semantic_sim("hot dog","cheese"))
# print(calc_semantic_sim("cheese","coffee"))
# print(calc_semantic_sim("coffee","tea"))
# print(calc_semantic_sim("tea","soda"))
# print(calc_semantic_sim("soda","water"))
# print(calc_semantic_sim("water","asparagus"))
# print(calc_semantic_sim("asparagus","broccoli"))
# print(calc_semantic_sim("strawberries","blueberries"))'

'''vocab_check'''
#vocab_check("forager/data/fluency_lists/speech2vec_100.txt", "forager/data/fluency_lists/food_data.txt","forager/data/fluency_lists/food_data - Sheet1.csv", "entry")
#vocab_check("forager/data/fluency_lists/speech2vec_100.txt", "forager/data/fluency_lists/food_vocab_data.txt","forager/data/lexical_data/vocab.csv", "vocab")
#vocab_check("forager/data/fluency_lists/speech2vec_100.txt","forager/data/fluency_lists/food_data_unduped.txt", "forager/data/fluency_lists/food_data_unduped.csv", "unduplicated entries")

'''unduplicate'''
#unduplicate("forager/data/fluency_lists/food_data - Sheet1.csv", "entry")

'''cosine_similarity'''
#cosine_similarity("apple", "apricot", "forager/data/fluency_lists/speech2vec_100.txt")

'''calc_pairwise_sim'''

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