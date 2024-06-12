from utils import *

'''merge_results_hearing_status'''
#print(merge_results_hearing_status(df1,df2,"Subject"))
#merge_results_hearing_status(df1,df2).to_csv(path_or_buf=results_path)

'''extract_unique'''
#extract_unique("forager/data/fluency_lists/speech2vec_100.txt")

'''calc_semantic_sim'''
# print(calc_semantic_sim("tea","soda"))
# print(calc_semantic_sim("soda","water"))
# print(calc_semantic_sim("water","asparagus"))
# print(calc_semantic_sim("asparagus","broccoli"))
# print(calc_semantic_sim("strawberries","blueberries"))

'''vocab_check'''
#vocab_check("forager/data/fluency_lists/speech2vec_100.txt", "forager/data/fluency_lists/food_data.txt","forager/data/fluency_lists/food_data - Sheet1.csv", "entry")
#vocab_check("forager/data/fluency_lists/speech2vec_100.txt", "forager/data/fluency_lists/food_vocab_data.txt","forager/data/lexical_data/vocab.csv", "vocab")
#vocab_check("forager/data/fluency_lists/speech2vec_100.txt","forager/data/fluency_lists/food_data_unduped.txt", "forager/data/fluency_lists/food_data_unduped.csv", "unduplicated entries")

'''unduplicate'''
#unduplicate("forager/data/fluency_lists/food_data - Sheet1.csv", "entry")

'''cosine_similarity'''
#cosine_similarity("apple", "apricot", "forager/data/fluency_lists/speech2vec_100.txt")

'''calc_pairwise_sim'''
calc_pairwise_sim("forager/data/fluency_lists/test_data.csv")

'''extract_entries'''
#extract_entries("forager/data/fluency_lists/food_data - Sheet1.csv", "entry")

'''extract_subject'''
#extract_subject("forager/data/fluency_lists/food_data - Sheet1.csv", "id")

'''composite_frequency'''
#composite_freq("forager/output/cochlear_food_fulldata_forager_results/lexical_results.csv")

'''pairwise_phon'''
#pairiwse_phon("broccoli", "cauliflower", "forager/data/lexical_data/USE_phonological_matrix.csv")