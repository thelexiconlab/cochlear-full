import itertools
from typing import Tuple, List, Any

import numpy as np
import pandas as pd
import os
import random
import time

# import processed data and create dict[(str, int), list(str)] mapping participant and time point to a fluency list
processed_data = pd.read_csv('forager/output/cochlear_food_fulldata_forager_results/processed_data.csv')
fluency_lists = {}
for index, row in processed_data.iterrows():
    fluency_lists[row['SID']] = fluency_lists.get((row['SID']), []) + [row['entry']]

# import vocab and create dict[str, int] mapping words to index
vocab_df = pd.read_csv('forager/data/lexical_data/vocab.csv')
vocab = {}
for index, word in vocab_df.itertuples():
    vocab[word] = index

# import similarity matrices and frequency information
semantic_sim_matrix = pd.read_csv('forager/data/lexical_data/USE_semantic_matrix.csv', header=None)
phon_sim_matrix = pd.read_csv('forager/data/lexical_data/USE_phonological_matrix.csv', header=None)
frequencies = pd.read_csv('forager/data/lexical_data/USE_frequencies.csv', header=None).set_index(0)


class grid_search:
    def scrambled_results(fluency_list: list,
                          vocab: dict[str, int],
                          phon_sim_matrix: pd.DataFrame) -> list[float]:
        """
            Generates the distribution of mean similarity and frequency for a scrambled dataset for hypothesis testing

            Args:
                fluency_list (list): the fluency list produced by a participant
                vocab (dict[str, int]): a dictionary mapping vocabulary words to its index
                semantic_sim_matrix (pd.DataFrame): the semantic similarity matrix
                phon_sim_matrix (pd.DataFrame): the phonological similarity matrix
                frequencies (pd.DataFrame): a list of log frequencies by vocabulary word

            Returns:
                s_dist (list[float]): a list of mean semantic similarities across scrambled fluency lists
                p_dist (list[float]): a list of mean phonological similarities across scrambled fluency lists
                f_dist (list[float]): a list of mean pairwise mean frequency across scrambled fluency lists
                pair_f_diff_dist (list[float]): a list of mean squared pairwise frequency differences across scrambled
                    fluency lists

        """

        p_sum = 0
        for i in range(len(fluency_list) - 1):
            p_sum += phon_sim_matrix.loc[vocab[fluency_list[i]], vocab[fluency_list[i + 1]]]
            p_bar = p_sum / (len(fluency_list) - 1)
        return p_bar

    def check_scrambled(distribution_results: pd.DataFrame, indiv_desc_stats: pd.DataFrame) -> bool:
        """
        Tests whether the first entry in each distribution matches the computed mean in indiv_desc_stats

        Args:
            distribution_results: the distribution_results file from the scrambled_results function
            indiv_desc_stats: the individual_descriptive_stats file from the forager package
        Returns:
            (bool): whether the tests were passed for all participants and timepoints
        """
        errors = []
        for index, ID, timepoint, s_bar, s_dist, p_bar, p_dist, f_mean, f_dist, pair_f_diff_mean, pair_f_diff_dist \
                in distribution_results.itertuples():
            expected_s_bar = indiv_desc_stats.loc[f"('{ID}', {timepoint})", 'Semantic_Similarity_mean']
            if expected_s_bar != s_bar:
                errors.append(f"('{ID}', {timepoint})'s semantic similarity mean is {s_bar}, expected {expected_s_bar}")
            expected_p_bar = indiv_desc_stats.loc[f"('{ID}', {timepoint})", 'Phonological_Similarity_mean']
            if expected_p_bar != p_bar:
                errors.append(
                    f"('{ID}', {timepoint})'s phonological similarity mean is {p_bar}, expected {expected_p_bar}")
        for error in errors:
            print(error)
        return errors == []


def scramble_test(participant, timepoint):
    fluency_list = fluency_lists[(participant, timepoint)]
    s_dist, p_dist, f_dist, pair_f_diff_dist = grid_search.scrambled_results(fluency_list, vocab, semantic_sim_matrix,
                                                                             phon_sim_matrix, frequencies)
    results = {"Participant": [participant],
               "Timepoint": [timepoint],
               "Mean_semantic_similarity": [s_dist[0]],
               "Semantic_similarity_distribution": [','.join(map(str, s_dist[1:]))],
               "Mean_phonological_similarity": [p_dist[0]],
               "Phonological_similarity_distribution": [','.join(map(str, p_dist[1:]))],
               "Pairwise_frequency_mean": [f_dist[0]],
               "Pairwise_frequency_distribution": [','.join(map(str, f_dist[1:]))],
               "Squared_pairwise_frequency_difference_mean": [pair_f_diff_dist[0]],
               "Squared_pairwise_frequency_difference_distribution": [','.join(map(str, pair_f_diff_dist[1:]))]
               }
    distribution_results = pd.DataFrame(results)
    print(
        f'Test {"successful" if grid_search.check_scrambled(distribution_results, indiv_desc_stats) else "unsuccessful"}')
    distribution_results.to_csv(path_or_buf=outpath)


for participant, fluency_list in fluency_lists.items():
    fluency_lists[participant] = grid_search.scrambled_results(fluency_list, vocab, phon_sim_matrix)

df_phonological = pd.DataFrame(fluency_lists, index=[0]).melt()
df_phonological.to_csv(path_or_buf="forager/output/cochlear_food_fulldata_forager_results/phonological_means.csv")
