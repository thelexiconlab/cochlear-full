#function to add column to lexical results by merging data

import pandas as pd
df1 = pd.read_csv("forager/output/cochlear_food_fulldata_forager_results 2/individual_descriptive_stats.csv")
#print(df1)
df2 = pd.read_csv("forager/data/fluency_lists/cochlear_status_data - Sheet1.csv")
#print(df2)


def merge_results_hearing_status(df1, df2):
    merged_df = pd.merge(df1,df2, on="Subject", how='left')
    return merged_df

print(merge_results_hearing_status(df1,df2))
merge_results_hearing_status(df1,df2).to_csv(path_or_buf='forager/output/merge_results.csv')
