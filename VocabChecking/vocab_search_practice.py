import pandas as pd
import difflib



def vocab_search(test_word, vocab_list):
    if test_word in vocab_list:
        print("the word",test_word.upper(),"is in the list.")
    else:
        print("the word",test_word.upper(),"is NOT in the list.")



practice_lexicon = ["cat","frog","salamanders","snake","turtle","horse"]
practice_list = ["cat","frog","salamanders","snake","turtle","horse","axolotl","hamster"]
speech2vec=pd.read_csv(r"C:\Users\c.hambric\OneDrive - Bowdoin College\Documents\cochlear-full\forager\data\lexical_data\foods\speech2vec_vocab.csv")
speech2vec=speech2vec.rename(columns={"0":"Words"})
speech2vec_vocab=list(speech2vec.Words)

test_word1="frog"
test_word2="axolotl"

animals_AZ_data = pd.read_csv(
    r"C:\Users\c.hambric\OneDrive - Bowdoin College\Documents\cochlear-full\forager\Animals_AZ.csv",
    encoding='latin1')
animals_AZ_entries=list(animals_AZ_data.animals)



animals_forager_data=pd.read_csv(r"C:\Users\c.hambric\OneDrive - Bowdoin College\Documents\cochlear-full\forager\animals_forager_vocab.csv")
animals_forager_entries=list(animals_forager_data.vocab)

animals_comb = pd.read_csv(r"C:\Users\c.hambric\OneDrive - Bowdoin College\Documents\cochlear-full\forager_AZ_comb.csv", encoding='latin1')
animals_comb_entries=list(animals_comb.vocab)

all_foods_check = pd.read_csv(r"C:\Users\c.hambric\OneDrive - Bowdoin College\Documents\cochlear-full\all_food_s2v.csv", encoding='latin1')
all_foods_check_entries=list(all_foods_check.words)


# Sample practice list
practice_list = ["cat","frog","salamanders","snake","turtle","horse","axolotl","hamster"]

# Read and clean speech2vec data
speech2vec = pd.read_csv(r"C:\Users\c.hambric\OneDrive - Bowdoin College\Documents\cochlear-full\forager\data\lexical_data\foods\speech2vec_vocab.csv")
speech2vec = speech2vec.rename(columns={"0": "Words"})
speech2vec_vocab = speech2vec.Words.dropna().astype(str).tolist()

# Function to find similar words
def find_similar_words(word, vocab_list, n=3, cutoff=0.65):
    matches = difflib.get_close_matches(word, vocab_list, n=n, cutoff=cutoff)
    return matches

# Function to search for words in the vocabulary
def list_search(fluency_list, vocab_list):
    results = []  # Initialize an empty list to store results
    for word in fluency_list:
        similar_words = find_similar_words(word, vocab_list)
        found = len(similar_words) > 0
        original_found = word in similar_words
        results.append((found, original_found, ', '.join(similar_words) if similar_words else "No close match found"))
    return results

# Perform the search
search_results = list_search(all_foods_check_entries,speech2vec_vocab)

# Convert results to a DataFrame
results_df = pd.DataFrame({
    'Word': all_foods_check_entries,
    'Found': [result[0] for result in search_results],
    'Original Found': [result[1] for result in search_results],
    'Close Matches': [result[2] for result in search_results]
})
# Save the DataFrame to a CSV file
results_df.to_csv(r"C:\Users\c.hambric\OneDrive - Bowdoin College\Documents\cochlear-full\food_check_results.csv", index=False)

print("Results saved to search_results.csv")