import pandas as pd
import nltk
from itertools import product as iterprod
from tqdm import tqdm

tqdm.pandas()


#Pull Phonemes
class phonology_funcs:
    @staticmethod
    def wordbreak(s):
        try:
            arpabet = nltk.corpus.cmudict.dict()
        except LookupError:
            nltk.download('cmudict')
            arpabet = nltk.corpus.cmudict.dict()

        s = s.lower()
        if s in arpabet:
            return arpabet[s]
        
        middle = len(s) / 2
        partition = sorted(list(range(len(s))), key=lambda x: (x - middle) ** 2 - x)
        for i in partition:
            pre, suf = (s[:i], s[i:])
            if pre in arpabet and phonology_funcs.wordbreak(suf) is not None:
                return [x + y for x, y in iterprod(arpabet[pre], phonology_funcs.wordbreak(suf))]
        
        return None

# Remove stress markers from phonemes
def remove_stress(phoneme_list):
    return [p[:-1] if p[-1].isdigit() else p for p in phoneme_list]

# Find length of longest shared phoneme sequence
def longest_shared_phoneme_sequence(p1, p2, min_len=1):
    if not p1 or not p2:
        return 0
    max_len = 0
    for i in range(len(p1)):
        for j in range(len(p2)):
            match_len = 0
            while (
                i + match_len < len(p1)
                and j + match_len < len(p2)
                and p1[i + match_len] == p2[j + match_len]
            ):
                match_len += 1
                if match_len >= min_len:
                    max_len = max(max_len, match_len)
    return max_len

# Processing

def process_words_to_phoneme_pairs(input_txt_file, output_csv_label):
    data = pd.read_csv(input_txt_file, sep="\t")  # Assuming tab-separated
    assert 'ID' in data.columns and 'Word' in data.columns, "Input file must have 'ID' and 'Word' columns."

    # Order words within each subject
    data['Order'] = data.groupby('ID').cumcount()

    # Create adjacent word pairs
    data_sorted = data.sort_values(by=['ID', 'Order'])
    data_sorted['Next_Word'] = data_sorted.groupby('ID')['Word'].shift(-1)

    # Drop rows where there is no next word (last)
    pairs = data_sorted.dropna(subset=['Next_Word'])
    pairs = pairs.rename(columns={'Word': 'Word1', 'Next_Word': 'Word2'}) #rename
    pairs = pairs[['ID', 'Word1', 'Word2']]  

    # Pull phoneme pairs
    pairs['Phonemes1'] = pairs['Word1'].progress_apply(
        lambda w: phonology_funcs.wordbreak(w))
    pairs['Phonemes2'] = pairs['Word2'].progress_apply(
        lambda w: phonology_funcs.wordbreak(w))

    # Compute shared phonemes
    def compute_shared_phonemes(row):
        p1 = row['Phonemes1'][0] if isinstance(row['Phonemes1'], list) else []
        p2 = row['Phonemes2'][0] if isinstance(row['Phonemes2'], list) else []

        p1 = remove_stress(p1)
        p2 = remove_stress(p2)

        return longest_shared_phoneme_sequence(p1, p2, min_len=1)

    pairs['Shared_Cont_Phonemes'] = pairs.progress_apply(compute_shared_phonemes, axis=1)

    # Save results
    output_file = f"{output_csv_label}_phoneme_pairs.csv"
    pairs.to_csv(output_file, index=False)
    print(f"✅ Saved to {output_file}")

# Example 
process_words_to_phoneme_pairs("C:/Users/c.hambric/OneDrive - Bowdoin College/Documents/cochlear-full/forager-cochlear/data/fluency_lists/cochlear-animals-raw.txt", "animals")
process_words_to_phoneme_pairs("C:/Users/c.hambric/OneDrive - Bowdoin College/Documents/cochlear-full/forager-cochlear/data/fluency_lists/cochlear-foods-raw.txt", "foods")