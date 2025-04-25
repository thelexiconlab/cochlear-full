import pandas as pd
import nltk
from itertools import product as iterprod
from tqdm import tqdm

tqdm.pandas()

# Define 
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

# Check for whole word stem overlap
def check_shared_stem(w1, w2, min_stem_len=4):
    w1 = w1.lower()
    w2 = w2.lower()
    if len(w1) >= min_stem_len and w1 in w2:
        return True
    if len(w2) >= min_stem_len and w2 in w1:
        return True
    for i in range(min_stem_len, min(len(w1), len(w2))):
        if w1[-i:] == w2[:i] or w2[-i:] == w1[:i]:
            return True
    return False

# Return the length of the longest continuous shared phoneme sequence
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

# Process file and save CSV
def process_phonemes(input_txt_file, output_csv_label):
    # Load data
    data = pd.read_csv(input_txt_file, sep="\t", header=0, usecols=[0, 1])
    data.columns = ['ID', 'Word']

    # Pull phonemes
    data['Phonemes'] = data['Word'].progress_apply(
        lambda w: ' '.join(phonology_funcs.wordbreak(w)[0]) if phonology_funcs.wordbreak(w) else None
    )

    # Initialize the new Stem column
    data["Shared_Cont_Phonemes"] = 0

     # Compare phoneme sequences only within the same ID
    for i in range(len(data) - 1):
        if data.at[i, "ID"] != data.at[i + 1, "ID"]:
            continue

        w1 = data.at[i, "Word"]
        w2 = data.at[i + 1, "Word"]
        p1 = data.at[i, "Phonemes"]
        p2 = data.at[i + 1, "Phonemes"]

        p1_list = p1.split() if isinstance(p1, str) else []
        p2_list = p2.split() if isinstance(p2, str) else []

        max_shared_len = longest_shared_phoneme_sequence(p1_list, p2_list, min_len=1)

        if check_shared_stem(w1, w2):
            max_shared_len = max(max_shared_len, 4)

        data.at[i, "Shared_Cont_Phonemes"] = max(data.at[i, "Shared_Cont_Phonemes"], max_shared_len)
        data.at[i + 1, "Shared_Cont_Phonemes"] = max(data.at[i + 1, "Shared_Cont_Phonemes"], max_shared_len)

    # Save results
    output_file = f"{output_csv_label}_phonemes_with_stems.csv"
    data.to_csv(output_file, index=False)
    print(f"✅ Saved to {output_file}")

# Example 
# Added input for csv label so it doesn't overwrite
#process_phonemes("C:/Users/c.hambric/OneDrive - Bowdoin College/Documents/cochlear-full/forager-cochlear/data/fluency_lists/cochlear_test.txt", "test")