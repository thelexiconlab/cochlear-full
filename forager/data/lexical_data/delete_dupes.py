import pandas as pd

vocab = pd.read_csv("forager/data/lexical_data/vocab.csv")
vocab_list = vocab['vocab'].tolist()

semantic, phonological, frequencies, embeddings = (
    pd.read_csv('forager/data/lexical_data/USE_semantic_matrix.csv', header=None),
    pd.read_csv('forager/data/lexical_data/USE_phonological_matrix.csv', header=None),
    pd.read_csv('forager/data/lexical_data/USE_frequencies.csv', header=None),
    pd.read_csv('forager/data/lexical_data/USE_embeddings.csv', header=None))


def find_first_dupes(vocab_list: list[str]):
    seen = {}
    first_indices = []

    for i, value in enumerate(vocab_list):
        if value in seen:
            if seen[value] == 1:  # Only add the first duplicate's index
                first_indices.append(vocab_list.index(value))
            seen[value] += 1
        else:
            seen[value] = 1

    return first_indices

def undupe(df: pd.DataFrame, first_indices: list[int], columns=False, rows=False) -> pd.DataFrame:
    # Drop rows with indices in first_indices
    df = df.drop(index=first_indices, errors='ignore') if rows else df
    # Drop columns with indices in first_indices
    df = df.drop(columns=first_indices, errors='ignore') if columns else df
    return df


first_indices = find_first_dupes(vocab_list)

vocab, semantic, phonological, frequencies, embeddings = (
    undupe(vocab, first_indices, rows=True),
    undupe(semantic, first_indices, columns=True, rows=True),
    undupe(phonological, first_indices, columns=True, rows=True),
    undupe(frequencies, first_indices, rows=True),
    undupe(embeddings, first_indices, columns=True)
)
vocab.to_csv(path_or_buf="vocab_unduped.csv")
semantic.to_csv(path_or_buf="forager/data/lexical_data/USE_semantic_matrix_unduped.csv")
phonological.to_csv(path_or_buf="forager/data/lexical_data/USE_phonological_matrix_unduped.csv")
frequencies.to_csv(path_or_buf="forager/data/lexical_data/USE_frequencies_unduped.csv")
embeddings.to_csv(path_or_buf="forager/data/lexical_data/USE_embeddings_unduped.csv")