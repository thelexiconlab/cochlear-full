import pandas as pd

def matrix_to_pairs(matrix_file, output_file):
    """
    Convert a symmetrical matrix from a CSV file into a CSV file of word pairs.
    
    Parameters:
    matrix_file (str): Path to the CSV file containing the symmetrical matrix.
    output_file (str): Path to the CSV file where the pairs will be written.
    """
    # Read the symmetrical matrix
    df = pd.read_csv(matrix_file, index_col=0)
    
    # Ensure the matrix is symmetrical
    if not df.equals(df.T):
        raise ValueError("The matrix is not symmetrical.")
    
    # Get the list of words (index)
    words = df.index.tolist()
    
    # Generate all possible pairs of words
    pairs = []
    for i in range(len(words)):
        for j in range(i+1, len(words)):
            word1 = words[i]
            word2 = words[j]
            value = df.iloc[i, j]
            pairs.append([word1, word2, value])
    
    # Create a DataFrame for the pairs
    pairs_df = pd.DataFrame(pairs, columns=['Word1', 'Word2', 'Value'])
    
    # Write the pairs to a CSV file
    pairs_df.to_csv(output_file, index=False)
    print(f"edge list has been written to file")

# Example usage
matrix_file = r"C:\Users\c.hambric\OneDrive - Bowdoin College\Documents\cochlear-full\forager-cochlear\thresholdapp_matrix.csv"
output_file = r"C:\Users\c.hambric\OneDrive - Bowdoin College\Documents\cochlear-full\forager-cochlear\edge_list.csv"
matrix_to_pairs(matrix_file, output_file)