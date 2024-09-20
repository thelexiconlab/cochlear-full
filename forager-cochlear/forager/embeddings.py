import pandas as pd 
import os

class embeddings:
    '''
        Description: 
            This class contains functions that create the embeddings.csv file from a list of words
            using the master speech2vec and word2vec model lists.
        
        Args:
            words: contains the vocab for which embeddings need to be extracted
            
        Functions: 
            (1) __init__: creates USE_embeddings.csv file
            (2) test_embeddings: tests the similarity of two words using cosine similarity from scipy.
    
    '''
    import pandas as pd

class Embeddings:
    def __init__(self, domain, model_for_embeddings, dimension):
        domain_path = '../data/lexical_data/' + domain + '/'
        path_for_lexical_data = '../data/lexical_data/'
        
        # Open the .txt file containing the embeddings
        if model_for_embeddings == "speech2vec":
            file = open(path_for_lexical_data + 'embeddings/Speech2Vec/speech2vec_' + dimension + '.txt', "r")
        else:
            file = open(path_for_lexical_data + 'embeddings/Word2Vec/word2vec_' + dimension + '.txt', "r")
        
        words = pd.read_csv(domain_path + 'vocab.csv')['word'].values.tolist()
        

        embeddings_dict = {}
        for line in file:
            parts = line.strip().split()
            word = parts[0]
            # If the word is in the list, store the embedding
            if word in words:
                embedding = list(map(float, parts[1:]))
                embeddings_dict[word] = embedding
        
        # Ensure the embeddings are added in the same order as the word list
        ordered_embeddings = [embeddings_dict[word] for word in words if word in embeddings_dict]
        
        # Convert the list of embeddings into a DataFrame
        embeddings_df = pd.DataFrame(ordered_embeddings, index=words)
        

        # Transpose the DataFrame so that words are columns and embeddings are rows
        embeddings_df = embeddings_df.transpose()
    
        # Save the DataFrame as a CSV file
        #self.path = domain_path + model_for_embeddings + '/' + dimension + '/' + 'embeddings.csv'
        # create directory if it doesn't exist
        # Construct the output path
        # Define the path components
        oname = os.path.join(domain_path, model_for_embeddings, dimension)
        file_path = os.path.join(oname, 'embeddings.csv')

        # Create the directory if it does not exist
        os.makedirs(oname, exist_ok=True)

        # Save the DataFrame to a CSV file
        embeddings_df.to_csv(file_path, index=False)

# SAMPLE RUN CODE
embeddings_instance = Embeddings('animals', 'word2vec', '300')
