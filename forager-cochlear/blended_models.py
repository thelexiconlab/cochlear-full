import pandas as pd 
import os
import numpy as np
from forager.cues import create_semantic_matrix

class blended:
    def __init__(self):
        domains = ['animals', 'foods'] 
        dimensions = ['50']
        alphas = np.array([1.0])

        for d in domains:
            for dim in dimensions:
                for a in alphas:
                    a = np.round(a,1)
                    print(f"for {d}-{dim} and alpha={a}")
                    self.create_blend(d, dim, a)
    
    def create_blend(self,domain, dimension, alpha_speech):
        domain_path = 'data/lexical_data/' + domain + '/'
    
        words = pd.read_csv(domain_path + 'vocab.csv')['word'].values.tolist()

        ## read in speech2vec and word2vec embeddings for specific domain and dimension

        word2vec = np.array(pd.read_csv(domain_path + 'word2vec/' + dimension + '/' + '1.0'+ '/embeddings.csv'))
        speech2vec = np.array(pd.read_csv(domain_path + 'speech2vec/' + dimension + '/' + '1.0'+ '/embeddings.csv'))
    
        # Compute the weighted embeddings
        weighted_speech2vec = alpha_speech * speech2vec
        weighted_word2vec = (1 - alpha_speech) * word2vec
        
        # Concatenate the weighted embeddings
        concatenated_embeddings = np.concatenate((weighted_speech2vec, weighted_word2vec), axis=0)
        

        embeddings_df = pd.DataFrame(concatenated_embeddings, columns=words)

        # Define the path components
        oname = os.path.join(domain_path, 'blended', dimension, str(alpha_speech))
        file_path = os.path.join(oname, 'embeddings.csv')

        # Create the directory if it does not exist
        os.makedirs(oname, exist_ok=True)

        # Save the DataFrame to a CSV file
        embeddings_df.to_csv(file_path, index=False)
        print("\nCreated and saved embeddings")

        # create semantic matrix
        # get semantic matrix 
        create_semantic_matrix(file_path)
        print("\nCreated and saved semantic similarity matrix ")
    

# SAMPLE RUN CODE
blended()
