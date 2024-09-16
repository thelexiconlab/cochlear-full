import gensim.downloader as api
import csv
import pandas as pd

# Load the Word2Vec model
model = api.load('word2vec-google-news-300')
vocab = model.key_to_index  # This is a dictionary of words in the vocabulary

#Note this was when i tested for food but just didn't change the variable names
animals_sample = ['apple','banana','lettuce','carrot','bread','cake','soup','pasta','taco','flour','oil']
# Find similar words
animal_words = set()
for animal in animals_sample:
    if animal in model.key_to_index:
        similar_words = model.most_similar(animal, topn=100)
        for word, _ in similar_words:
            if word in vocab:
                animal_words.add(word)

# Define the path for the CSV file where results will be saved
csv_file_path = 'final_animal_word_check_s2v.csv'

# Write the results to a CSV file
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Animal Word'])
    for word in animal_words:
        writer.writerow([word])

print(f"Potential animal words in the vocabulary have been written to {csv_file_path}")