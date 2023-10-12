"""
Run the main section at the bottom of this file to calculate vectors for prompts and for documents.

Vectors for documents will be stored in Data/ExtractedData/ModelTraining/document_vectors.pkl. Loading this file yields
a dictionary in the format { 'prompt_id': { 'filename' }: vector  }. Vectors are 1000 dimensions.
"""

import pickle
import urllib3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from Logs.log_config import Logger
from root_path import ROOT_PATH
import os
import statistics
import string

# Ignore warnings
urllib3.disable_warnings()


class TFIDFVectorController:

    def __init__(self):
        """
        Class of helpers to get document vectors using TF-IDF
        """

        self.logs = Logger()
        self.vectorizer = TfidfVectorizer(use_idf=True, stop_words='english')
        self.successful_files = 0
        self.failed_files = 0
        self.failed_filenames = []
        self.filename_to_vector = dict()
        self.corpora = self.define_corpora()
        self.tfidf_vectors = self.score_corpora()

    def define_corpora(self):
        """Function to define a corpus for each prompt"""
        corpora = dict()
        prompts = os.listdir(f"{ROOT_PATH}/Data/ExtractedData/ModelTraining/FullData")
        # Iterate through prompts
        for prompt in prompts:
            corpora[prompt] = []
            counter = 1
            documents = os.listdir(f"{ROOT_PATH}/Data/ExtractedData/ModelTraining/FullData/{prompt}")
            # Iterate through documents in each prompt and define corpus for prompt
            for document in documents:
                self.logs.info(f"Defining corpus for prompt: {prompt}; - Progress: {counter}/{len(documents)}")
                FILE_PATH = f"{ROOT_PATH}/Data/ExtractedData/ModelTraining/FullData/{prompt}/{document}"
                with open(FILE_PATH, 'r', encoding='utf-8') as file:
                    text = file.read()
                file.close()
                corpora[prompt].append(text)
                counter += 1

        return corpora

    def score_corpora(self):
        """Function to calculate TFIDF score for each word in a corpus"""
        word_scores = dict()
        # Iterate through prompts
        for prompt in self.corpora.keys():
            prompt_dict = {}
            self.logs.info(f"Vectorizing corpus for prompt: {prompt}")
            # Vectorize documents
            documents = self.corpora[prompt]
            vectors = self.vectorizer.fit_transform(documents).todense()
            words = self.vectorizer.get_feature_names_out()
            df = pd.DataFrame(vectors, columns=words)
            # Find score for each word
            for word in words:
                score = set(df[word].values)
                score.remove(0)
                try:
                    prompt_dict[word] = statistics.mean(score)
                except statistics.StatisticsError:
                    prompt_dict[word] = 0

            word_scores[prompt] = prompt_dict
        return word_scores

    def get_document_vector(self, text, filename, prompt, total_files, counter):
        """Function to calculate a document vector using wording scores"""
        vector = []
        # Replace each word with wording score
        for word in text.split(" "):
            try:
                score = self.tfidf_vectors[prompt][word]
                vector.append(score)
            except KeyError:
                vector.append(0)

        # Fill rest of document with zeros until reached max length
        while len(vector) < 1000:
            vector.append(0)

        # Only use first 1000 words
        if len(vector) > 1000:
            vector = vector[0:1000]

        self.logs.info(f"Successfully retrieved vector for {filename}\tSTATS: - Total Processed: {counter}/{total_files}")
        self.successful_files += 1
        return vector

    def save_document_vectors(self):
        """Function to save document vectors to a pickle file"""
        self.logs.info("Saving document vectors")
        DATA_PATH = f"{ROOT_PATH}/Data/ExtractedData/ModelTraining"
        with open(f"{DATA_PATH}/document_vectors.pkl", "wb") as pklfile:
            pickle.dump(self.filename_to_vector, pklfile)
            pklfile.close()

    def get_all_document_vectors(self):
        """Function to find vectors for all documents"""
        DATA_PATH = f"{ROOT_PATH}/Data/ExtractedData/ModelTraining/FullData"
        # Iterate through prompts
        prompts = os.listdir(DATA_PATH)
        prompt_lengths = sum([len(os.listdir(f"{DATA_PATH}/{prompt}")) for prompt in prompts])
        counter = 1
        for prompt in prompts:
            self.logs.info(f"Getting vectors for prompt: {prompt}")

            # Get document vectors
            self.filename_to_vector[prompt] = dict()
            # Iterate through files
            for filename in os.listdir(f"{DATA_PATH}/{prompt}"):
                FILE_PATH = f"{DATA_PATH}/{prompt}/{filename}"
                try:
                    with open(FILE_PATH, 'r', encoding='utf-8') as file:
                        text = file.read()
                        text = text.translate(str.maketrans('', '', string.punctuation)) # Remove punctuation
                    file.close()
                except UnicodeDecodeError as e:
                    self.logs.error(f"Failed opening {filename}; Details: {str(e)}")
                    self.failed_files += 1
                    self.failed_filenames.append(filename)
                else:
                    document_vector = self.get_document_vector(text, filename, prompt, prompt_lengths, counter)
                    self.filename_to_vector[prompt][filename] = document_vector
                counter += 1

        # Save all filenames that failed
        with open("vector_failed_files.txt", "w") as failed_file:
            failed_file.write(str(self.failed_filenames))
        failed_file.close()

        # Save vectors
        self.save_document_vectors()


# Run Script
if __name__ == '__main__':
    vector_controller = TFIDFVectorController()
    vector_controller.get_all_document_vectors()
