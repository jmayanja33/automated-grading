"""
Run the main section at the bottom of this file to calculate vectors for prompts and for documents.

Vectors for documents will be stored in Data/ExtractedData/ModelTraining/document_vectors.pkl. Loading this file yields
a dictionary in the format { 'prompt_id': { 'filename' }: vector  }. Vectors are 350 dimensions.

To calculate vectors a Doc2Vec model is created when this class is initialized, using all the documents as a corpus.
The model is trained on the corpus for 200 epochs, and produces 350 dimensional vectors.

To partition vectors into n segments, process a vector for each partition, and then concatenate the resulting vectors
into one, edit line 223 by adding the parameters: partition=True, num_partitions=n (with n = the number of partitions
desired) to the end of the function call.
"""

import pickle
import urllib3
import string
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from Logs.log_config import Logger
from root_path import ROOT_PATH
import os

# Ignore warnings
urllib3.disable_warnings()


def split_text(text, num_sections=3):
    """Function to split a text into three parts for processing"""
    clean_text = " ".join(text.split())
    split_length = int(len(clean_text)/num_sections)
    strings = []
    partition_start = 0
    ready_to_partition = False

    # Iterate through characters in text
    for i in range(len(clean_text)):
        # Find where split needs to occur
        if (i > 0 and i % split_length == 0):
            ready_to_partition = True
        # Split at next space
        if (clean_text[i] == ' ' and ready_to_partition) or ((i == len(clean_text) - 1) and (len(strings) < num_sections)):
            partition = clean_text[partition_start: i]
            partition_start = i
            strings.append(partition)
            ready_to_partition = False
    # Return list of partitions
    return strings


def concatenate_vectors(vector_list):
    """Function to concatenate a list of vectors into one"""
    return [v for vector in vector_list for v in vector]


class Doc2VecController:

    def __init__(self):
        """
        Class of helpers to get document vectors using Doc2Vec
        """

        self.logs = Logger()
        self.successful_files = 0
        self.failed_files = 0
        self.failed_filenames = []
        self.filename_to_vector = dict()
        self.prompt_to_vector = dict()
        self.prompts = dict()
        self.corpus = self.define_corpus()
        self.model = self.create_doc2vec_model()

    def define_corpus(self):
        """Function to define a corpus for each prompt"""
        corpus = []
        prompts = os.listdir(f"{ROOT_PATH}/Data/ExtractedData/ModelTraining/FullData")
        # Iterate through prompts
        for prompt in prompts:
            # corpora[prompt] = []

            # Add prompt to corpus
            data = pd.read_csv(f"{ROOT_PATH}/Data/ProvidedData/prompts_train.csv")
            prompt_lengths = len(data)
            counter = 1
            for i in range(prompt_lengths):
                text = data["prompt_text"][i]
                text = text.translate(str.maketrans('', '', string.punctuation))
                text = " ".join(text.split())

                # Tag text (Turn document into list of words and tags)
                words = text.lower().split(" ")
                tags = [str(counter)]
                tagged_text = TaggedDocument(words=words, tags=tags)
                corpus.append(tagged_text)

            counter = 1
            documents = os.listdir(f"{ROOT_PATH}/Data/ExtractedData/ModelTraining/FullData/{prompt}")
            # Iterate through documents in each prompt and define corpus for prompt
            for document in documents:
                self.logs.info(f"Defining corpus for prompt: {prompt}; - Progress: {counter}/{len(documents)}")
                # Load documnet
                FILE_PATH = f"{ROOT_PATH}/Data/ExtractedData/ModelTraining/FullData/{prompt}/{document}"
                with open(FILE_PATH, 'r', encoding='utf-8') as file:
                    text = file.read()
                    # Format document
                    text = text.translate(str.maketrans('', '', string.punctuation))
                    text = " ".join(text.split())

                    # Tag text (Turn document into list of words and tags)
                    words = text.lower().split(" ")
                    tags = [str(counter)]
                    tagged_text = TaggedDocument(words=words, tags=tags)
                file.close()
                # Saved tagged text object
                corpus.append(tagged_text)
                counter += 1

        return corpus

    def create_doc2vec_model(self):
        """Function to calculate TFIDF score for each word in a corpus"""
        # prompt_models = dict()
        # # Iterate through prompts
        # for prompt in self.corpora.keys():
        #     corpus = self.corpora[prompt]
        #     self.logs.info(f"Creating Doc2Vec model for prompt: {prompt}")
        #     # Create model
        #     model = Doc2Vec(documents=corpus, vector_size=350, alpha=0.025, min_alpha=0.00025, seed=33,
        #                     min_count=1, dm=1)
        #     # model.build_vocab(tagged_text)
        #     # Train model
        #     self.logs.info(f"Training Doc2Vec model for prompt: {prompt}")
        #     model.train(corpus, total_examples=model.corpus_count, epochs=100)
        #     # for i in range(100):
        #     #     self.logs.info(f"Training Doc2Vec model for prompt: {prompt}; PROGRESS: - Epoch {i}/100")
        #     #     model.train(corpus, total_examples=model.corpus_count, epochs=10)
        #     #     model.alpha -= 0.0002
        #     #     model.min_alpha = model.alpha
        #     # Save model
        #     prompt_models[prompt] = model
        self.logs.info(f"TrainingDoc2Vec model")
        # Create and train model
        model = Doc2Vec(documents=self.corpus, vector_size=350, alpha=0.025, min_alpha=0.00025, seed=33,
                        dm=1, epochs=200)
        # model.build_vocab(tagged_text)
        # Train model
        #self.logs.info(f"Training Doc2Vec model")
        # model.train(self.corpus, total_examples=model.corpus_count, epochs=20)

        return model

    def add_prompt(self, text, prompt):
        """Function to add prompt to a text"""
        prompt_text = self.prompts[prompt]
        return f"{prompt_text} \n\nANSWER: {text}"

    def get_document_vector(self, text, filename, total_files, counter, prompt=None, partition=False,
                            num_partitions=3):
        """Function to calculate a document vector using wording scores"""

        # Load model and calculate vector
        # model = self.prompt_models[prompt]
        # Set to prompt id to add prompt text to document text
        if prompt is not None:
            text = self.add_prompt(text, prompt)

        # Set partition to True to split text into multiple vectors
        if partition:
            partitions = split_text(text, num_partitions)
            partition_vectors = []
            # Iterate through partitions
            for partition in partitions:
                partition = partition.translate(str.maketrans('', '', string.punctuation))
                partition = " ".join(partition.split())
                partition = partition.lower().split(" ")
                partition_vector = self.model.infer_vector(partition)
                partition_vectors.append(partition_vector)
            # Concatenate vectors
            vector = concatenate_vectors(partition_vectors)

        else:
            document = text.translate(str.maketrans('', '', string.punctuation))
            document = " ".join(document.split())
            document = document.lower().split(" ")
            vector = self.model.infer_vector(document)

        self.logs.info(f"Successfully calculated vector for {filename}\tSTATS: - Total Processed: {counter}/{total_files}")
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
                    file.close()
                except UnicodeDecodeError as e:
                    self.logs.error(f"Failed opening {filename}; Details: {str(e)}")
                    self.failed_files += 1
                    self.failed_filenames.append(filename)
                else:
                    document_vector = self.get_document_vector(text, filename, prompt_lengths, counter)
                    self.filename_to_vector[prompt][filename] = document_vector
                counter += 1

        # Save all filenames that failed
        with open("vector_failed_files.txt", "w") as failed_file:
            failed_file.write(str(self.failed_filenames))
        failed_file.close()

        # Save vectors
        self.save_document_vectors()

    def get_prompt_vectors(self):
        """Function to find vectors for all documents"""
        data = pd.read_csv(f"{ROOT_PATH}/Data/ProvidedData/prompts_train.csv")

        # Iterate through prompts
        prompt_lengths = len(data)
        counter = 1
        for i in range(prompt_lengths):
            prompt = data["prompt_id"][i]
            prompt_question = data["prompt_question"][i]
            prompt_title = data["prompt_title"][i]
            prompt_text = data["prompt_text"][i]

            text = f"""TITLE: {prompt_title} \n\nQUESTION: {prompt_question} \n\nDIRECTIONS: {prompt_text}"""

            # Save formatted prompt
            if prompt not in self.prompts.keys():
                self.prompts[prompt] = text

            self.logs.info(f"Getting vector for prompt: {prompt}")

            # Get prompt vectors
            self.prompt_to_vector[prompt] = dict()
            document_vector = self.get_document_vector(text, prompt, prompt_lengths, counter)
            if document_vector is not None:
                self.prompt_to_vector[prompt] = document_vector
                counter += 1

        # Save vectors
        self.save_prompt_vectors()

    def save_prompt_vectors(self):
        """Function to save document vectors to a pickle file"""
        self.logs.info("Saving prompt vectors")
        DATA_PATH = f"{ROOT_PATH}/Data/ExtractedData/ModelTraining"
        with open(f"{DATA_PATH}/prompt_vectors.pkl", "wb") as pklfile:
            pickle.dump(self.prompt_to_vector, pklfile)
            pklfile.close()


# Run Script
if __name__ == '__main__':
    vector_controller = Doc2VecController()
    vector_controller.get_prompt_vectors()
    vector_controller.get_all_document_vectors()
