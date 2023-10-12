"""
Run the main section at the bottom of this file to calculate vectors for prompts and for documents.

To select a model to create embeddings, change the `model_name` parameter of the `self.embeddings` attribute in line 70
The default model used is `BAAI/bge-large-en-v1.5`. Other models can be found at
https://huggingface.co/spaces/mteb/leaderboard

Vectors for documents will be stored in Data/ExtractedData/ModelTraining/document_vectors.pkl. Loading this file yields
a dictionary in the format { 'prompt_id': { 'filename' }: vector  }. Vectors are 768 dimensions (or n * 768 if the text
is being partitioned).

Vectors for prompts will be stored in Data/ExtractedData/ModelTraining/prompt_vectors.pkl. Loading this file yields
a dictionary in the format { 'prompt_id': vector  }. Again, vectors are 768 dimensions (or n * 768 if the text
is being partitioned).

To partition vectors into n segments, process a vector for each partition, and then concatenate the resulting vectors
into one, edit line 156 by adding the parameters: partition=True, num_partitions=n (with n = the number of partitions
desired) to the end of the function call.

"""

import pickle
import urllib3
import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
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


class HuggingFaceVectorController:

    def __init__(self):
        """
        Class of helpers to get document vectors from Hugging Face
        """

        self.logs = Logger()
        self.embeddings = HuggingFaceEmbeddings(# model_name="sentence-transformers/all-mpnet-base-v2",
                                                model_name="BAAI/bge-large-en-v1.5",
                                                # model_name="BAAI/bge-small-en-v1.5
                                                cache_folder=f"{ROOT_PATH}/HuggingFace/BigModels",
                                                encode_kwargs={'normalize_embeddings': False})
        self.successful_files = 0
        self.failed_files = 0
        self.failed_filenames = []
        self.filename_to_vector = dict()
        self.prompt_to_vector = dict()
        self.prompts = dict()

    def add_prompt(self, text, prompt):
        """Function to add prompt to a text"""
        prompt_text = self.prompts[prompt]
        return f"{prompt_text} \n\nANSWER: {text}"

    def get_document_vector(self, text, filename, total_files=1, counter=0, prompt=None, partition=False,
                            num_partitions=3):
        """
        Function to get a vector for a document
        :param text: Text from the file being processed
        :param filename: The name of the file being processed
        :param total_files: The total number of files being processed
        :param counter: The total number of files that have been processed so far
        :param prompt: The prompt the document is based off. Only use to add prompt text to document text
        :param partition: Set to true to partition documents into n strings, get a vector for each and then concatenate them
        :param num_partitions: Number of partitions to divide text into
        :return: The document vector
        """
        # Set to prompt id to add prompt text to document text
        if prompt is not None:
            text = self.add_prompt(text, prompt)

        # Set partition to True to split text into multiple vectors
        if partition:
            partitions = split_text(text, num_partitions)
            partition_vectors = []
            # Iterate through partitions
            for partition in partitions:
                partition_vector = self.embeddings.embed_query(partition)
                partition_vectors.append(partition_vector)
            # Concatenate vectors
            document_vector = concatenate_vectors(partition_vectors)

        # Get vector for document as a whole
        else:
            document_vector = self.embeddings.embed_query(text)

        # Return vector
        self.logs.info(f"Successfully retrieved vector for {filename}\tSTATS:  - Successful Documents: {self.successful_files}  - Failed Documents: {self.failed_files}  - Total Processed: {counter}/{total_files}")
        self.successful_files += 1
        return document_vector

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
                    document_vector = self.get_document_vector(text, filename, prompt_lengths, counter)  ### ADD PROMPT/PARTTION PARAMETERS TO THE END OF THIS FUNCTION CALL ###
                    if document_vector is not None:
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

            self.logs.info(f"Getting vectors for prompt: {prompt}")

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
    vector_controller = HuggingFaceVectorController()
    vector_controller.get_prompt_vectors()
    vector_controller.get_all_document_vectors()
