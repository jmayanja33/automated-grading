"""
Run the main section at the bottom of this file to train and evaluate models for content and wording scores.

Inside the folder for each model in Models/NeuralNetwork/Tensorflow/Content or Wording Score there will be 3 files.
ScoreStats.txt has info on RMSE and R-squared.  ScoreTrainingLoss.png contains a plot of the training losses to gauge
over fitting. ScoreModel.keras is the model that can be loaded at any time.

Inside Models/NeuralNetwork/Tensorflow there is a file MeanColumwiseRMSESummary.txt which contains info on the final
competition metric which is MCRMSE.

The current configuration contains use the best hyperparameters found using the auto tuner. These can be changed between
lines 139-168 in the train_model function. The commented out code from lines 170-198 is the automated hyperparameter
tuner. If desired to use, comment out line 139-168, and then uncomment 170-198. Update the input shape arguments in
lines 56 and 193. To update different values of hidden layers, neurons per layer, and activation functions,
update lines 58-61.

The current configuration also uses a document vector as is. To use the document vector and append the cosine similarity
to the corresponding prompt as an extra component uncomment lines 125-127 and comment out line 122. To concatenate
the prompt and document vectors (in that order) comment out lines 122 and 125-127 and thn uncomment lines 130-131.

"""


import pickle
import pandas as pd
import numpy as np
import keras_tuner as kt
import matplotlib.pyplot as plt
import tensorflow as tf
from numpy.linalg import norm
from tensorflow import keras
from keras.callbacks import EarlyStopping
from Logs.log_config import Logger
from Models.model_evaluator import *
from root_path import ROOT_PATH
import os
import shutil

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set seed
keras.utils.set_random_seed(33)


def cosine_similarity(vector_1, vector_2):
    """Function to calculate cosine similarity"""
    return np.dot(vector_1, vector_2) / (norm(vector_1) * norm(vector_2))


def concatenate_vectors(vector_list):
    """Function to concatenate a list of vectors into one"""
    return [v for vector in vector_list for v in vector]


def normalize_vector(vector):
    """Function to normalize a vector"""
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return [i/norm for i in vector]


def build_tuner_model(hp):
    """Function to build a models to tune hyperparameters with 2-5 hidden layers"""
    # Create model
    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(769,)))
    # Set num hidden layers, num neurons, and activation functions to be tested
    for i in range(hp.Int('layers', 9, 10)):
        hp_layer = hp.Int(f'hiddenlayer{i}', min_value=700, max_value=760, step=10)
        hp_activation = hp.Choice(f'activation{i}', values=['relu', 'leaky_relu'])
        model.add(keras.layers.Dense(units=hp_layer, activation=hp_activation))
    # Add output layer and compile model
    model.add(keras.layers.Dense(1, name="output"))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
    return model


class TensorflowTrainer:
    """Class to train a tensorflow model"""

    def __init__(self, dependent_variable):
        self.logs = Logger()
        self.dependent_variable = dependent_variable
        self.best_params = None
        self.train_rmse = None
        self.validation_rmse = None
        self.test_rmse = None

        # Delete old tuning data
        self.remove_saved_tuning_data()

        self.logs.info("Loading Document Vectors")
        with open(f"{ROOT_PATH}/Data/ExtractedData/ModelTraining/document_vectors.pkl", "rb") as pklfile:
            self.document_vectors = pickle.load(pklfile)
            pklfile.close()

        self.logs.info("Loading Prompt Vectors")
        with open(f"{ROOT_PATH}/Data/ExtractedData/ModelTraining/prompt_vectors.pkl", "rb") as pklfile:
            self.prompt_vectors = pickle.load(pklfile)
            pklfile.close()

        self.logs.info("Loading Split Data")
        training_set = pd.read_csv(f"{ROOT_PATH}/Data/ExtractedData/ModelTraining/SplitData/TrainingSet.csv")
        validation_set = pd.read_csv(f"{ROOT_PATH}/Data/ExtractedData/ModelTraining/SplitData/ValidationSet.csv")
        test_set = pd.read_csv(f"{ROOT_PATH}/Data/ExtractedData/ModelTraining/SplitData/TestSet.csv")

        self.X_train = self.define_vector_sets(training_set)
        self.X_validation = self.define_vector_sets(validation_set)
        self.X_test = self.define_vector_sets(test_set)
        self.y_train = training_set[dependent_variable]
        self.y_validation = validation_set[dependent_variable]
        self.y_test = test_set[dependent_variable]

    def remove_saved_tuning_data(self):
        """Function to remove saved keras tuner data"""
        tuning_data_path = f"{self.dependent_variable.replace(' ', '')}/KerasTuner"
        if os.path.exists(tuning_data_path):
            self.logs.info("Removing saved tuning data")
            shutil.rmtree(tuning_data_path)

    def define_vector_sets(self, df):
        """Function to grab vectors for train, validation, and tests sets"""
        vectors = []
        for i in range(len(df)):
            # Load document and prompt vectors
            filename = df["Filename"][i]
            prompt = df["Prompt"][i]
            document_vector = self.document_vectors[prompt][filename]
            prompt_vector = self.prompt_vectors[prompt]

            # Use this line to use the document vector as is
            vectors.append(document_vector)

            # Use this code block to append cosine similarity to end of document vector
            # similarity = cosine_similarity(document_vector, prompt_vector)
            # document_vector.append(similarity)
            # vectors.append(document_vector)

            # Use this code block to concatenate the prompt and document vectors
            # vector = concatenate_vectors([prompt_vector, document_vector])
            # vectors.append(vector)

        return pd.DataFrame(vectors)

    def train_model(self):
        """Function to train model"""
        self.logs.info("Training tensorflow model")

        # Content score hyperparameters
        if self.dependent_variable == 'Content Score':
            model = keras.Sequential([
                keras.layers.InputLayer(input_shape=(1024,)),
                keras.layers.Dense(1024, activation='relu', name='hidden1'),
                keras.layers.Dense(1024, activation='relu', name='hidden2'),
                # keras.layers.Dense(1024, activation='relu', name='hidden3'),
                # keras.layers.Dense(769, activation='relu', name='hidden4'),
                # keras.layers.Dense(1536, activation='relu', name='hidden5'),
                # keras.layers.Dense(768, activation='relu', name='hidden6'),
                # keras.layers.Dense(700, activation='relu', name='hidden7'),
                # keras.layers.Dense(730, activation='leaky_relu', name='hidden8'),
                # keras.layers.Dense(760, activation='relu', name='hidden9'),
                keras.layers.Dense(1, name='output')
            ])
        # Wording score hyperparameters
        else:
            model = keras.Sequential([
                keras.layers.InputLayer(input_shape=(1024,)),
                keras.layers.Dense(1024, activation='relu', name='hidden1'),
                keras.layers.Dense(1024, activation='relu', name='hidden2'),
                # keras.layers.Dense(1024, activation='relu', name='hidden3'),
                # keras.layers.Dense(1536, activation='relu', name='hidden4'),
                # keras.layers.Dense(1536, activation='relu', name='hidden5'),
                # keras.layers.Dense(768, activation='relu', name='hidden6'),
                # keras.layers.Dense(710, activation='leaky_relu', name='hidden7'),
                # keras.layers.Dense(730, activation='relu', name='hidden8'),
                keras.layers.Dense(1, name='output')
            ])
        early_stop = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=3)

        # # Evaluate hidden layers, Number of neurons
        # tuner = kt.GridSearch(
        #     build_tuner_model,
        #     objective='val_loss',
        #     seed=33,
        #     overwrite=True,
        #     directory=f"{self.dependent_variable.replace(' ', '')}",
        #     project_name="KerasTuner"
        # )
        #
        # # tuner.search_space_summary()
        #
        # # Find best params
        # early_stop = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=3)
        # tuner.search(self.X_train, self.y_train, epochs=10, batch_size=2048, verbose=1,
        #              validation_data=(self.X_validation, self.y_validation), callbacks=[early_stop])
        # self.best_params = tuner.get_best_hyperparameters(num_trials=1)[0].values
        #
        # # Removed saved tuning data
        # self.remove_saved_tuning_data()
        #
        # # Fit model
        # self.logs.info("Training tensorflow model with best hyperparameters")
        # model = keras.Sequential([keras.layers.InputLayer(input_shape=(769,))])
        # for i in range(self.best_params['layers']):
        #     num_neurons = self.best_params[f'layer{i}']
        #     activation = self.best_params[f'activation{i}']
        #     model.add(keras.layers.Dense(num_neurons, activation, name=f'hidden{i+1}'))
        # model.add(keras.layers.Dense(1, name="output"))

        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
        model.fit(self.X_train, self.y_train, epochs=2000, batch_size=2048, verbose=1,
                  validation_data=(self.X_validation, self.y_validation), callbacks=[early_stop])

        # Plot training losses
        self.logs.info("Plotting training losses")
        losses = pd.DataFrame(model.history.history)
        plt.figure(figsize=(6, 4))
        plt.plot(losses["loss"], label="Loss")
        plt.plot(losses["val_loss"], label="Val. Loss")
        plt.title(f"{self.dependent_variable} Training Loss")
        plt.legend()
        plt.savefig(f"{self.dependent_variable.replace(' ', '')}/{self.dependent_variable.replace(' ', '')}TrainingLoss.png")

        # Evaluate model
        self.logs.info(f"Collecting {self.dependent_variable} model performance statistics")
        self.train_rmse, self.validation_rmse, self.test_rmse = calculate_performance_metrics('Tensorflow',
                                                                                              self.dependent_variable,
                                                                                              model, self.X_train,
                                                                                              self.X_validation,
                                                                                              self.X_test, self.y_train,
                                                                                              self.y_validation,
                                                                                              self.y_test,
                                                                                              best_params=self.best_params)

        # Save model
        self.logs.info(f"Saving {self.dependent_variable} model and performance statistics")
        tf.keras.models.save_model(model, f"{self.dependent_variable.replace(' ', '')}/{self.dependent_variable.replace(' ', '')}SavedModel")


# Run Script
if __name__ == '__main__':
    #  Initialize model trainers
    content_trainer = TensorflowTrainer("Content Score")
    wording_trainer = TensorflowTrainer("Wording Score")

    # Train BigModels
    content_trainer.train_model()
    wording_trainer.train_model()

    # Initialize model evaluator
    model_evaluator = ModelEvaluator('Tensorflow', content_trainer, wording_trainer)

    # Evaluate models
    model_evaluator.evaluate_models()
