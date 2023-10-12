"""
Run the main section at the bottom of this file to train and evaluate models for content and wording scores.

Inside the folder for each model in Models/XGBoost/(Content or Wording Score) there will be 3 files.
ScoreStats.txt has info on RMSE and R-squared.  ScoreTrainingLoss.png contains a plot of the training losses to gauge
over fitting. ScoreModel.keras is the model that can be loaded at any time.

Inside Models/XGBoost there is a file MeanColumwiseRMSESummary.txt which contains info on the final
competition metric which is MCRMSE.

The current configuration contains uses vectors of 1000 dimensions. These can be changed if vectors of a different size
are going to be used by changing the higher end of the range in line 111 to the length of the vectors plus one.

This script automatically tunes hyperparemeters for the final model through a grid search cross validation.
To edit which hyperparameters are tested and the range of values for each hyperparameter tested, edit the
`param_tuning` dictionary on line 187.

"""

import pickle
import warnings
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
from numpy import sort
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from Logs.log_config import Logger
from Models.model_evaluator import ModelEvaluator, calculate_performance_metrics
from root_path import ROOT_PATH

# Ignore sklearn warnings
warnings.filterwarnings(action='ignore', category=UserWarning)


def cosine_similarity(vector1, vector2):
    """Function to calculate cosine similarity between two vectors"""
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))


def normalize_vector(vector):
    """Function to normalize a vector"""
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return [i/norm for i in vector]


class VectorXGBoostTrainer:
    """Class to train an xgboost model"""

    def __init__(self, dependent_variable):
        self.logs = Logger()
        self.dependent_variable = dependent_variable
        self.best_params = None
        self.train_rmse = None
        self.validation_rmse = None
        self.test_rmse = None

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

    def define_vector_sets(self, df):
        """Function to grab vectors for train, validation, and tests sets"""
        vectors = []
        for i in range(len(df)):
            filename = df["Filename"][i]
            prompt = df["Prompt"][i]
            # Load document vectors, prompt vectors, tfidf scores
            document_vector = self.document_vectors[prompt][filename]

            # Save vector
            vectors.append(document_vector)

        # Return vector data frame
        columns = [f"Component {i}" for i in range(1, 351)]
        # columns.append("Cosine Similarity to Prompt")
        return pd.DataFrame(vectors, columns=columns)

    def find_feature_importance(self):
        """Function to plot and select most important features using the elbow method"""
        self.logs.info("Training XGB model with default hyperparameters to find feature importance")

        # Find feature significance
        model = XGBRegressor(random_state=33)
        model.fit(self.X_train, self.y_train)

        self.logs.info("Finding feature significance")
        thresholds = set(sort(list(model.feature_importances_)))

        # Create a model using each feature importance as a feature threshold, pick threshold with lowest RMSE
        model_thresholds = dict()
        file = open(f"{self.dependent_variable.replace(' ', '')}/{self.dependent_variable.replace(' ', '')}ThresholdEval.txt", 'w')
        counter = 1
        # Iterate through thresholds
        for thresh in thresholds:
            features = SelectFromModel(estimator=model, threshold=thresh, prefit=True)
            # Transform feature sets
            features_X_train = features.transform(self.X_train)
            features_X_validation = features.transform(self.X_validation)
            # Fit model
            feature_model = xgb.XGBRegressor(random_state=33)
            feature_model.fit(features_X_train, self.y_train)
            # Make predictions
            predictions = feature_model.predict(features_X_validation)
            rmse = mean_squared_error(self.y_validation, predictions, squared=False)
            model_thresholds[thresh] = rmse
            # Save to file
            file.write(f"\n- Threshold: {thresh}  - RMSE: {rmse}")
            self.logs.info(f"Evaluating feature threshold: {thresh}; - RMSE: {rmse}; - PROGRESS: {counter}/{len(thresholds)}")
            counter += 1
        file.close()

        # Find threshold that has best RMSE
        min_rmse = min(model_thresholds.values())
        for thresh in model_thresholds:
            if model_thresholds[thresh] == min_rmse:
                self.threshold = thresh
                break
        self.logs.info(f"Found best threshold is {self.threshold} with RMSE of {min_rmse}")
        self.logs.info("Filtering data to only include selected features")

        # Use the best threshold for final feature selection
        significant_features = SelectFromModel(model, threshold=self.threshold, prefit=True)
        self.significant_feature_names = [self.X_train.columns[i] for i in significant_features.get_support(indices=True)]

        # Save most significant features to a file
        self.logs.info("Writing feature names and importance to a file")
        with open(f"{ROOT_PATH}/BigModels/FeatureSelection/{self.dependent_variable.replace(' ', '')}MostSignificantFeatures.pkl", "wb") as pklfile:
            pickle.dump(self.significant_feature_names, pklfile)
            pklfile.close()

        # Save second file with importance
        importance_vals = model.feature_importances_
        importance_dict = dict(sorted({model.feature_names_in_[i]: str(importance_vals[i]) for i in range(len(importance_vals)) if importance_vals[i] >= self.threshold}.items(),
                                      key=lambda x: x[1], reverse=True))
        with open(f"{self.dependent_variable.replace(' ', '')}/{self.dependent_variable.replace(' ', '')}MostSignificantFeatureValues.txt", "w") as file:
            file.write(str(importance_dict))
            file.close()

        # Update X_train and X_test with selected features
        self.X_train = significant_features.transform(self.X_train)
        self.X_validation = significant_features.transform(self.X_validation)
        self.X_test = significant_features.transform(self.X_test)

    def find_optimal_hyperparameters(self):
        """Function which uses cross validation to find optimal model hyperparameters"""
        # Create xgboost D-matrices
        self.logs.info("Creating D-matrices and setting parameter values for cross validation")
        d_train = xgb.DMatrix(self.X_train, self.y_train, enable_categorical=True)
        d_test = xgb.DMatrix(self.X_validation, self.y_validation, enable_categorical=True)

        # Create dictionary of potential parameters for testing in cross validation
        param_tuning = {
            "max_depth": np.arange(3, 5),
            "learning_rate": np.arange(0.1, 1, 0.1),
            "n_estimators": np.arange(100, 500, 100),
            "gamma": np.arange(0, 2)
        }

        # Use grid search to perform k-fold cross validation with k=5 to find best parameters
        self.logs.info("Performing 5 fold cross validation:")
        xgb_object = XGBRegressor(random_state=33)
        params_model = GridSearchCV(estimator=xgb_object, param_grid=param_tuning, scoring="neg_mean_squared_error",
                                    verbose=10, n_jobs=-1)
        params_model.fit(self.X_train, self.y_train)
        self.best_params = params_model.best_params_

    def train_model(self):
        """Function to train an xgboost model"""
        self.logs.info(f"Creating final {self.dependent_variable} model with best parameters from cross validation")
        model = XGBRegressor(**self.best_params, random_state=33)
        model.fit(self.X_train, self.y_train)

        # Evaluate model
        self.logs.info(f"Collecting model {self.dependent_variable} performance statistics")

        # Evaluate model
        self.train_rmse, self.validation_rmse, self.test_rmse = calculate_performance_metrics('XGBoost',
                                                                                              self.dependent_variable,
                                                                                              model, self.X_train,
                                                                                              self.X_validation,
                                                                                              self.X_test, self.y_train,
                                                                                              self.y_validation,
                                                                                              self.y_test,
                                                                                              best_params=self.best_params,
                                                                                              significant_feature_names=self.significant_feature_names)

        # Save Model
        self.logs.info(f"Saving {self.dependent_variable} model and performance statistics")
        model.save_model(f"{self.dependent_variable.replace(' ', '')}/{self.dependent_variable.replace(' ', '')}XGBoostModel.json")

    def create_model(self):
        self.find_feature_importance()
        self.find_optimal_hyperparameters()
        self.train_model()


if __name__ == '__main__':
    #  Initialize model trainers
    content_trainer = VectorXGBoostTrainer("Content Score")
    wording_trainer = VectorXGBoostTrainer("Wording Score")

    # Train BigModels
    content_trainer.create_model()
    wording_trainer.create_model()

    # Initialize model evaluator
    model_evaluator = ModelEvaluator('XGBoost', content_trainer, wording_trainer)

    # Evaluate models
    model_evaluator.evaluate_models()
