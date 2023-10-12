import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from root_path import ROOT_PATH


def calculate_adj_r2(r2, data):
    """
    Function to calculate adjusted r2
    :param r2: Previously calculated regular r2 number
    :param data: Dataframe from which r^2 was calculated
    :return:  Adjusted r2 value
    """
    df = pd.DataFrame(data)
    num_observations = len(df)
    num_features = len(df.columns)
    return 1 - (1-r2) * (num_observations-1)/(num_observations-num_features-1)


def calculate_mcrmse(content_rmse, wording_rmse):
    """
    Function to calculate Mean Columnwise RMSE
    :param content_rmse: Content Score RMSE
    :param wording_rmse: Wording Scpre RMSE
    :return: MCRMSE value
    """
    return (content_rmse + wording_rmse)/2


def calculate_performance_metrics(model_type, dependent_variable, model, X_train, X_validation, X_test, y_train,
                                  y_validation, y_test, best_params=None, significant_feature_names=None):
    """
     Function to calculate RMSE, R-squared, and Adj. R-squared, and save to a file
    :param model_type: The kind of model being evaluated, one of 'XGBoost', 'MLPRegressor', 'Tensorflow'
    :param dependent_variable: A string, either 'Content Score' or 'Wording Score'
    :param model: The model being evaluated
    :param X_train: Training set features
    :param X_validation: Validation set features
    :param X_test: Test set features
    :param y_train: Training set actual values
    :param y_validation: Validation set actual values
    :param y_test: Test set actual values
    :param best_params: Best hyperparameters (xgboost only)
    :param significant_feature_names: Features used if feature selection occurred
    :return: None
    """

    # Make predictions
    train_predictions = model.predict(X_train)
    validation_predictions = model.predict(X_validation)
    test_predictions = model.predict(X_test)

    # Calculate RMSE
    train_rmse = mean_squared_error(y_train, train_predictions, squared=False)
    validation_rmse = mean_squared_error(y_validation, validation_predictions, squared=False)
    test_rmse = mean_squared_error(y_test, test_predictions, squared=False)

    # Calculate R-squared
    train_r2 = r2_score(y_train, train_predictions)
    validation_r2 = r2_score(y_validation, validation_predictions)
    test_r2 = r2_score(y_test, test_predictions)

    # Calculate Adj. R-Squared
    train_adj_r2 = calculate_adj_r2(train_r2, X_train)
    validation_adj_r2 = calculate_adj_r2(validation_r2, X_validation)
    test_adj_r2 = calculate_adj_r2(test_r2, X_test)

    # Write evaluation stats to a file
    if model_type == 'XGBoost':
        file = open(f"{ROOT_PATH}/Models/{model_type}/{dependent_variable.replace(' ', '')}/{dependent_variable.replace(' ', '')}Stats.txt", 'w')
    else:
        file = open(f"{ROOT_PATH}/Models/NeuralNetwork/{model_type}/{dependent_variable.replace(' ', '')}/{dependent_variable.replace(' ', '')}Stats.txt", 'w')
    file.write(f"""\n- Selected Features: {significant_feature_names}
    \n- Best Params: {best_params}
    \n- Training RMSE: {train_rmse}\n- Training R-Squared: {train_r2}\n- Training Adj. R-squared: {train_adj_r2}
    \n- Validation RMSE: {validation_rmse}\n- Test R-squared: {validation_r2}\n- Test Adj R-Squared: {validation_adj_r2}
    \n- Test RMSE: {test_rmse}\n- Test R-squared: {test_r2}\n- Test Adj R-Squared: {test_adj_r2}
    """)
    file.close()

    return train_rmse, validation_rmse, test_rmse


class ModelEvaluator:

    def __init__(self, model_type, content_model, wording_model):
        """
        Object to evaluate content and wording score models and save performance stats to a file
        :param model_type: The general type of model being evaluated, either 'XGBoost' or 'NeuralNetwork'
        :param content_model: The content score model being evaluated
        :param wording_model: The wording score model being evaluated
        """
        self.model_type = model_type
        self.content_trainer = content_model
        self.wording_trainer = wording_model

    def evaluate_models(self):
        """Function to train and evaluate models. Write MCRMSE stats to a file"""
        # Calculate MCRMSE
        train_mcrmse = calculate_mcrmse(self.content_trainer.train_rmse, self.wording_trainer.train_rmse)
        validation_mcrmse = calculate_mcrmse(self.content_trainer.validation_rmse, self.wording_trainer.validation_rmse)
        test_mcrmse = calculate_mcrmse(self.content_trainer.test_rmse, self.wording_trainer.test_rmse)

        # Write CRMSE to file
        if self.model_type == 'XGBoost':
            file = open(f"{ROOT_PATH}/Models/XGBoost/MeanColumnwiseRMSESummary.txt", 'w')
        else:
            file = open(f"{ROOT_PATH}/Models/NeuralNetwork/{self.model_type}/MeanColumnwiseRMSESummary.txt", 'w')
        file.write(f"""- Training MCRMSE: {train_mcrmse}\n- Validation MCRMSE: {validation_mcrmse}\n- Test MCRMSE: {test_mcrmse}""")
        file.close()
