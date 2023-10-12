"""
Script to split data into train, validation, and test sets. Split is done using a rotation to make sure that each
prompt has an equal representation of documents in each set. When iterating over the documents, every fourth document
is added to the validation set, every eighth document is added to the test set, and the rest are added to the
training set. The resulting split is about 75% of the data in training set and 12.5% in each of the training and
validation sets.

The resulting sets are saved with search stat data into `Data/ExtractedData/ModelTraining/SplitData/(TestSet.csv,
TrainingSet.csv, or ValidationSet.csv).

"""

import pandas as pd
from Logs.log_config import Logger
from root_path import ROOT_PATH


class DataSplitController:
    """Class to spilt data into train, test and validation sets"""

    def __init__(self):
        self.logs = Logger()
        self.logs.info("Loading in dataset")
        self.df = pd.read_csv(f"{ROOT_PATH}/Data/ExtractedData/ModelTraining/full_search_results.csv")
        self.training_rows = []
        self.validation_rows = []
        self.test_rows = []

    def save_data_to_csv(self, rows, set_type):
        """
        Function to take training, test, or validation set rows from the main dataset and create a csv file from them
        :param rows: Training, test, or validation set row indexes
        :param set_type: String which could be any of: 'Training', 'Test', 'Validation'
        :return: None
        """
        self.logs.info(f"Saving {set_type} Set to csv")
        columns = ["Filename", "Prompt", "Text", "Content Score", "Wording Score"]
        set_df = pd.DataFrame(rows, columns=columns)
        set_df.to_csv(f"{set_type}Set.csv", index=False)

    def sort_data_with_rotation(self):
        """Function to split data using a rotation"""
        for prompt in self.df["Prompt"].unique():
            prompt_df = self.df[self.df["Prompt"] == prompt]
            prompt_df.reset_index(inplace=True, drop=False)
            counter = 1
            for i in range(len(prompt_df)):
                self.logs.info(f"Sorting data for prompt: {prompt};  Progress: {counter}/{len(prompt_df)}")
                # Extract data and assign to a set
                prompt_id = prompt_df["prompt_id"][i]
                filename = f"{prompt_id}_{prompt_df['student_id'][i]}.txt"
                text = prompt_df["text"][i]
                content_score = prompt_df["content"][i]
                wording_score = prompt_df["wording"][i]
                row = (filename, prompt_id, text, content_score, wording_score)
                if counter % 8 == 0:
                    self.test_rows.append(row)
                elif counter % 4 == 0:
                    self.validation_rows.append(row)
                else:
                    self.training_rows.append(row)
                counter += 1

    def split_data(self):
        """Function to split and save data"""
        self.sort_data_with_rotation()
        self.save_data_to_csv(self.training_rows, 'Training')
        self.save_data_to_csv(self.test_rows, 'Test')
        self.save_data_to_csv(self.validation_rows, 'Validation')


if __name__ == '__main__':
    data_split_controller = DataSplitController()
    data_split_controller.split_data()
