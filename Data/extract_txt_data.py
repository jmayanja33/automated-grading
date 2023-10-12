import os
import pandas as pd
from Logs.log_config import Logger
from root_path import ROOT_PATH


def extract_data(full_data, logs):
    """
    Function to split data and write to text files
    :param full_data: Full csv dataset
    :return: None
    """
    train_scores = []
    test_scores = []
    max_x = max(full_data["content"].max(), full_data["wording"].max())*2
    min_x = min(full_data["content"].min(), full_data["wording"].min())
    logs.info(f"Extracting text:")
    # Iterate through each prompt and split data
    for prompt in full_data["prompt_id"].unique():
        prompt_df = full_data[full_data["prompt_id"] == prompt]
        train_df, test_df = split_data(prompt_df)
        train_scores += write_txt_files(train_df, prompt, 'train')
        test_scores += write_txt_files(test_df, prompt, 'test')

    # Create csv file with scores
    scores_list = train_scores + test_scores
    scores_df = pd.DataFrame(scores_list, columns=["filetype", "promptid", "filename", "contentscore", "wordingscore"])
    scores_df.to_csv(f'{ROOT_PATH}/Data/ExtractedData/essays_scores.csv', index=False)


def split_data(prompt_df):
    """
    Function to split each prompt type in to training and test sets. 100 docs in train set, rest in test
    :param prompt_df: All rows of csv file of a certain prompt
    :return: 2 data frames of split train and test data
    """

    # GRADING SCALE:
    # x =< -2 --> Failing
    # -2 < x <= -1 --> Below Average
    # -1 < x < 1 --> Average
    # 1 <= x < 2 --> Below Average
    # 2 <= x --> Strong

    # Assign rows for training and test set
    logs.info(f"Splitting data into training and test sets")
    prompt_df.reset_index(inplace=True, drop=True)
    counter = 0
    train_rows = []
    while len(train_rows) < 75:
        if prompt_df["content"][counter] >= 1.35 and prompt_df["wording"][counter] >= 1.35:
            train_rows.append(counter)
        counter += 1
    # train_rows = random.sample(range(0, len(prompt_df)), 100)
    test_rows = [i for i in range(0, len(prompt_df)) if i not in set(train_rows)]

    # Create training and test sets
    train_df = prompt_df.iloc[train_rows]
    train_df.reset_index(inplace=True, drop=True)
    test_df = prompt_df.iloc[test_rows]
    test_df.reset_index(inplace=True, drop=True)

    return train_df, test_df


def write_txt_files(df, prompt, file_type):
    """
    Function to write all files in train or test set to txt files
    :param df: Training or test data frame
    :param prompt: Prompt id
    :param file_type: String to indicate if the training or test set is being processed
    :return: List of tuples of documents with their scores to turn into a dataframe
    """

    logs.info(f"Writing {file_type} txt files for prompt: {prompt}")
    scores = []
    for i in range(len(df)):
        student_id = df["student_id"][i]
        text = df["text"][i]
        filename = f"{prompt}_{student_id}.txt"
        content = df["content"][i]
        wording = df["wording"][i]

        # Write text to file in train or test folder
        # if file_type == "train":
        #     FILE_DIR_PATH = f"{ROOT_PATH}/Data/ExtractedData/SkillTraining/train/{prompt}"
        # else:
        #     FILE_DIR_PATH = f"{ROOT_PATH}/Data/ExtractedData/SkillTraining/test/{prompt}"
        # if not os.path.exists(FILE_DIR_PATH):
        #     os.mkdir(FILE_DIR_PATH)
        # file = open(f"{FILE_DIR_PATH}/{filename}", 'w', encoding='utf-8')
        # file.write(text)
        # file.close()

        # Write text to file in model training folder
        FILE_DIR_PATH = f"{ROOT_PATH}/Data/ExtractedData/ModelTraining/FullData/{prompt}"
        if not os.path.exists(FILE_DIR_PATH):
            os.mkdir(FILE_DIR_PATH)
        file = open(f"{FILE_DIR_PATH}/{filename}", 'w', encoding='utf-8')
        file.write(text)
        file.close()

        # Scale scores
        scores.append((file_type, prompt, filename, content, wording))

    return scores


# Load data
if __name__ == "__main__":
    logs = Logger()
    data = pd.read_csv(f"{ROOT_PATH}/Data/ProvidedData/summaries_train.csv", encoding='utf-8')
    
    # Extract text, write files, and get scores
    extract_data(data, logs)
