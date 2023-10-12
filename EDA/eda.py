import pandas as pd
from matplotlib import pyplot as plt
from root_path import ROOT_PATH


def visualize_data_distributions():
    """Function to visualize content and wording score distributions"""

    # Load data
    data = pd.read_csv(f"{ROOT_PATH}/Data/ProvidedData/summaries_train.csv", encoding='utf-8')

    # Round scores
    data["content"] = round(data["content"], 1)
    data["wording"] = round(data["wording"], 1)

    # Sort scores
    content_dict = dict()
    wording_dict = dict()
    for i in sorted(data["content"].unique(), reverse=False):
        content_dict[i] = len(data[data["content"] == i])
    for i in sorted(data["wording"].unique(), reverse=False):
        wording_dict[i] = len(data[data["wording"] == i])

    # Plot scores
    plt.bar(content_dict.keys(), content_dict.values(), width=0.33)
    plt.xlabel("Content Score")
    plt.ylabel("Count")
    plt.title("Content Score Counts")
    plt.savefig(f"{ROOT_PATH}/EDA/ContentScoreCounts.png")
    plt.show()
    plt.bar(wording_dict.keys(), wording_dict.values(), width=0.33)
    plt.xlabel("Wording Score")
    plt.ylabel("Count")
    plt.title("Wording Score Counts")
    plt.savefig(f"{ROOT_PATH}/EDA/WordingScoreCounts.png")


if __name__ == '__main__':
    visualize_data_distributions()
