import json
from tqdm import tqdm
import pandas as pd
from matplotlib import pyplot as plt

def run_statistics(dataset_name: str):
    # Load data
    data_path = "../../data"
    train = pd.read_csv(f"{data_path}/{dataset_name}/train/data.csv")

    y_train = train.iloc[:, -1]
    value_counts = y_train.value_counts()

    classes = [int(i) for i in value_counts.index]
    values = value_counts.values

    plt.clf()
    plt.bar(classes, values)
    plt.xlabel('Class')
    plt.xticks(classes)
    plt.ylabel('Amount of samples')
    plt.title(f'{dataset_name}')
    plt.savefig(f"results_{dataset_name}.png")




if __name__ == "__main__":
    config_path = "../../datasets/config.json"
    # get datasets config
    with open(config_path) as f:
        config = json.load(f)
    # get datasets from config
    datasets: list = config['datasets']
    # run experiment for each dataset
    for dataset in tqdm(datasets):
        dataset_name_ = dataset['name']

        run_statistics(dataset_name_)
