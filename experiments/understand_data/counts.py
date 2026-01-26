import json
import pandas as pd


if __name__ == "__main__":
    config_path = "../../datasets/config.json"
    # get datasets config
    with open(config_path) as f:
        config = json.load(f)
    # get datasets from config
    datasets: list = config['datasets']
    #  datasets_names: list = [
    #      "Tokyo",
    #      "Connectionist Bench",
    #      "Ionosphere",
    #      "Pima Indians Diabetes Database",
    #     "Heart Disease",
    #       "Statlog (German Credit Data)",
    #   ]
    # get only the datasets that are in the list
    # datasets = [dataset for dataset in datasets if dataset['name'] in datasets_names]
    # run experiment for each dataset
    for dataset in datasets:
        # if dataset['name'] not in ["electricity"]:
        #     print(f'skipping {dataset["name"]}')
        #     continue
        # print(f'current dataset: {dataset["name"]}')
        dataset_name_ = dataset['name']

        # Load data
        data_path = "../../data"
        train_set = pd.read_csv(f"{data_path}/{dataset_name_}/train/data.csv")

        print(f"{dataset_name_} & {len(train_set)} & {train_set.shape[1] - 1} & 2")
