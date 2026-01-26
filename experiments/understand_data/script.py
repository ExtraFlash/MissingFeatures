import json
import os


def analyze_dataset(dataset_name):
    ...


def main():
    config_path = '../../datasets/config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    datasets_names = [dataset['name'] for dataset in config['datasets']]

    print(datasets_names)

    dataset_name = datasets_names[0]
    analyze_dataset(dataset_name)


if __name__ == '__main__':
    main()
