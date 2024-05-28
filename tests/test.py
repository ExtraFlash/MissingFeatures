import json
import pandas as pd

config_path = "../datasets/config.json"

with open(config_path) as f:
    config = json.load(f)

datasets: list = config['datasets']

dataset = datasets[-1]

dataset_name = dataset['name']
relative_path = dataset['relative_path']
label_position = dataset['label_position']
has_header = dataset['has_header']
has_id = dataset['has_id']
positive = dataset.get('positive', None)
negative = dataset.get('negative', None)

# Load data
datasets_path = "../datasets"

# Define a custom delimiter regex
delimiter_pattern = '\s+'

path = datasets_path + '/' + relative_path
if not has_header:
    data = pd.read_csv(path, header=None, sep=delimiter_pattern)
else:
    data = pd.read_csv(path, sep=delimiter_pattern)

# Replace categories with numbers for the label
if positive:  # if there is a positive and negative strings, replace them with 1 and 0
    # categories = data.iloc[:, -1].unique()
    print('nowww')
    data.iloc[:, -1].replace([positive, negative], [1, 0], inplace=True)
print(data.head())


