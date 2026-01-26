import json

with open("results_Ionosphere.json") as f:
    ratios = json.load(f)
print(ratios)
print('---')
print(type(ratios))