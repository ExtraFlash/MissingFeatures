import pandas as pd
from matplotlib import pyplot as plt
import json
import numpy as np
from my_models import ModelFactory
from plots_utils import utils


def make_plot(dataset_name: str, dataset_plot_name: str, metric: str, metric_title: str,
              plot_row_index: int, plot_col_index: int):
    utils.make_style(plt)
    ax = axs[plot_row_index, plot_col_index]
    # get data
    cvs = 5
    models_names = ModelFactory.MODELS
    # {'model': {max_features: scores}, 'model': ...}
    results = {model_name: [] for model_name in models_names}
    # get the max features count
    data = pd.read_csv(f"{dataset_name}/cv_0/{metric}.csv")
    max_features = data["num_features"].max()

    for model_name in models_names:
        scores = []
        for cv in range(cvs):
            data = pd.read_csv(f"{dataset_name}/cv_{cv}/{metric}.csv", index_col=0)
            score = data.loc[max_features, model_name]
            scores.append(score)
        # store scores for the model at the max feature count
        results[model_name] = scores

    # make plot
    for model_name in models_names:
        means = np.mean(results[model_name])
        stds = np.std(results[model_name], ddof=1) / np.sqrt(len(results[model_name]))
        # plot the mean score for max features only
        ax.bar(model_name, means, yerr=stds, label=model_name, alpha=0.7)

    ax.set_xlabel("Model")
    if plot_col_index == 0:
        ax.set_ylabel(metric_title, rotation=90)
    ax.grid()
    ax.set_title(dataset_plot_name)
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_xticklabels([])


if __name__ == '__main__':
    metrics = ['auc', 'accuracy']
    metrics_titles = ['AUC', 'Accuracy']

    config_path = "../../datasets/config.json"

    # get datasets config
    with open(config_path) as f:
        config = json.load(f)
    # get datasets from config
    datasets: list = config['datasets']
    # run experiment for each dataset
    for i, metric_ in enumerate(metrics):

        # Create a figure and a set of subplots
        fig, axs = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")

        for j, dataset in enumerate(datasets):
            dataset_name_ = dataset['name']
            dataset_plot_name_ = dataset['plot_name']
            relative_path_ = dataset['relative_path']
            label_position_ = dataset['label_position']
            has_header_ = dataset['has_header']
            has_id_ = dataset['has_id']
            is_multy_class_ = dataset['is_multy_class']

            col = j % 3
            row = (j - col) // 3

            make_plot(dataset_name=dataset_name_,
                      dataset_plot_name=dataset_plot_name_,
                      metric=metric_,
                      metric_title=metrics_titles[i],
                      plot_row_index=row,
                      plot_col_index=col)

        # Add a single legend for all subplots
        handles, labels = axs[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4)

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(f"{metric_}_max_features.png", bbox_inches='tight')
