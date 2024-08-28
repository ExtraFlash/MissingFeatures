import pandas as pd
from matplotlib import pyplot as plt
import json
import numpy as np
from models import ModelFactory
from plots_utils import utils


def make_plot(dataset_name: str, metric: str, metric_title: str):
    # First subplot
    # plt.subplot(2, 3, plot_index)
    # plt.clf()
    utils.make_style(plt)
    # ax = axs
    # get data
    cvs = 5
    models_names = ModelFactory.MODELS
    # {'model': {50: scores, 49, scores, ...}, 'model': ...}
    results = {model_name: {} for model_name in models_names}
    # take the features amounts list from the first cv
    data = pd.read_csv(f"{dataset_name}/cv_0/{metric}.csv")
    features_amounts = data["num_features"].to_list()

    for model_name in models_names:
        for features_amount in features_amounts:
            # scores for model with features over all the cvs
            scores = []
            for cv in range(cvs):
                data = pd.read_csv(f"{dataset_name}/cv_{cv}/{metric}.csv", index_col=0)
                score = data.loc[features_amount, model_name]
                scores.append(score)
            # save scores in results
            results[model_name][features_amount] = scores

    # make plot
    x_ticks = features_amounts[::(len(features_amounts) // 5) + 1]
    if 1 not in x_ticks:
        x_ticks.append(1)

    for model_name in models_names:
        means = np.array([np.mean(scores) for scores in results[model_name].values()])
        stds = np.array([np.std(scores, ddof=1) / np.sqrt(np.size(scores)) for scores in results[model_name].values()])
        # plot the mean scores
        plt.plot(features_amounts, means, label=model_name)
        # plot errors
        plt.fill_between(features_amounts, means+stds, means-stds, alpha=0.2)

    plt.xlabel("Num features")
    plt.ylabel(metric_title, rotation=90)
    plt.xticks(x_ticks)
    # ax = plt.gca()

    # invert the x axis
    ax = plt.gca()
    ax.invert_xaxis()
    plt.grid()
    plt.legend()
    plt.title("Simulated data")


if __name__ == '__main__':
    metrics = ['auc', 'accuracy']
    metrics_titles = ['AUC', 'Accuracy']

    # run experiment for each dataset
    for i, metric_ in enumerate(metrics):

        # Create a figure and a set of subplots
        plt.figure()
        # fig, axs = plt.subplots(1, 1, figsize=(13, 7), layout="constrained")

        dataset_name_ = "data"
        # print(row, col)

        make_plot(dataset_name=dataset_name_,
                  metric=metric_,
                  metric_title=metrics_titles[i])
        # axs[1, 2].set_visible(False)
        plt.savefig(f"{metric_}.png", bbox_inches='tight')
