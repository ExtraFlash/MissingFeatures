import pandas as pd
from matplotlib import pyplot as plt
import json
import numpy as np
import math
from my_models import ModelFactory
from plots_utils import utils


def make_plot(dataset_name: str, dataset_plot_name: str, metric: str, metric_title: str,
              plot_row_index: int, plot_col_index: int):
    # First subplot
    # plt.subplot(2, 3, plot_index)
    # plt.clf()
    utils.make_style(plt)
    ax = axs[plot_row_index, plot_col_index]
    # get data
    cvs = 5

    colors = plt.cm.tab20(np.linspace(0, 1, len(models_names)))  # Use tab20 for unique colors
    # line_styles = ['-', '--', '-.', ':']  # Define different line styles

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

    for idx, model_name in enumerate(models_names):
        means = np.array([np.mean(scores) for scores in results[model_name].values()])
        stds = np.array([np.std(scores, ddof=1) / np.sqrt(np.size(scores)) for scores in results[model_name].values()])
        # plot the mean scores
        ax.plot(features_amounts, means, label=model_name,
                color=colors[idx % len(colors)])
        # plot errors
        ax.fill_between(features_amounts, means + stds, means - stds,
                        color=colors[idx % len(colors)], alpha=0.2)

    ax.set_xlabel("Num features")
    if plot_col_index == 0:
        ax.set_ylabel(metric_title, rotation=90)
    ax.set_xticks(x_ticks)
    # ax = plt.gca()
    ax.invert_xaxis()
    ax.grid()
    # Set y-axis limits
    ax.set_ylim(0.5, 1.0)

    if plot_row_index == 0 and plot_col_index == 0:
        # ax.legend()
        global handles, labels
        handles, labels = ax.get_legend_handles_labels()
    ax.set_title(dataset_plot_name)


if __name__ == '__main__':
    handles, labels = None, None
    models_names = MODELS = [
        ModelFactory.Mean_Gradient_Boosting_Classifier_Name,
        # DAE_NAME,
        # ModelFactory.GAT_NAME,
        # ModelFactory.GCN_NAME,
        ModelFactory.Denoising_Graph_Encoder_Name,
        ModelFactory.Complete_Random_Forest_NAME,
        ModelFactory.Complete_Gradient_Boosting_Classifier_Name,
        ModelFactory.XGB_NAME,
        # DAE_Dynamic_TYPE_LIGHTNING_NAME,
        # Neural_Network_NAME,
        # Teacher_Students_NAME,
        ModelFactory.Random_Forest_NAME,
        # ModelFactory.Ada_Boost_NAME,
        ModelFactory.LGBM_NAME,
        # Logistic_Regression_NAME
    ]

    metrics = ['auc', 'accuracy']
    metrics_titles = ['AUC', 'Accuracy']

    config_path = "../../datasets/config.json"

    # get datasets config
    with open(config_path) as f:
        config = json.load(f)
    # get datasets from config
    datasets: list = config['datasets']
    # datasets_names: list = [
    #     "Tokyo",
    #     "Connectionist Bench",
    #     "Ionosphere",
    #     "Pima Indians Diabetes Database",
    #     "Heart Disease",
    #     "Statlog (German Credit Data)",
    # ]
    # get only the datasets that are in the list
    # datasets = [dataset for dataset in datasets if dataset['name'] in datasets_names]

    num_datasets = len(datasets)
    num_cols = 4  # Define columns dynamically based on how you want the layout to look
    num_rows = math.ceil(num_datasets / num_cols)  # Calculate the required number of rows

    # run experiment for each dataset
    for i, metric_ in enumerate(metrics):

        # Create a figure and a set of subplots
        # fig, axs = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(3.5 * num_cols, 3 * num_rows), layout="constrained")

        axs = np.array(axs)  # Convert to array in case of single row
        axs = axs.reshape(num_rows, num_cols)  # Ensure it's always a 2D array

        for j, dataset in enumerate(datasets):
            dataset_name_ = dataset['name']
            dataset_plot_name_ = dataset['plot_name']
            relative_path_ = dataset['relative_path']
            label_position_ = dataset['label_position']
            has_header_ = dataset['has_header']
            has_id_ = dataset['has_id']
            is_multy_class_ = dataset['is_multy_class']

            # col = j % 3
            # row = (j - col) // 3
            # print(row, col)

            row, col = divmod(j, num_cols)  # Compute row and col position dynamically

            make_plot(dataset_name=dataset_name_,
                      dataset_plot_name=dataset_plot_name_,
                      metric=metric_,
                      metric_title=metrics_titles[i],
                      plot_row_index=row,
                      plot_col_index=col)

        # Hide empty subplots if the number of datasets is not a perfect fit
        for j in range(num_datasets, num_rows * num_cols):
            row, col = divmod(j, num_cols)
            fig.delaxes(axs[row, col])

        # Place legend outside the subplots
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=len(models_names), fontsize=10)

        # axs[1, 2].set_visible(False)
        plt.savefig(f"{metric_}.png", bbox_inches='tight')
