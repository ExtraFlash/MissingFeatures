import pandas as pd
from matplotlib import pyplot as plt
import json
import numpy as np
import math
from my_models import ModelFactory
from plots_utils import utils
from experiments.calculate_datasets_scores import plot_style


def make_plot(dataset_name: str, dataset_plot_name: str, metric: str, metric_title: str,
              plot_row_index: int, plot_col_index: int, is_difference: bool,
              models_names: list, placeholder_model_name: str):
    utils.make_style(plt)
    ax = axs[plot_row_index, plot_col_index]
    cvs = 5

    colors = plt.cm.tab20(np.linspace(0, 1, len(models_names)))

    results = {model_name: {} for model_name in models_names}
    results_max = {model_name: {} for model_name in models_names}

    # take the features amounts list from the first cv
    data = pd.read_csv(f"data_percentage_100/{dataset_name}/{placeholder_model_name}/cv_0/{metric}.csv")
    features_amounts = data["num_features"].to_list()
    total_features = features_amounts[0]
    features_amounts_filtered = [total_features] + [k for k in features_amounts[1:] if k % 5 == 1]
    features_amounts_filtered_max = [k for k in features_amounts[1:] if k % 5 == 1]

    for model_name in models_names:
        for features_amount in features_amounts_filtered:
            scores = []
            for cv in range(cvs):
                data = pd.read_csv(f"data_percentage_100/{dataset_name}/{model_name}/cv_{cv}/{metric}.csv", index_col=0)
                score = data.loc[features_amount, 'auc']
                scores.append(score)
            results[model_name][features_amount] = scores

    for model_name in models_names:
        for features_amount in features_amounts_filtered_max:
            scores_max = []
            for cv in range(cvs):
                data = pd.read_csv(f"data_percentage_100/{dataset_name}/{model_name}/cv_{cv}/{metric}_available.csv",
                                   index_col=0)
                score = data.loc[features_amount, 'auc']
                scores_max.append(score)
            results_max[model_name][features_amount] = scores_max

    x_ticks = features_amounts[::(len(features_amounts) // 5) + 1]
    if 1 not in x_ticks:
        x_ticks.append(1)

    for idx, model_name in enumerate(models_names):
        means = np.array([np.mean(scores) for scores in results[model_name].values()])
        stds = np.array(
            [np.std(scores, ddof=1) / np.sqrt(np.size(scores)) for scores in results[model_name].values()])

        means_max = np.array([np.mean(scores) for scores in results_max[model_name].values()])
        stds_max = np.array(
            [np.std(scores, ddof=1) / np.sqrt(np.size(scores)) for scores in results_max[model_name].values()])

        # Calculate the difference (results_max - results)
        if len(means_max) < len(means):
            diff_means = means_max - means[1:]
        else:
            diff_means = means_max - means

        # For error propagation, assuming independence: sqrt(std1^2 + std2^2)
        if len(means_max) < len(means):
            diff_stds = np.sqrt(stds[1:] ** 2 + stds_max ** 2)
        else:
            diff_stds = np.sqrt(stds ** 2 + stds_max ** 2)

        if model_name == ModelFactory.DAE_NAME:
            color = ModelFactory.color_for(model_name)
        else:
            color = colors[idx % len(colors)]

        # Plot the AUC or AUC Difference
        if not is_difference:
            ax.plot(features_amounts_filtered, means,
                    label=ModelFactory.display_name(model_name),
                    color=color)
            ax.fill_between(features_amounts_filtered, means + stds, means - stds,
                            color=color, alpha=0.2)
        else:
            ax.plot(features_amounts_filtered_max, diff_means,
                    label=ModelFactory.display_name(model_name),
                    color=color)
            ax.fill_between(features_amounts_filtered_max, diff_means + diff_stds, diff_means - diff_stds,
                            color=color, alpha=0.2)

    if plot_row_index == 1:
        ax.set_xlabel("Num features")
    if plot_col_index == 0:
        if is_difference:
            ax.set_ylabel(f"{metric_title} Difference", rotation=90)
        else:
            ax.set_ylabel(metric_title, rotation=90)

    ax.set_xticks(x_ticks)
    ax.invert_xaxis()
    ax.grid()

    # Set y-axis limits
    if is_difference:
        ax.set_ylim(-0.5, 0.5)
    else:
        ax.set_ylim(0.4, 1.0)

    if plot_row_index == 0 and plot_col_index == 0:
        global handles, labels
        handles, labels = ax.get_legend_handles_labels()

    if plot_row_index == 0:
        ax.set_title(dataset_plot_name)


if __name__ == '__main__':
    plot_style.set_plot_style()

    handles, labels = None, None

    placeholder_model_name = ModelFactory.MODELS[0]

    auc_file = pd.read_csv(f"data_percentage_100/Tokyo/{placeholder_model_name}/cv_0/auc_available.csv")

    models_type = 'xgb'

    models_names = [ModelFactory.DAE_NAME]

    if models_type == 'random_forest':
        models_names += [
            ModelFactory.Random_Forest_Imputation_MEAN,
            ModelFactory.Random_Forest_Imputation_KNN
        ]
    elif models_type == 'xgb':
        models_names += [
            ModelFactory.XGB_Imputation_MEAN,
            ModelFactory.XGB_Imputation_KNN
        ]

    elif models_type == 'lgbm':
        models_names += [
            ModelFactory.LGBM_Imputation_MEAN,
            ModelFactory.LGBM_Imputation_KNN
        ]

    else:
        raise ValueError(f"Unknown models_type: {models_type}")

    metric = 'auc'
    metric_title = 'AUC'

    config_path = "../../datasets/config.json"

    # get datasets config
    with open(config_path) as f:
        config = json.load(f)

    # get datasets from config
    datasets: list = config['datasets']

    num_datasets = 4
    num_cols = 4
    num_rows = 2

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(3.5 * num_cols, 3 * num_rows), layout="constrained")

    axs = np.array(axs).reshape(num_rows, num_cols)

    dataset_index = 0

    # Plot AUC
    for j, dataset in enumerate(datasets):
        dataset_name_ = dataset['name']

        if dataset_name_ not in ModelFactory.DATASETS_NON_HOMOPHILITY:
            continue

        dataset_plot_name_ = dataset['plot_name']

        row, col = divmod(dataset_index, num_cols)

        make_plot(dataset_name=dataset_name_,
                  dataset_plot_name=dataset_plot_name_,
                  metric=metric,
                  metric_title=metric_title,
                  plot_row_index=row,
                  plot_col_index=col,
                  is_difference=False,
                  models_names=models_names,
                  placeholder_model_name=placeholder_model_name)

        dataset_index += 1

    # Plot AUC Difference
    for j, dataset in enumerate(datasets):
        dataset_name_ = dataset['name']

        if dataset_name_ not in ModelFactory.DATASETS_NON_HOMOPHILITY:
            continue

        dataset_plot_name_ = dataset['plot_name']

        row, col = divmod(dataset_index, num_cols)

        make_plot(dataset_name=dataset_name_,
                  dataset_plot_name=dataset_plot_name_,
                  metric=metric,
                  metric_title=metric_title,
                  plot_row_index=row,
                  plot_col_index=col,
                  is_difference=True,
                  models_names=models_names,
                  placeholder_model_name=placeholder_model_name)

        dataset_index += 1

    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=math.ceil(len(models_names) / 2),
        fontsize=14,
        frameon=False,
        columnspacing=1.0,
        handlelength=1.5,
    )

    plt.savefig(f"{metric}_heterophilic_data_{models_type}_mae.png", bbox_inches='tight', dpi=300)