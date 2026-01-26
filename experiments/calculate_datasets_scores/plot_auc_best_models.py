import pandas as pd
from matplotlib import pyplot as plt
import json
import numpy as np
import math
from my_models import ModelFactory
from plots_utils import utils
from experiments.calculate_datasets_scores import plot_style


def get_auc(dataset_name, model_name):
    cvs = 5
    aucs = []
    for cv in range(cvs):
        data = pd.read_csv(f"data_percentage_100/{dataset_name}/{model_name}/cv_{cv}/auc.csv", index_col=0)
        auc = data['auc'].mean()
        aucs.append(auc)
    return np.mean(aucs)


def get_best_model_for_dataset(dataset_name):
    models_names = ModelFactory.MODELS  # Make sure num_features was first
    forbidden_models = [
        ModelFactory.DAE_NAME,
        ModelFactory.GAT_NAME,
        ModelFactory.GCN_NAME,
    ]

    models_names = [model for model in models_names if model not in forbidden_models]

    best_model = None
    best_auc = -math.inf
    for model_name in models_names:
        auc = get_auc(dataset_name=dataset_name,
                      model_name=model_name)
        if auc > best_auc:
            best_auc = auc
            best_model = model_name

    return best_model


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
    results['Best Model'] = {}

    # take the features amounts list from the first cv
    data = pd.read_csv(f"data_percentage_100/{dataset_name}/{placeholder_model_name}/cv_0/{metric}.csv")
    features_amounts = data["num_features"].to_list()
    total_features = features_amounts[0]
    features_amounts_filtered = [total_features] + [k for k in features_amounts[1:] if k % 5 == 1]

    for model_name in models_names:
        for features_amount in features_amounts_filtered:
            # scores for model with features over all the cvs
            scores = []
            for cv in range(cvs):
                data = pd.read_csv(f"data_percentage_100/{dataset_name}/{model_name}/cv_{cv}/{metric}.csv", index_col=0)
                score = data.loc[features_amount, 'auc']
                scores.append(score)
            # save scores in results
            results[model_name][features_amount] = scores

    # Do the same for the best model
    best_model_name = get_best_model_for_dataset(dataset_name=dataset_name)
    print(f'Best model for dataset {dataset_name} is {best_model_name}')
    for features_amount in features_amounts_filtered:
        # scores for model with features over all the cvs
        scores = []
        for cv in range(cvs):
            data = pd.read_csv(f"data_percentage_100/{dataset_name}/{best_model_name}/cv_{cv}/{metric}.csv", index_col=0)
            score = data.loc[features_amount, 'auc']
            scores.append(score)
        # save scores in results
        results['Best Model'][features_amount] = scores

    # make plot
    x_ticks = features_amounts[::(len(features_amounts) // 5) + 1]
    if 1 not in x_ticks:
        x_ticks.append(1)

    for idx, model_name in enumerate(models_names):
        # Calculate means and stds for both results and results_max
        means = np.array([np.mean(scores) for scores in results[model_name].values()])
        stds = np.array(
            [np.std(scores, ddof=1) / np.sqrt(np.size(scores)) for scores in results[model_name].values()])


        # Plot the difference
        ax.plot(features_amounts_filtered, means, label=ModelFactory.display_name(model_name),
                color=ModelFactory.color_for(model_name))
        # plot errors
        ax.fill_between(features_amounts_filtered, means + stds, means - stds,
                        color=ModelFactory.color_for(model_name), alpha=0.2)

    # Plot the best model
    # Calculate means and stds for both results and results_max
    means = np.array([np.mean(scores) for scores in results['Best Model'].values()])
    stds = np.array(
        [np.std(scores, ddof=1) / np.sqrt(np.size(scores)) for scores in results['Best Model'].values()])

    # Plot the difference
    ax.plot(features_amounts_filtered, means, label='Best Model',
            color="green")
    # plot errors
    ax.fill_between(features_amounts_filtered, means + stds, means - stds,
                    color="green", alpha=0.2)



        # means = np.array([np.mean(scores) for scores in results_max[model_name].values()])
        # stds = np.array(
        #     [np.std(scores, ddof=1) / np.sqrt(np.size(scores)) for scores in results_max[model_name].values()])
        # # plot the mean scores
        # ax.plot(feautres_amounts_filtered, means, label=model_name,
        #         color=colors[idx % len(colors)])
        # # plot errors
        # ax.fill_between(feautres_amounts_filtered, means + stds, means - stds,
        #                 color=colors[idx % len(colors)], alpha=0.2)

        #
        # Plot results (solid line)
        # means = np.array([np.mean(scores) for scores in results[model_name].values()])
        # stds = np.array(
        #     [np.std(scores, ddof=1) / np.sqrt(np.size(scores)) for scores in results[model_name].values()])
        # # plot the mean scores with solid line
        # ax.plot(feautres_amounts_filtered, means, label=model_name,
        #         color=colors[idx % len(colors)])
        # # plot errors
        # ax.fill_between(feautres_amounts_filtered, means + stds, means - stds,
        #                 color=colors[idx % len(colors)], alpha=0.2)
        #
        # # Plot results_max (dotted line)
        # means_max = np.array([np.mean(scores) for scores in results_max[model_name].values()])
        # stds_max = np.array(
        #     [np.std(scores, ddof=1) / np.sqrt(np.size(scores)) for scores in results_max[model_name].values()])
        # # plot the mean scores with dotted line
        # ax.plot(feautres_amounts_filtered, means_max, linestyle='--',
        #         color=colors[idx % len(colors)])
        # # plot errors
        # ax.fill_between(feautres_amounts_filtered, means_max + stds_max, means_max - stds_max,
        #                 color=colors[idx % len(colors)], alpha=0.1)  # lower alpha for dotted line





    ax.set_xlabel("Num features")
    if plot_col_index == 0:
        ax.set_ylabel(metric_title, rotation=90)
    ax.set_xticks(x_ticks)
    # ax = plt.gca()
    ax.invert_xaxis()
    ax.grid()
    # Set y-axis limits
    ax.set_ylim(0.4, 1.0)
    if plot_row_index == 0 and plot_col_index == 0:
        # ax.legend()
        global handles, labels
        handles, labels = ax.get_legend_handles_labels()
    ax.set_title(dataset_plot_name)


if __name__ == '__main__':
    plot_style.set_plot_style()

    handles, labels = None, None
    # models_names = MODELS = [
    #     ModelFactory.Mean_Gradient_Boosting_Classifier_Name,
    #     # DAE_NAME,
    #     # ModelFactory.GAT_NAME,
    #     ModelFactory.GCN_NAME,
    #     ModelFactory.Denoising_Graph_Encoder_Name,
    #     ModelFactory.Complete_Random_Forest_NAME,
    #     ModelFactory.Complete_Gradient_Boosting_Classifier_Name,
    #     ModelFactory.XGB_NAME,
    #     # DAE_Dynamic_TYPE_LIGHTNING_NAME,
    #     # Neural_Network_NAME,
    #     # Teacher_Students_NAME,
    #     ModelFactory.Random_Forest_NAME,
    #     # ModelFactory.Ada_Boost_NAME,
    #     ModelFactory.LGBM_NAME,
    #     # Logistic_Regression_NAME
    # ]

    placeholder_model_name = ModelFactory.MODELS[0]

    auc_file = pd.read_csv(f"data_percentage_100/Tokyo/{placeholder_model_name}/cv_0/auc_available.csv")

    models_names = [
        ModelFactory.DAE_NAME,
        ModelFactory.GAT_NAME,
        ModelFactory.GCN_NAME,
    ]  # Make sure num_features was first

    metric = 'auc'
    metric_title = 'AUC'

    config_path = "../../datasets/config.json"

    # get datasets config
    with open(config_path) as f:
        config = json.load(f)
    # get datasets from config
    datasets: list = config['datasets']  # Not include the last 2 datasets for now

    num_datasets = len(datasets)
    num_cols = 5  # Define columns dynamically based on how you want the layout to look
    num_rows = math.ceil(num_datasets / num_cols)  # Calculate the required number of rows

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

        row, col = divmod(j, num_cols)  # Compute row and col position dynamically

        make_plot(dataset_name=dataset_name_,
                  dataset_plot_name=dataset_plot_name_,
                  metric=metric,
                  metric_title=metric_title,
                  plot_row_index=row,
                  plot_col_index=col)

    # Hide empty subplots if the number of datasets is not a perfect fit
    for j in range(num_datasets, num_rows * num_cols):
        row, col = divmod(j, num_cols)
        fig.delaxes(axs[row, col])

    # Place legend outside the subplots
    # fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=len(models_names), fontsize=10)

    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        bbox_to_anchor=(0.5, 1.09),  # Shift legend upward (increase value to move higher)
        ncol=math.ceil(len(models_names) / 2),
        fontsize=14,
        frameon=False,
        columnspacing=1.0,
        handlelength=1.5,
    )

    # axs[1, 2].set_visible(False)
    plt.savefig(f"auc_best_models.png", bbox_inches='tight', dpi=300)
