import pandas as pd
from matplotlib import pyplot as plt
import json
import numpy as np
import math
from my_models import ModelFactory
from plots_utils import utils


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




# def make_plot(dataset_name: str, dataset_plot_name: str, metric: str, metric_title: str,
#               plot_row_index: int, plot_col_index: int):
#     # First subplot
#     # plt.subplot(2, 3, plot_index)
#     # plt.clf()
#     utils.make_style(plt)
#     ax = axs[plot_row_index, plot_col_index]
#     # get data
#     cvs = 5
#
#     colors = plt.cm.tab20(np.linspace(0, 1, len(models_names)))  # Use tab20 for unique colors
#     # line_styles = ['-', '--', '-.', ':']  # Define different line styles
#
#     # {'model': {50: scores, 49, scores, ...}, 'model': ...}
#     results = {model_name: {} for model_name in models_names}
#     # take the features amounts list from the first cv
#     data = pd.read_csv(f"data_percentage_100/{dataset_name}/{placeholder_model_name}/cv_0/{metric}.csv")
#     features_amounts = data["num_features"].to_list()
#     feautres_amounts_filtered = [feature_amount for feature_amount in features_amounts if feature_amount % 5 == 1]
#
#     for model_name in models_names:
#         for features_amount in feautres_amounts_filtered:
#             # scores for model with features over all the cvs
#             scores = []
#             for cv in range(cvs):
#                 data = pd.read_csv(f"data_percentage_100/{dataset_name}/{model_name}/cv_{cv}/{metric}.csv", index_col=0)
#                 score = data.loc[features_amount, 'auc']
#                 scores.append(score)
#             # save scores in results
#             results[model_name][features_amount] = scores
#
#     # make plot
#     x_ticks = features_amounts[::(len(features_amounts) // 5) + 1]
#     if 1 not in x_ticks:
#         x_ticks.append(1)
#
#     for idx, model_name in enumerate(models_names):
#         # Calculate means and stds for both results and results_max
#         means = np.array([np.mean(scores) for scores in results[model_name].values()])
#         stds = np.array(
#             [np.std(scores, ddof=1) / np.sqrt(np.size(scores)) for scores in results[model_name].values()])
#
#
#         # Plot the difference
#         ax.plot(feautres_amounts_filtered, means, label=ModelFactory.display_name(model_name),
#                 color=ModelFactory.color_for(model_name))
#         # plot errors
#         ax.fill_between(feautres_amounts_filtered, means + stds, means - stds,
#                         color=ModelFactory.color_for(model_name), alpha=0.2)
#         # means = np.array([np.mean(scores) for scores in results_max[model_name].values()])
#         # stds = np.array(
#         #     [np.std(scores, ddof=1) / np.sqrt(np.size(scores)) for scores in results_max[model_name].values()])
#         # # plot the mean scores
#         # ax.plot(feautres_amounts_filtered, means, label=model_name,
#         #         color=colors[idx % len(colors)])
#         # # plot errors
#         # ax.fill_between(feautres_amounts_filtered, means + stds, means - stds,
#         #                 color=colors[idx % len(colors)], alpha=0.2)
#
#         #
#         # Plot results (solid line)
#         # means = np.array([np.mean(scores) for scores in results[model_name].values()])
#         # stds = np.array(
#         #     [np.std(scores, ddof=1) / np.sqrt(np.size(scores)) for scores in results[model_name].values()])
#         # # plot the mean scores with solid line
#         # ax.plot(feautres_amounts_filtered, means, label=model_name,
#         #         color=colors[idx % len(colors)])
#         # # plot errors
#         # ax.fill_between(feautres_amounts_filtered, means + stds, means - stds,
#         #                 color=colors[idx % len(colors)], alpha=0.2)
#         #
#         # # Plot results_max (dotted line)
#         # means_max = np.array([np.mean(scores) for scores in results_max[model_name].values()])
#         # stds_max = np.array(
#         #     [np.std(scores, ddof=1) / np.sqrt(np.size(scores)) for scores in results_max[model_name].values()])
#         # # plot the mean scores with dotted line
#         # ax.plot(feautres_amounts_filtered, means_max, linestyle='--',
#         #         color=colors[idx % len(colors)])
#         # # plot errors
#         # ax.fill_between(feautres_amounts_filtered, means_max + stds_max, means_max - stds_max,
#         #                 color=colors[idx % len(colors)], alpha=0.1)  # lower alpha for dotted line
#
#



    # ax.set_xlabel("Num features")
    # if plot_col_index == 0:
    #     ax.set_ylabel(metric_title, rotation=90)
    # ax.set_xticks(x_ticks)
    # # ax = plt.gca()
    # ax.invert_xaxis()
    # ax.grid()
    # # Set y-axis limits
    # ax.set_ylim(0.4, 1.0)
    # if plot_row_index == 0 and plot_col_index == 0:
    #     # ax.legend()
    #     global handles, labels
    #     handles, labels = ax.get_legend_handles_labels()
    # ax.set_title(dataset_plot_name)


if __name__ == '__main__':
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

    models_names = ModelFactory.MODELS  # Make sure num_features was first

    metric = 'auc'
    metric_title = 'AUC'

    config_path = "../../datasets/config.json"

    # get datasets config
    with open(config_path) as f:
        config = json.load(f)
    # get datasets from config
    datasets: list = config['datasets']  # Not include the last 2 datasets for now

    models_auc = {}

    for m, model_name in enumerate(models_names):
        aucs = []
        for j, dataset in enumerate(datasets):
            dataset_name_ = dataset['name']
            dataset_plot_name_ = dataset['plot_name']
            relative_path_ = dataset['relative_path']
            label_position_ = dataset['label_position']
            has_header_ = dataset['has_header']
            has_id_ = dataset['has_id']
            is_multy_class_ = dataset['is_multy_class']

            auc = get_auc(dataset_name=dataset_name_,
                          model_name=model_name)
            aucs.append(auc)

        models_auc[model_name] = np.mean(aucs)


    # sort models_auc by the values
    models_auc = dict(sorted(models_auc.items(), key=lambda item: item[1], reverse=True))

    # print the results in order
    print("Models AUC results:")
    for model_name, auc in models_auc.items():
        print(f"{model_name}: {auc:.4f}")

