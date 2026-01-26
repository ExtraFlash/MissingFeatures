import pandas as pd
from matplotlib import pyplot as plt
import json
import numpy as np
import math
from my_models import ModelFactory
from plots_utils import utils
from sklearn.linear_model import LinearRegression


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


# def add_point_to_scatter(dataset_name: str, dataset_plot_name: str, metric: str):
#     # First subplot
#     # plt.subplot(2, 3, plot_index)
#     # plt.clf()
#     utils.make_style(plt)
#     # get data
#     cvs = 5
#
#     # colors = plt.cm.tab20(np.linspace(0, 1, len(models_names)))  # Use tab20 for unique colors
#     # line_styles = ['-', '--', '-.', ':']  # Define different line styles
#
#     # {'model': {50: scores, 49, scores, ...}, 'model': ...}
#     results = {model_name: {} for model_name in models_names}
#     results['Best Model'] = {}
#
#     # take the features amounts list from the first cv
#     data = pd.read_csv(f"data_percentage_100/{dataset_name}/{placeholder_model_name}/cv_0/{metric}.csv")
#     features_amounts = data["num_features"].to_list()
#     total_features = features_amounts[0]
#     features_amounts_filtered = [total_features] + [k for k in features_amounts[1:] if k % 5 == 1]
#
#     for model_name in models_names:
#         for features_amount in features_amounts_filtered:
#             # scores for model with features over all the cvs
#             scores = []
#             for cv in range(cvs):
#                 data = pd.read_csv(f"data_percentage_100/{dataset_name}/{model_name}/cv_{cv}/{metric}.csv", index_col=0)
#                 score = data.loc[features_amount, 'auc']
#                 scores.append(score)
#             # save scores in results
#             results[model_name][features_amount] = scores
#
#     # Do the same for the best model
#     best_model_name = get_best_model_for_dataset(dataset_name=dataset_name)
#     print(f'Best model for dataset {dataset_name} is {best_model_name}')
#     for features_amount in features_amounts_filtered:
#         # scores for model with features over all the cvs
#         scores = []
#         for cv in range(cvs):
#             data = pd.read_csv(f"data_percentage_100/{dataset_name}/{best_model_name}/cv_{cv}/{metric}.csv",
#                                index_col=0)
#             score = data.loc[features_amount, 'auc']
#             scores.append(score)
#         # save scores in results
#         results['Best Model'][features_amount] = scores
#
#     difference = 0
#     for model_name in models_names:
#         for features_amount in features_amounts_filtered:
#             model_scores = np.array(results[model_name][features_amount])
#             best_model_scores = np.array(results['Best Model'][features_amount])
#             diff = best_model_scores - model_scores
#             difference += np.mean(diff)
#     difference = difference / (len(models_names) * len(features_amounts_filtered))
#     homophility_score = datasets_to_homophilities[dataset_name]
#
#     differences_in_auc.append(difference)
#     homophility_scores.append(homophility_score)


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
    datasets_names = [ds['name'] for ds in datasets]

    num_datasets = len(datasets)

    # Create scatter plot
    plt.figure(figsize=(8, 6))

    homophility_df = pd.read_csv('../check_homophily/homophily_scores.csv', index_col=0)

    datasets_to_homophilities = homophility_df.to_dict()['scores']

    models = [
        ModelFactory.DAE_NAME,
        ModelFactory.GAT_NAME,
        ModelFactory.GCN_NAME,
    ]

    # Styling per model
    markers = {
        ModelFactory.DAE_NAME: 'o',
        ModelFactory.GAT_NAME: 's',
        ModelFactory.GCN_NAME: '^',
        'Best Model': 'D'
    }

    # small horizontal jitter so points at the same x don't overlap
    offsets = {
        ModelFactory.DAE_NAME: -0.015,
        ModelFactory.GAT_NAME: -0.005,
        ModelFactory.GCN_NAME: 0.005,
    }

    plt.figure(figsize=(9, 6))

    for m in models:
        x_points = []
        y_points = []
        xs, ys, labels = [], [], []
        for ds in datasets:
            dataset_name = ds['name']

            utils.make_style(plt)
            # get data
            cvs = 5

            results_model = {}
            results_best_model = {}

            # take the features amounts list from the first cv
            data = pd.read_csv(f"data_percentage_100/{dataset_name}/{placeholder_model_name}/cv_0/{metric}.csv")
            features_amounts = data["num_features"].to_list()
            total_features = features_amounts[0]
            features_amounts_filtered = [total_features] + [k for k in features_amounts[1:] if k % 5 == 1]

            for features_amount in features_amounts_filtered:
                # scores for model with features over all the cvs
                scores = []
                for cv in range(cvs):
                    data = pd.read_csv(f"data_percentage_100/{dataset_name}/{m}/cv_{cv}/{metric}.csv",
                                       index_col=0)
                    score = data.loc[features_amount, 'auc']
                    scores.append(score)
                # save scores in results
                results_model[features_amount] = scores

            # Do the same for the best model
            best_model_name = get_best_model_for_dataset(dataset_name=dataset_name)
            print(f'Best model for dataset {dataset_name} is {best_model_name}')
            for features_amount in features_amounts_filtered:
                # scores for model with features over all the cvs
                scores = []
                for cv in range(cvs):
                    data = pd.read_csv(f"data_percentage_100/{dataset_name}/{best_model_name}/cv_{cv}/{metric}.csv",
                                       index_col=0)
                    score = data.loc[features_amount, 'auc']
                    scores.append(score)
                # save scores in results
                results_best_model[features_amount] = scores

            difference = 0
            for features_amount in features_amounts_filtered:
                model_scores = np.array(results_model[features_amount])
                best_model_scores = np.array(results_best_model[features_amount])
                diff = best_model_scores - model_scores
                difference += np.mean(diff)

            difference = difference / (len(features_amounts_filtered))

            homophility_score = datasets_to_homophilities[dataset_name]

            x_points.append(homophility_score)
            y_points.append(difference)

            xs.append(homophility_score + offsets[m])  # apply small jitter
            ys.append(difference)
            labels.append(ModelFactory.display_name(m))

        color = ModelFactory.color_for(m) if m in models_names else "green"

        label = ModelFactory.display_name(m) if m in models_names else "Best Model"

        # one scatter call per model
        plt.scatter(
            xs, ys,
            s=80,
            alpha=0.85,
            marker=markers[m],
            edgecolors='k',
            label=label,
            color=color
        )

        datasets_names = [dataset['name'] for dataset in datasets]
        for i in range(len(datasets_names)):
            if y_points[i] > 0.2:
                print(f'{datasets_names[i]}: {y_points[i]}')

        x_arr = np.array(x_points).reshape(-1, 1)
        y_arr = np.array(y_points)
        model = LinearRegression()
        model.fit(x_arr, y_arr)

        # Predicted line
        x_range = np.linspace(min(x_points), max(x_points), 100).reshape(-1, 1)
        y_pred = model.predict(x_range)
        plt.plot(x_range, y_pred, color=color, linewidth=2, linestyle='--')

        # print intercept and slope
        print(f'Model: {m}, Intercept: {model.intercept_}, Slope: {model.coef_[0]}')


    plt.xlabel('Homophily Score', fontsize=14)
    plt.ylabel(f'Difference in {metric_title} Between Models', fontsize=14)
    plt.title(f'{metric_title} Difference vs. Homophily', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title='Model', frameon=True)

    # # Optionally label points
    # for i, dataset in enumerate(datasets):
    #     plt.text(
    #         x_points[i] + 0.01,
    #         y_points[i],
    #         datasets[i]['plot_name'],
    #         fontsize=9
    #     )

    plt.tight_layout()
    plt.savefig("scatter_plot_homophility_vs_difference_in_auc_between_models.png",
                bbox_inches='tight', dpi=300)
    plt.show()
