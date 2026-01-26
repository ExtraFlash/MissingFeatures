import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from typing import Tuple


def preprocess_split(train_set, seed: int, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                                                                 pd.DataFrame]:
    """

    :param train_set:
    :param test_size:
    :return: X_train, y_train, X_val, y_val
    """
    # train_set = pd.read_csv(f"{path_to_dataset}/train/data.csv")
    # test_set = pd.read_csv(f"{path_to_dataset}/test/data.csv")
    if seed is not None:
        train, val = train_test_split(train_set, test_size=test_size, random_state=seed)
    else:
        train, val = train_test_split(train_set, test_size=test_size)
    # Scale train and val
    # TODO: perform scaling only on non one-hot encoded features
    non_categorical_columns = [col for col in train.columns[:-1] if train[col].nunique() > 2]
    categorical_columns = [col for col in train.columns[:-1] if col not in non_categorical_columns]

    real_positive_columns = [col for col in train.columns[:-1] if train[col].nunique() > 2 and (train[col] >= 0).all()]
    real_columns = [col for col in train.columns[:-1] if train[col].nunique() > 2 and not (train[col] >= 0).all()]

    # print(f'real columns: {real_columns}')
    # print(f'real_positive_columns: {real_positive_columns}')

    target_column = [train.columns[-1]]
    original_columns = train.columns

    scaler_pipeline = ColumnTransformer([
        ('minmax_scaler', MinMaxScaler(), real_positive_columns),
        ('standard_scaler', StandardScaler(), real_columns)
    ], remainder='passthrough')

    train = pd.DataFrame(scaler_pipeline.fit_transform(train),
                         columns=non_categorical_columns + categorical_columns + target_column)
    # keep the order of columns
    train = train[original_columns]

    val = pd.DataFrame(scaler_pipeline.transform(val),
                       columns=non_categorical_columns + categorical_columns + target_column)
    # keep the order of columns
    val = val[original_columns]

    # Split features and labels
    X_train, y_train = train.iloc[:, :-1], train.iloc[:, -1]
    X_val, y_val = val.iloc[:, :-1], val.iloc[:, -1]

    # test
    # target_column = [test_set.columns[-1]]
    # original_columns = test_set.columns
    # test = pd.DataFrame(scaler_pipeline.transform(test_set),
    #                     columns=non_categorical_columns + categorical_columns + target_column)
    # # keep the order of columns
    # test = test[original_columns]
    # X_test, y_test = test.iloc[:, :-1], test.iloc[:, -1]

    return X_train, y_train, X_val, y_val

