from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

MODELS = ["RandomForest", "AdaBoost", "LGBM", "XGB", "SVM", "LogisticRegression"]


def get_model(model_name: str):
    model = None
    if model_name == "RandomForest":
        model = RandomForestClassifier()
    elif model_name == "AdaBoost":
        model = AdaBoostClassifier()
    elif model_name == "LGBM":
        model = LGBMClassifier()
    elif model_name == "XGB":
        model = XGBClassifier()
    elif model_name == "SVM":
        model = SVC()
    elif model_name == "LogisticRegression":
        model = LogisticRegression()
    return model
