import os.path
import pickle
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from models.denoising_auto_encoder import DAE
from models.dynamic_type_dae import DynamicTypeDAE
from models.teacher_students import TeacherStudents
from models.neural_network import NeuralNetworkModel
from models.dae_lightning import DynamicTypeDAEModel

from models import ActivationFactory

Neural_Network_NAME = "NeuralNetwork"
Teacher_Students_NAME = "TeacherStudents"
DAE_NAME = "DAE"
DAE_Dynamic_TYPE_NAME = "DAE_Dynamic_TYPE"
DAE_Dynamic_TYPE_LIGHTNING_NAME = "DAE_Dynamic_TYPE_LIGHTNING"
Random_Forest_NAME = "RandomForest"
Ada_Boost_NAME = "AdaBoost"
LGBM_NAME = "LGBM"
XGB_NAME = "XGB"
Logistic_Regression_NAME = "LogisticRegression"

MODELS = [DAE_Dynamic_TYPE_LIGHTNING_NAME, DAE_NAME, Neural_Network_NAME, Teacher_Students_NAME,
          Random_Forest_NAME, Ada_Boost_NAME, LGBM_NAME, XGB_NAME, Logistic_Regression_NAME]
# MODELS = ["DAE"]

dims_dae = {
    'Tokyo': 15,
    'Connectionist Bench': 15,
    'Ionosphere': 35,
    'Pima Indians Diabetes Database': 20,
    'Heart Disease': 22,
    'Statlog (German Credit Data)': 30,
}


def get_model(model_name: str, input_size: int, checkpoint_dir: str = None, dataset_name: str = None, **kwargs):
    """
    Get the model based on the model name
    :param model_name: Model name
    :param input_size: Input size
    :param latent_dim: Latent dimension for training DAE
    :param checkpoint_dir: checkpoint directory to load the model
    :param kwargs: default hyperparameters (each model should refer to its own hyperparameters).
                   if the hyperparameter is not provided, it will use the default values from the package.
    :return:
    """

    # To use this function, provide the model_name, input_size.
    # For the DAE provide the latent_dim (if not it will use a default value)
    # the kwargs are hyperparameters to use in the model, if not provided it will use the default values in each model.
    # if checkpoint_dir is provided, it will load the model from the checkpoint directory
    # (For neural networks, provide kwargs of the optimal hyperparameters used in the checkpoint directory)

    model = None
    loaded = False

    if model_name == Random_Forest_NAME:
        if 'max_depth' not in kwargs:
            kwargs['max_depth'] = 6
        n_estimators = kwargs.get('n_estimators', 100)
        max_depth = kwargs.get('max_depth', None)
        min_samples_split = kwargs.get('min_samples_split', 2)
        min_samples_leaf = kwargs.get('min_samples_leaf', 1)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                       min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
        if os.path.exists(checkpoint_dir):
            model_path = os.path.join(checkpoint_dir, 'model' + ".pkl")
            with open(model_path, 'rb') as file:
                model = pickle.load(file)
                loaded = True

    elif model_name == Ada_Boost_NAME:
        model = AdaBoostClassifier()

    elif model_name == LGBM_NAME:
        if 'max_depth' not in kwargs:
            kwargs['max_depth'] = 6
        model = LGBMClassifier(max_depth=kwargs['max_depth'])

    elif model_name == XGB_NAME:
        model = XGBClassifier()  # automatically with 6

    elif model_name == Logistic_Regression_NAME:
        model = LogisticRegression()

    elif model_name == DAE_NAME:
        latent_dim = kwargs.get('latent_dim', 10)
        encoder_units = kwargs.get('encoder_units', (128, 64))
        decoder_units = kwargs.get('decoder_units', (64, 128))
        activation_name = kwargs.get('activation_name', ActivationFactory.leaky_relu_NAME)
        dropout_rate = kwargs.get('dropout_rate', 0.0)
        learning_rate = kwargs.get('learning_rate', 1e-3)
        model = DAE(input_size=input_size, latent_dim=latent_dim,
                    encoder_units=encoder_units, decoder_units=decoder_units,
                    activation_name=activation_name, dropout_rate=dropout_rate, learning_rate=learning_rate)
        if checkpoint_dir and os.path.exists(checkpoint_dir):
            print(f"Loading DAE from {checkpoint_dir}")
            with open(f"{checkpoint_dir}/best_params.json", "r") as file:
                best_params = json.load(file)
            model = DAE(input_size=input_size,
                        latent_dim=best_params["latent_dim"],
                        encoder_units=(best_params["encoder_units_0"], best_params["encoder_units_1"]),
                        decoder_units=(best_params["decoder_units_0"], best_params["decoder_units_1"]),
                        activation_name=ActivationFactory.leaky_relu_NAME,
                        dropout_rate=best_params["dropout_rate"],
                        learning_rate=best_params["learning_rate"]
                        )
            loaded = model.load_checkpoint(checkpoint_dir)

    elif model_name == DAE_Dynamic_TYPE_NAME:
        latent_dim = kwargs.get('latent_dim', 10)
        types_list = kwargs.get('types_list', None)
        encoder_units = kwargs.get('encoder_units', (128, 64))
        decoder_units = kwargs.get('decoder_units', (64, 128))
        activation_name = kwargs.get('activation_name', ActivationFactory.relu_NAME)
        dropout_rate = kwargs.get('dropout_rate', 0.0)
        learning_rate = kwargs.get('learning_rate', 1e-3)
        # categorical_size = kwargs.get('categorical_size', None)
        model = DynamicTypeDAE(input_size=input_size, latent_dim=latent_dim,
                               types_list=types_list,
                               encoder_units=encoder_units, decoder_units=decoder_units,
                               activation_name=activation_name, dropout_rate=dropout_rate, learning_rate=learning_rate)
        if checkpoint_dir and os.path.exists(checkpoint_dir):
            print(f"Loading {DAE_Dynamic_TYPE_NAME} from {checkpoint_dir}")
            with open(f"{checkpoint_dir}/best_params.json", "r") as file:
                best_params = json.load(file)
            model = DynamicTypeDAE(input_size=input_size,
                                   types_list=types_list,
                                   latent_dim=best_params["latent_dim"],
                                   encoder_units=(best_params["encoder_units_0"], best_params["encoder_units_1"]),
                                   decoder_units=(best_params["decoder_units_0"], best_params["decoder_units_1"]),
                                   activation_name=ActivationFactory.leaky_relu_NAME,
                                   dropout_rate=best_params["dropout_rate"],
                                   learning_rate=best_params["learning_rate"]
                                   )
            loaded = model.load_checkpoint(checkpoint_dir)

    elif model_name == DAE_Dynamic_TYPE_LIGHTNING_NAME:
        latent_dim = kwargs.get('latent_dim', 10)
        types_list = kwargs.get('types_list', None)
        encoder_units = kwargs.get('encoder_units', (128, 64))
        decoder_units = kwargs.get('decoder_units', (64, 128))
        activation_name = kwargs.get('activation_name', ActivationFactory.relu_NAME)
        dropout_rate = kwargs.get('dropout_rate', 0.0)
        learning_rate = kwargs.get('learning_rate', 1e-3)

        model = DynamicTypeDAEModel(input_size=input_size,
                                    types_list=types_list,
                                    latent_dim=latent_dim,
                                    encoder_units=encoder_units,
                                    decoder_units=decoder_units,
                                    activation_name=activation_name,
                                    dropout_rate=dropout_rate,
                                    learning_rate=learning_rate)

        loaded = False
        if checkpoint_dir and os.path.exists(checkpoint_dir):
            print(f"Loading DAE from {checkpoint_dir}")
            model.load_checkpoint(checkpoint_dir)
            loaded = True


    elif model_name == Teacher_Students_NAME:
        model = TeacherStudents(input_size=input_size)

    elif model_name == Neural_Network_NAME:
        model = NeuralNetworkModel(input_size=input_size)

    return model, loaded


def is_masked_model(model_name: str):
    """ Check if the model is a masked model, i.e., in the set of masked models"""
    return model_name in {DAE_NAME, DAE_Dynamic_TYPE_NAME}


def is_lightning_model(model_name: str):
    return model_name == DAE_Dynamic_TYPE_LIGHTNING_NAME
