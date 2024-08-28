import torch.nn as nn

relu_NAME = "ReLU"
leaky_relu_NAME = "LeakyReLU"
tanh_NAME = "Tanh"


def get_activation(activation_name: str):
    if activation_name == relu_NAME:
        return nn.ReLU
    elif activation_name == leaky_relu_NAME:
        return nn.LeakyReLU
    elif activation_name == tanh_NAME:
        return nn.Tanh
    else:
        raise ValueError(f"Activation {activation_name} not supported")
