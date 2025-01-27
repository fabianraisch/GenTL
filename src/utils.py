import os
import json
import random
import torch
import joblib
import torch.nn as nn
import torch.optim as optim
from src.preprocessing.preprocessing import *
from src.models.lstm import LSTM


def load_config_file(path: str):

    with open(path) as json_file:
        data = json.load(json_file)

    data["optimizer"] = optim.Adam

    data["loss_function"] = nn.MSELoss()

    return data


def get_file_name(path: str):

    base_name = os.path.basename(path)

    return base_name


def sample_from_directory(path: str, num_of_elements: int, random_select: bool):
    if num_of_elements == 0:
        return []
    
    data = []
    for (root, _, files) in os.walk(path, topdown=True):
        for file in files:
            if file.startswith('_'):  # assuming this is the condition to select CSV files
                path_to_csv = os.path.join(root, file)
                data.append(path_to_csv)
    
    # If random_select is True, sample randomly
    if random_select:
        try:
            samples = random.sample(data, num_of_elements)
        except ValueError as e:
            print(f"ValueError while sampling training data: {e}")
            samples = []  # In case of an error, return an empty list
    else:
        # If random_select is False, take the first `num_of_elements` in order
        samples = data[:num_of_elements]

    return samples


def load_gentl_model(config_gentl: dict):

    gentl_model_path = os.path.join("models", "gentl_model.pt")

    feature_scaler_path = os.path.join("models", "scaler", "gentl_feature_scaler.gz")
    target_scaler_path = os.path.join("models", "scaler", "gentl_target_scaler.gz")


    # Load the model

    gentl_state_dict = torch.load(gentl_model_path)

    model = LSTM(
        num_features=len(config_gentl["feature_cols"]),
        hidden_size=config_gentl["hidden_size"],
        num_layers=config_gentl["num_layers"],
        forecast_horizon=config_gentl["forecast_horizon"]
    )

    model.load_state_dict(gentl_state_dict["model"])

    optimizer = config_gentl["optimizer"](model.parameters(), lr=config_gentl["learning_rate"])

    grad_scaler = torch.cuda.amp.GradScaler(enabled=True)

    # Load scalers

    feature_scaler = joblib.load(feature_scaler_path)

    target_scaler = joblib.load(target_scaler_path)

    scalers = (feature_scaler, target_scaler)

    return model, optimizer, grad_scaler, scalers

    