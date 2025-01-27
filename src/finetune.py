import torch
import copy

from src.models.naive_model import NaiveModel
from src.models.lstm import LSTM
from src.preprocessing.preprocessing import *


def finetune_model(
    config: dict,
    scalers,
    model,
    optimizer,
    grad_scaler,
    device: str = "cpu"

):

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using {device} to train the model.")

    # Override autocast if cuda is not available
    if str(device) != "cuda":
        print("Autocast is not supported without cuda. Set autocast config variables to false.")
        config["use_amp"] = False
        config["use_amp_outside_train"] = False

    train_data_loader, val_data_loader, test_data_loader, train_batch_sampler, scaler_features, scaler_target = preprocess_data(
        csv_file_dir=config["data_path"],
        finetune_split=config.get("finetune_split"),
        feature_columns=config["feature_cols"],
        target_columns=config["target_name"],
        dataframe_limit=config["dataframe_limit"],
        train_split=config.get("train_split"),
        val_split=config.get("val_split"),
        exclude=config.get("exclude"),
        lookback=config["lookback"],
        forecast_horizon=config["forecast_horizon"],
        batch_size=config["batch_size"],
        dataloader_shuffle=config["dataloader_shuffle"],
        scaler=scalers
    )

    model.to(device)

    naive_model = NaiveModel(forecast_horizon=config["forecast_horizon"])

    best_val = float("inf")
    best_val_errors = {}
    best_dict = {}

    # Train the model here
    for epoch in range(config["epochs"]):

        model.train()

        train_batch_sampler.shuffle()

        for x_batch, y_batch, _, _ in train_data_loader:

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # Train for the epoch

            with torch.autocast(device_type="cuda",dtype=torch.float16, enabled=config["use_amp"]):
                predicted = model(x_batch)
                loss = config["loss_function"](predicted, y_batch)

            optimizer.zero_grad()
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()


        # Evaluation for the epoch

        model.eval()
        with torch.no_grad():
             
            with torch.autocast(device_type="cuda", dtype=torch.float16,enabled=False):     # Do not use amp in the evaluation step.

                # Collect all the predictions and actuals

                train_predictions = torch.cat([model(X_batch.to(device)) for X_batch, _, _, _ in train_data_loader])
                train_actuals = torch.cat([y_batch.to(device) for _, y_batch, _, _ in train_data_loader])

                if len(val_data_loader):
                    val_predictions = torch.cat([model(X_batch.to(device)) for X_batch, _, _, _ in val_data_loader])
                    val_actuals = torch.cat([y_batch.to(device) for _, y_batch, _, _ in val_data_loader])

                # Inverse scaling

                train_actuals = scaler_target.inverse_transform(train_actuals.cpu().numpy().reshape(-1, 1)).flatten()
                train_predictions = scaler_target.inverse_transform(train_predictions.cpu().numpy().reshape(-1, 1)).flatten()

                if len(val_data_loader):
                    val_actuals = scaler_target.inverse_transform(val_actuals.cpu().numpy().reshape(-1, 1)).flatten()
                    val_predictions = scaler_target.inverse_transform(val_predictions.cpu().numpy().reshape(-1, 1)).flatten()

                # Convert back to tensors for loss calculation
                train_predictions_tensor = torch.tensor(train_predictions, dtype=torch.float32)
                train_actuals_tensor = torch.tensor(train_actuals, dtype=torch.float32)

                if len(val_data_loader):
                    val_predictions_tensor = torch.tensor(val_predictions, dtype=torch.float32)
                    val_actuals_tensor = torch.tensor(val_actuals, dtype=torch.float32)

                # Evaluate the Naive Model to calculate the MASE score.

                naive_train_predictions = torch.cat([naive_model(X_batch.to(device)) for X_batch, _, _, _ in train_data_loader])
                naive_train_actuals = torch.cat([y_batch.to(device) for _, y_batch, _, _ in train_data_loader])

                naive_train_actuals = scaler_target.inverse_transform(naive_train_actuals.cpu().numpy().reshape(-1, 1)).flatten() 
                naive_train_predictions = scaler_target.inverse_transform(naive_train_predictions.cpu().numpy().reshape(-1, 1)).flatten()

                naive_train_pred_tensor = torch.tensor(naive_train_predictions, dtype=torch.float32)
                naive_train_act_tensor = torch.tensor(naive_train_actuals, dtype=torch.float32)

                # For the val set also
                if len(val_data_loader):
                    naive_val_predictions = torch.cat([naive_model(X_batch.to(device)) for X_batch, _, _, _ in val_data_loader])
                    naive_val_actuals = torch.cat([y_batch.to(device) for _, y_batch, _, _ in val_data_loader])

                    naive_val_actuals = scaler_target.inverse_transform(naive_val_actuals.cpu().numpy().reshape(-1, 1)).flatten()
                    naive_val_predictions = scaler_target.inverse_transform(naive_val_predictions.cpu().numpy().reshape(-1, 1)).flatten()

                    naive_val_pred_tensor = torch.tensor(naive_val_predictions, dtype=torch.float32)
                    naive_val_act_tensor = torch.tensor(naive_val_actuals, dtype=torch.float32)

                
                mae = torch.nn.L1Loss()

                loss_function = config["loss_function"]

                # Calculate RMSE
                rmse_train = torch.sqrt(loss_function(train_predictions_tensor, train_actuals_tensor))
                if len(val_data_loader):
                    rmse_val = torch.sqrt(loss_function(val_predictions_tensor, val_actuals_tensor))

                # Calculate MAE
                mae_train = mae(train_predictions_tensor, train_actuals_tensor)
                if len(val_data_loader):
                    mae_val = mae(val_predictions_tensor, val_actuals_tensor)

                # Calcluate MASE
                naive_mae_train = mae(naive_train_pred_tensor, naive_train_act_tensor)
                mase_train = mae_train / naive_mae_train
                if len(val_data_loader): 
                    naive_mae_val = mae(naive_val_pred_tensor, naive_val_act_tensor)
                    mase_val = mae_val / naive_mae_val

                from datetime import datetime

                if len(val_data_loader):
                    print(f"{datetime.now().strftime('%H:%M:%S')} Epoch {epoch}: Train RMSE: {rmse_train.item()}; Val RMSE: {rmse_val.item()}")
                else:
                    print(f"{datetime.now().strftime('%H:%M:%S')} Epoch {epoch}: Train RMSE: {rmse_train.item()}")


                if (len(val_data_loader) and config["save_best_val"] and best_val > rmse_val):
                    print("--> RMSE decreased - model will be saved as best model.")
                    best_val = rmse_val
                    best_dict = {
                        "model": copy.deepcopy(model),
                        "optimizer": copy.deepcopy(optimizer),
                        "grad_scaler": copy.deepcopy(grad_scaler),
                        "config": copy.deepcopy(config)
                    }
                    best_val_errors = {
                        "rmse": rmse_val,
                        "mae": mae_val,
                        "mase": mase_val
                    }

    # Return the best model if it is saved.
    if best_dict: 
        return best_dict, test_data_loader, (scaler_features, scaler_target), best_val_errors

    return {
        "model": model,
        "optimizer": optimizer,
        "grad_scaler": grad_scaler,
        "config": config
    }, test_data_loader, (scaler_features, scaler_target), best_val_errors
