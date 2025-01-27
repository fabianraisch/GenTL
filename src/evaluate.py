import torch

from src.models.naive_model import NaiveModel


def evaluate_model(
        model_dict,
        test_dataloader,
        scaler,
        device="cpu"
):
    
    model = model_dict["model"]
    model_config = model_dict["config"]

    target_scaler = scaler[1]

    naive_model = NaiveModel(model_config["forecast_horizon"])

    if not len(test_dataloader):
        print("Test set was empty --> Skip Test evaluation.")
        exit()


    with torch.autocast(device_type="cuda", dtype=torch.float16,enabled=False):
        test_predictions = torch.cat([model(X_batch.to(device)) for X_batch, _, _, _ in test_dataloader])
        test_actuals = torch.cat([y_batch.to(device) for _, y_batch, _, _ in test_dataloader])

    test_actuals = target_scaler.inverse_transform(test_actuals.cpu().numpy().reshape(-1, 1)).flatten()
    test_predictions = target_scaler.inverse_transform(test_predictions.detach().cpu().numpy().reshape(-1, 1)).flatten()

    test_predictions_tensor = torch.tensor(test_predictions, dtype=torch.float32)
    test_actuals_tensor = torch.tensor(test_actuals, dtype=torch.float32)

    # For the test
    naive_test_predictions = torch.cat([naive_model(X_batch.to(device)) for X_batch, _, _, _ in test_dataloader])
    naive_test_actuals = torch.cat([y_batch.to(device) for _, y_batch, _, _ in test_dataloader])

    naive_test_actuals = target_scaler.inverse_transform(naive_test_actuals.cpu().numpy().reshape(-1, 1)).flatten()
    naive_test_predictions = target_scaler.inverse_transform(naive_test_predictions.cpu().numpy().reshape(-1, 1)).flatten()

    naive_test_pred_tensor = torch.tensor(naive_test_predictions, dtype=torch.float32)
    naive_test_act_tensor = torch.tensor(naive_test_actuals, dtype=torch.float32)


    # Calculate metrics

    loss_function = model_config["loss_function"]
        
    rmse_test = torch.sqrt(loss_function(test_predictions_tensor, test_actuals_tensor))

    mae = torch.nn.L1Loss()
    mae_test = mae(test_predictions_tensor, test_actuals_tensor)

    naive_mae_test = mae(naive_test_pred_tensor, naive_test_act_tensor)
    mase_test = mae_test / naive_mae_test

    return rmse_test, mase_test, mase_test, test_actuals_tensor, test_predictions_tensor


    