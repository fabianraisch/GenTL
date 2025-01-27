import torch.nn as nn
import torch

class LSTM(nn.Module):
    def __init__(
            self, 
            num_features, 
            hidden_size, 
            num_layers, 
            forecast_horizon=1, 
            dropout=0.0
        ):

        super().__init__()

        self.lstm = nn.LSTM(input_size=num_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)

        self.fc = nn.Linear(hidden_size, forecast_horizon)
    
    
    def forward(self, x):

        lstm_out, _ = self.lstm(x)

        return self.fc(lstm_out[:, -1, :])