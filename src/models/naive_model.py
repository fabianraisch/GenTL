import torch.nn as nn
import torch

class NaiveModel(nn.Module):
    '''
    Naive Model that just returns the last seen target variable
    '''
    def __init__(self, forecast_horizon=1):
        super().__init__()

        self.forecast_horizon = forecast_horizon
        
        # This is just mock such that the model does not crash in some modes.
        self.fc = nn.Linear(8, 1)
    
    def forward(self, x):

        res = x[:, -1, -1].unsqueeze(1).repeat(1, self.forecast_horizon)
        
        return res