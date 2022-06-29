import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class rnn_classify(nn.Module):
    def __init__(self, input_size=28, hidden_size=100, num_layers=1, num_classes=2, model_type="lstm"):
        super(rnn_classify, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model_type = model_type

        if model_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        else:
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.reshape(-1,28,28)
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
#        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        if self.model_type == 'lstm':
            state_vec, _ = self.rnn(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        else:
            state_vec, _ = self.rnn(x, h0)
        # Decode the hidden state of the last time step
        out = self.fc(state_vec[:, -1, :])
        # dense = self.fc(state_vec)
        return state_vec, out


    
class NetList(torch.nn.Module):
    def __init__(self, list_of_models):
        super(NetList, self).__init__()
        self.models = torch.nn.ModuleList(list_of_models)
    
    def forward(self, x, idx=0):
        return self.models[idx](x)
