import torch
import torch.nn as nn


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Reshape input tensor to match GRU input size
        x = x.view(x.size(0), x.size(1), 1)

        output, _ = self.gru(x)
        output = self.fc(output[:, -1, :])
        return output.squeeze()
