import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Reshape input tensor to match LSTM input size
        x = x.view(x.size(0), x.size(1), 1)

        output, (hidden, cell) = self.lstm(x)
        output = self.fc(hidden[-1])
        return output.squeeze()
