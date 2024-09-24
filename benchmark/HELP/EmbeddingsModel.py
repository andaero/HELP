import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingsModel(nn.Module):
    def __init__(self, input_dim, output_dim, dropout):
        super(EmbeddingsModel, self).__init__()
        self.hidden1 = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.hidden2 = nn.Linear(output_dim, output_dim)
        # self.hidden3 = nn.Linear(250, output_dim)



    def forward(self, x):
        # x = F.relu(self.hidden1(x))
        x = self.hidden1(x)
        x = self.dropout(x)
        x = self.hidden2(x)

        return x
