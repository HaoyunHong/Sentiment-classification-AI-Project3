import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout, tagset_size):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, tagset_size)

    def forward(self, x):
        out = self.linear1(x)
        out = self.dropout(out)
        out = self.linear2(F.relu(out))

        # softmax变概率
        out = F.softmax(out, dim=1)
        return out
