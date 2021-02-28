import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, embed_size, num_hiddens, embed_matrix, tagset_size, dropout):
        super(LSTM, self).__init__()
        self.num_hiddens = num_hiddens
        self.tagset_size = tagset_size  # 标签种数
        self.embedding = nn.Embedding.from_pretrained(embed_matrix, freeze=True)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=self.num_hiddens,
                            num_layers=1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_hiddens * 2, tagset_size)

    def forward(self, batch):
        emb_text = self.embedding(batch)
        states, hidden = self.lstm(emb_text.permute([1, 0, 2]))
        x = torch.cat([states[0], states[-1]], dim=1)
        # dropout层，防止过拟合
        outputs = self.dropout(x)
        outputs = F.relu(self.fc(outputs))

        # softmax变概率
        outputs = F.softmax(outputs, dim=1)
        return outputs
