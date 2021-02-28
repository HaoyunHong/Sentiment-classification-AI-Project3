# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F

'''Convolutional Neural Networks for Sentence Classification'''


class CNN(nn.Module):
    def __init__(self, embed_dim, weights, filter_num, filter_sizes, dropout, tagset_size):
        super(CNN, self).__init__()
        self.embed_dim = embed_dim  # 词向量维度
        self.filter_num = filter_num  # 卷积核数量(channels数)
        self.tagset_size = tagset_size  # 标签种数
        self.filter_sizes = filter_sizes

        embed_matrix = torch.tensor(weights)
        self.embedding = nn.Embedding.from_pretrained(embed_matrix, freeze=True)

        # 只有一个embedding，只有一个通道
        self.convs = nn.ModuleList(
            [nn.Conv1d(1, self.filter_num, (k, self.embed_dim)) for k in self.filter_sizes])

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(len(filter_sizes) * filter_num, tagset_size)

    def forward(self, x):
        # 输入x的维度为(batch_size, max_len)
        x = self.embedding(x)  # 经过embedding,x的维度为(batch_size, max_len, embedding_dim)

        # 经过view函数x的维度变为(batch_size, input_chanel=1, w=max_len, h=embedding_dim)
        x = x.view(x.size(0), 1, x.size(1), self.embed_dim)

        # 经过卷积运算,x中每个运算结果维度为(batch_size, out_chanel, w, h=1)
        x = [F.relu(conv(x)) for conv in self.convs]

        # 经过最大池化层,维度变为(batch_size, out_chanel, w=1, h=1)
        x = [F.max_pool2d(input=x_item, kernel_size=(x_item.size(2), x_item.size(3))) for x_item in x]

        # 将不同卷积核运算结果维度（batch，out_chanel,w,h=1）展平为（batch, outchanel*w*h）
        x = [x_item.view(x_item.size(0), -1) for x_item in x]

        # 将不同卷积核提取的特征组合起来,维度变为(batch, sum:outchanel*w*h)
        x = torch.cat(x, 1)

        # dropout层，防止过拟合
        x = self.dropout(x)

        # 全连接层
        out = self.linear(x)

        # softmax变概率
        out = F.softmax(out, dim=1)
        return out
