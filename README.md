# Read Me

洪昊昀    计82    2017010591

## 1. 运行环境

python 版本：3.8.2

pytorch 版本：1.5.0+cu101，但是因为显卡内存太小，所以都只能用CPU训练

运行环境：Win 10，Pycharm Community Edition 2019.3.4

## 2. 文件结构

其中`checkpoints`，`data`，`word2vec`这几个空文件夹中的文件请从清华云盘https://cloud.tsinghua.edu.cn/d/3f756deb69d04a01b17c/中的对应文件夹下载，并分别放在对应目录下。

```
.
|-- CNN.py
|-- CNN_eval.py
|-- CNN_train.py
|-- LSTM.py
|-- LSTM_eval.py
|-- LSTM_train.py
|-- MLP.py
|-- MLP_eval.py
|-- MLP_train.py
|-- average_emb.py
|-- checkpoints
|   |-- CNN_model.ckpt
|   |-- LSTM_model.ckpt
|   `-- MLP_model.ckpt
|-- data
|   |-- average_test_MLP.pkl
|   |-- average_train_MLP.pkl
|   |-- test_CNN.pkl
|   |-- test_MLP.pkl
|   |-- test_RNN.pkl
|   |-- train_CNN.pkl
|   |-- train_MLP.pkl
|   `-- train_RNN.pkl
|-- deal_data.py
|-- overSampling.py
|-- preprocess.py
|-- readme.md
|-- settings.py
|-- word2vec
    |-- id2word_list.pkl
    |-- word2id_dict.pkl
    `-- word2vex_matrix.npy
```

### 2.1 代码结构

其中`MLP.py`，`CNN.py`，`LSTM.py`为MLP、CNN、RNN（LSTM）的模型代码，`MLP_train.py`，`CNN_train.py`，`LSTM_train.py`为对应模型的训练代码，`MLP_eval.py`，`CNN_eval.py`，`LSTM_eval.py`为评价最终模型的代码，其中最终的模型在`checkpoints`文件夹中，其它代码均为数据和词向量的预处理代码。

对最终模型的评价方式：例如，在cmd中运行`python CNN_eval.py`，会得到如下输出：

![image-20200601180518275](C:\Users\Jacqueline\AppData\Roaming\Typora\typora-user-images\image-20200601180518275.png)

### 2.2 文件夹内容

`checkpoints`中存储最终的完整模型文件，可用`MLP_eval.py`，`CNN_eval.py`，`LSTM_eval.py`代码进行评价，`data`中为处理后的训练集和测试集文件，`word2vec`为用实验说明文档链接提供的预训练词向量（sgns.merge.word文档）处理后的词的token和预训练词向量矩阵文件。

