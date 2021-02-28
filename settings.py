import pickle
from os.path import join

RAW_TRAIN_DATA_PATH = "./sina/sinanews.train"
RAW_TEST_DATA_PATH = "./sina/sinanews.test"
W2V_DIR = "./word2vec/"
W2V_PATH = "./word2vec/sgns.merge.word"
DATA_DIR = "./data/"
CNN_HISTORY_PATH = "./cnn_history/train_history_eval.txt"
RNN_HISTORY_PATH = "./rnn_history/train_history.txt"
MLP_HISTORY_PATH = "./mlp_history/bal_train_history.txt"
CHECKPOINT_DIR = "./checkpoints/"
BATCH_SIZE = 16
HIDDEN_DIM = 100
EMB_DIM = 300
FILTER_NUM = 100
FILTER_SIZES = [2, 3, 4]
DROPOUT = 0.5
TARGET_SIZE = 8
padding = 0

emotions = {0: "感动", 1: "同情", 2: "无聊", 3: "愤怒", 4: "搞笑", 5: "难过", 6: "新奇", 7: "温馨"}
emo2id = {"感动": 0, "同情": 1, "无聊": 2, "愤怒": 3, "搞笑": 4, "难过": 5, "新奇": 6, "温馨": 7}


# 以二进制保存对象
def store_obj(dir, fname, obj):
    f = open(join(dir, fname), 'wb')
    pickle.dump(obj, f)
    f.close()


# 以读取存储的对象
def load_obj(dir, fname):
    f = open(join(dir, fname), 'rb')
    obj = pickle.load(f)
    f.close()
    return obj
