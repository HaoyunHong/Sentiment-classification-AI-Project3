import torch
import numpy as np
import os
from os.path import join
from settings import emotions, emo2id, DATA_DIR, TRAIN_DATA_NAME, BALANCED_TRAIN_DATA_NAME, CHECKPOINT_DIR, \
    store_obj, load_obj, EMB_DIM, FILTER_NUM, FILTER_SIZES, DROPOUT, TARGET_SIZE, BATCH_SIZE, CNN_HISTORY_PATH


if __name__ == '__main__':
    train_data = load_obj(DATA_DIR, TRAIN_DATA_NAME)
    samples = {"感动": [], "同情": [], "无聊": [], "愤怒": [], "搞笑": [], "难过": [], "新奇": [], "温馨": []}
    for i, emo in enumerate(train_data["label"]):
        samples[emotions[int(emo[0])]].append(train_data["token"][i])

    max_num = 0
    for (k, v) in samples.items():
        if max_num < len(v):
            max_num = len(v)
    print(max_num)

    for (k, v) in samples.items():
        copy_num = max_num // len(v)
        v_copy = v.copy()
        for t in v_copy:
            for i in range(copy_num - 1):
                v.append(t)

    for (k, v) in samples.items():
        print(k)
        print(len(v))

    balanced_train_data = {"label": [], "token": []}
    for (k, v) in samples.items():
        # print(v[0])
        print([emo2id[k]])
        balanced_train_data["token"] += v
        for i in range(len(v)):
            balanced_train_data["label"].append([emo2id[k]])

    print(balanced_train_data["label"])

    store_obj(DATA_DIR, BALANCED_TRAIN_DATA_NAME, balanced_train_data)

