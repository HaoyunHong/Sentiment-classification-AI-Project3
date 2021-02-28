import codecs
from settings import RAW_TRAIN_DATA_PATH, RAW_TEST_DATA_PATH, DATA_DIR, TRAIN_DATA_NAME, TEST_DATA_NAME, W2V_DIR, \
    store_obj, load_obj, VOCAB_SIZE, emotions
from collections import Counter


def is_Chinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False


def deal_data(path):
    word2id = load_obj(W2V_DIR, "word2id_dict.pkl")
    # w2v_matrix = np.load(join(W2V_DIR, "word2vex_matrix.npy"))
    with codecs.open(path, "r", "utf-8") as f:
        data = {"distribution": [], "label": [], "token": []}
        for i, line in enumerate(f):
            data["token"].append([])
            line_list = line.strip().split()
            _sum = int(line_list[1].split(':')[1])
            # scale directly here
            _labels = [int(v.split(':')[1]) / _sum for v in line_list[2:10]]
            data["label"].append([_labels.index(max(_labels))])
            data["distribution"].append(_labels)

            result = Counter(line_list[10:])
            for w in line_list[10:]:
                # if it's not Chinese, ignore it
                if not is_Chinese(w):
                    continue
                '''
                if result[w] < 2:
                    continue
                '''
                try:
                    data["token"][i].append(word2id[w])
                # if a word is not in pre-trained embedding, ignore it
                except KeyError:
                    continue
                seq_lens.append(len(data["token"][i]))
    return data


def uniform_seq_len(data):
    data_copy = data["token"].copy()
    for i, t in enumerate(data_copy):
        if seq_len > len(t):
            data["token"][i] += ([0] * (seq_len - len(t)))
        else:
            data["token"][i] = data["token"][i][:seq_len]
    return data


#  将数据写入新文件
def label_analyze(file_path, datas):
    distr = {"感动": 0, "同情": 0, "无聊": 0, "愤怒": 0, "搞笑": 0, "难过": 0, "新奇": 0, "温馨": 0}
    num = 0
    for i, data in enumerate(datas):
        num += 1
        distr[emotions[int(data)]] += 1
    for i in range(0, 8):
        distr[emotions[i]] = distr[emotions[i]]/num
    print(distr)


if __name__ == '__main__':
    seq_lens = []
    train_data = deal_data(RAW_TRAIN_DATA_PATH)
    test_data = deal_data(RAW_TEST_DATA_PATH)
    # seq_len = np.percentile(np.array(seq_lens), 90)
    # print(seq_len)
    seq_len = VOCAB_SIZE
    # Uniform text length
    train_data = uniform_seq_len(train_data)
    test_data = uniform_seq_len(test_data)
    
    store_obj(DATA_DIR, TRAIN_DATA_NAME, "train_RNN.pkl")
    store_obj(DATA_DIR, TEST_DATA_NAME, "test_RNN.pkl")
