import codecs
import numpy as np
from settings import W2V_DIR, W2V_PATH, store_obj, load_obj
from os.path import join


def deal_word2vec(path):
    vocab_num, embed_dim = 0, 0
    word2id = {}
    id2word = []
    count = 1
    with codecs.open(path, "r", "utf-8") as f:
        is_first = True
        for line in f:
            if is_first:
                is_first = False
                vocab_num = int(line.strip().split()[0])
                embed_dim = int(line.rstrip().split()[1])
                # first row used for padding (all 0)
                matrix = np.zeros(shape=(vocab_num + 1, embed_dim), dtype=np.float32)
                print("vocab_num: ", vocab_num)
                print("embed_dim: ", embed_dim)
                continue
            vector = line.strip().split()
            # deduplicate
            if not word2id.__contains__(vector[0]):
                word2id[vector[0]] = count
                matrix[count, :] = np.array([float(x) for x in vector[1:]])
                count += 1

    for w, i in word2id.items():
        id2word.append(w)

    matrix = matrix[0:len(id2word)+1, :]

    # store the token and matrix
    store_obj(W2V_DIR, "word2id_dict.pkl", word2id)
    store_obj(W2V_DIR, "id2word_list.pkl", id2word)
    np.save(join(W2V_DIR, "word2vex_matrix.npy"), matrix)


if __name__ == '__main__':
    deal_word2vec(W2V_PATH)


