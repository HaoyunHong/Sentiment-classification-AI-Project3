import torch
import torch.utils.data as Data
import numpy as np
from scipy.stats import pearsonr
from os.path import join
from settings import W2V_DIR, DATA_DIR, CHECKPOINT_DIR, load_obj
from sklearn.metrics import f1_score


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


if __name__ == '__main__':
    setup_seed(1)
    device = torch.device("cpu")
    test_data = load_obj(DATA_DIR, "test_RNN.pkl")

    y_test = torch.tensor(np.array(test_data["label"]), dtype=torch.long)
    X_test = torch.tensor(np.array(test_data["token"]), dtype=torch.long)
    test_distr = test_data["distribution"]

    sub_test_dataset = Data.TensorDataset(X_test, y_test)
    sub_test_loader = Data.DataLoader(
        dataset=sub_test_dataset,  # torch TensorDataset format
        batch_size=len(sub_test_dataset),  # all test data
    )

    weights = np.load(join(W2V_DIR, "word2vex_matrix.npy"))

    model = torch.load(join(CHECKPOINT_DIR, "LSTM_model.ckpt"), map_location='cpu')
    print(model)
    model = model.to(device)

    pearson = 0
    test_accuracy = 0
    for step0, (batch_x, batch_y) in enumerate(sub_test_loader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        with torch.no_grad():
            tag_scores = model(batch_x)
            ans_score = batch_y[:, 0].view(sub_test_loader.batch_size)

            prediction = torch.max(tag_scores, 1)[1]  # 返回每行最大的数的索引
            pred_y = prediction.data.numpy()
            target_y = batch_y[:, 0].view(sub_test_loader.batch_size).data.numpy()
            for index, distr in enumerate(test_distr):
                pearson += pearsonr(np.array(test_distr[index]), np.array(tag_scores[index]))[0]

            test_accuracy += (pred_y == target_y).astype(int).sum()

            if step0 == len(sub_test_loader) - 1:
                weighted_f1 = f1_score(target_y, pred_y, average='weighted')
                print("test accuracy = {:.2f}%".format(test_accuracy * 100 / (sub_test_loader.batch_size * (step0 + 1))))
                print("test F1 = {:.2f}%".format(weighted_f1*100))
                print("coef = {:.4f}".format(pearson/len(test_distr)))

