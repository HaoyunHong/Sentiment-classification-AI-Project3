from MLP import MLP
import torch
from torch import nn
import torch.optim as optim
import torch.utils.data as Data
import numpy as np
from scipy.stats import pearsonr
import os
from settings import DATA_DIR, load_obj, EMB_DIM, DROPOUT, TARGET_SIZE, BATCH_SIZE, MLP_HISTORY_PATH, HIDDEN_DIM
from sklearn.metrics import f1_score


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


if __name__ == '__main__':
    # 设置随机数种子
    setup_seed(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_data = load_obj(DATA_DIR, "train_MLP.pkl")
    test_data = load_obj(DATA_DIR, "test_MLP.pkl")
    train_input = load_obj(DATA_DIR, "average_train_MLP.pkl")
    test_input = load_obj(DATA_DIR, "average_test_MLP.pkl")

    X_train = torch.tensor(np.array(train_input), dtype=torch.float)
    X_test = torch.tensor(np.array(test_input), dtype=torch.float)
    y_train = torch.tensor(np.array(train_data["label"]), dtype=torch.long)
    y_test = torch.tensor(np.array(test_data["label"]), dtype=torch.long)
    test_distr = test_data["distribution"]

    sub_train_dataset = Data.TensorDataset(X_train, y_train)
    sub_train_loader = Data.DataLoader(
        dataset=sub_train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )

    # 为了作对应就不打乱数据了
    sub_test_dataset = Data.TensorDataset(X_test, y_test)
    sub_test_loader = Data.DataLoader(
        dataset=sub_test_dataset,  # torch TensorDataset format
        batch_size=len(sub_test_dataset),  # all test data
        drop_last=True,
    )

    model = MLP(EMB_DIM, 10, DROPOUT, TARGET_SIZE)
    model = model.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=0.001)

    # start training
    for epoch in range(0, 300):
        train_accuracy = 0
        test_accuracy = 0
        pearson = 0

        for step, (batch_x, batch_y) in enumerate(sub_train_loader):
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            model.zero_grad()

            # writer.add_graph(model, (batch_x,))
            tag_scores = model(batch_x)

            prediction = torch.max(tag_scores, 1)[1]  # 返回每行最大的数的索引
            pred_y = prediction.cpu().data.numpy()
            target_y = batch_y[:, 0].view(BATCH_SIZE).cpu().data.numpy()

            train_accuracy += (pred_y == target_y).astype(int).sum()
            ans_score = batch_y[:, 0].view(BATCH_SIZE)
            loss = loss_function(tag_scores, ans_score)

            if step == len(sub_train_loader) - 1:
                print("epoch: {}".format(epoch))
                print(' ')
                print("train accuracy = {:.2f}%".format(train_accuracy * 100 / (BATCH_SIZE * (step + 1))))

            loss.backward()
            optimizer.step()

        for step0, (batch_x, batch_y) in enumerate(sub_test_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            with torch.no_grad():
                tag_scores = model(batch_x)
                ans_score = batch_y[:, 0].view(sub_test_loader.batch_size)
                loss = loss_function(tag_scores, ans_score)

                prediction = torch.max(tag_scores, 1)[1]  # 返回每行最大的数的索引
                pred_y = prediction.cpu().data.numpy()
                target_y = batch_y[:, 0].view(sub_test_loader.batch_size).cpu().data.numpy()

                for index, distr in enumerate(test_distr):
                    pearson += pearsonr(np.array(test_distr[index]), np.array(tag_scores[index].cpu()))[0]
                test_accuracy += (pred_y == target_y).astype(int).sum()

                if step0 == len(sub_test_loader) - 1:
                    weighted_f1 = f1_score(target_y, pred_y, average='weighted')
                    print("test accuracy = {:.2f}%".format(
                        test_accuracy * 100 / (sub_test_loader.batch_size * (step0 + 1))))
                    print("test F1 = {:.2f}%".format(weighted_f1 * 100))
                    print("coef = {:.4f}".format(pearson / len(test_distr)))
                    print('-' * 30)
