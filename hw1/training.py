import pandas as pd
import numpy as np


def init1(d):  # 每一天的向量
    ret = {}
    for month in range(12):
        sample = np.empty([18, 480])
        for day in range(20):
            cur = month * 12 + day
            sample[:, day * 24: (day + 1) * 24] = d[cur * 18:(cur + 1) * 18, :]
        ret[month] = sample
    return ret


def get_feature(d):
    x = np.empty([12 * 471, 18 * 9], dtype=float)  # 共有12个月*(480 - 9)组数据， 每组数据有18 * 9个特征
    y = np.empty([12 * 471, 1], dtype=float)
    cnt = 0
    for month in range(12):
        dm = d[month]
        for day in range(20):
            for hour in range(24):
                cur = day * 24 + hour
                if cur == 471:
                    break
                y[cnt, :] = dm[9, cur + 9]
                x[cnt, :] = dm[:, cur: cur + 9].reshape(1, -1)
                cnt += 1
    return x, y


def normalize(x):  # 归一化，减去均值除以方差
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if std[j] != 0:
                x[i][j] = (x[i][j] - mean[j]) / std[j]
    return x


def ML(x, y):
    dim = 18 * 9 + 1  # 有一个常数b
    w = np.zeros([dim, 1])
    x = np.concatenate((x, np.ones((5652, 1))), axis=1)  # x的最后一列补1 得到统一形式 --> sigma(w*x + b*1)
    learning_rate = 100
    train_times = 1000
    adagrad = np.zeros([dim, 1])  # 使用adagrad梯度下降法
    eps = 1e-9  # 防止adagrad分母为0
    for T in range(train_times):
        if T % 100 == 0:
            loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2)) / 471 / 12)
            print(str(T) + ":" + str(loss))

        gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y)  # dim * 1
        adagrad += gradient ** 2
        w = w - learning_rate * gradient / np.sqrt(adagrad + eps)
    np.save('weight.npy', w)


if __name__ == "__main__":
    df = pd.read_csv("train.csv", encoding="Big5")  # 读入数据，注意编码方式
    df = df.iloc[:, 3:]
    df[df == "NR"] = 0
    data = df.to_numpy()
    data = init1(data)

    train_x, train_y = get_feature(data)
    print(train_x.shape, train_x.dtype)
    train_x = normalize(train_x)
    ML(train_x, train_y)
