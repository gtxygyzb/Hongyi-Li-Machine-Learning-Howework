import pandas as pd
import numpy as np
import csv


def normalize(x):  # 归一化，减去均值除以方差
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if std[j] != 0:
                x[i][j] = (x[i][j] - mean[j]) / std[j]
    return x


def init(d):
    ret = np.empty([240, 18 * 9])
    for i in range(240):
        sample = np.empty(18 * 9)
        for j in range(18):
            sample[j * 9: (j + 1) * 9] = d[i * 18 + j, :]
        ret[i] = sample
    return ret


if __name__ == "__main__":
    df = pd.read_csv("test.csv", header=None, encoding="Big5")  # 读入数据，注意编码方式
    df[df == "NR"] = 0
    df = df.iloc[:, 2:]
    data = df.to_numpy()

    test_x = init(data)
    test_x = normalize(test_x)
    test_x = np.concatenate((test_x, np.ones((240, 1))), axis=1)

    w = np.load('weight.npy')
    ans_y = np.dot(test_x, w)

    with open('submit.csv', mode='w', newline='') as submit_file:
        csv_writer = csv.writer(submit_file)
        header = ['id', 'value']
        print(header)
        csv_writer.writerow(header)
        for i in range(240):
            row = ['id_' + str(i), ans_y[i][0]]
            csv_writer.writerow(row)
            print(row)
