import training
import numpy as np
import csv


if __name__ == "__main__":
    x_test = training.read_data(open("./data/X_test"))
    x_test = training.normalize(x_test)
    n = x_test.shape[0]
    w = np.load("weights.npy")
    b = np.load("bias.npy")
    ans = np.round(training.F(x_test, w, b)).astype(int)

    with open('./data/logistic.csv', mode='w', newline='') as submit_file:
        csv_writer = csv.writer(submit_file)
        header = ['id', 'label']
        csv_writer.writerow(header)
        for i in range(n):
            row = [i, ans[i][0]]
            csv_writer.writerow(row)
