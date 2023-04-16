import numpy as np
import matplotlib.pyplot as plt


def read_data(f):
    next(f)
    return np.array([line.strip('\n').split(',')[1:] for line in f], dtype=float)


def normalize(x):  # 归一化，减去均值除以方差
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    return (x - mean) / (std + 1e-9)


def split_data(x, y, ratio):  # validation data ratio
    siz = int((1 - ratio) * len(x))
    return x[:siz], y[:siz], x[siz:], y[siz:]


def sigmoid(z):
    # to avoid overflow, minimum/maximum output value is set
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - 1e-8)


def F(x, w, b):
    return sigmoid(np.matmul(x, w) + b)


def cross_entropy_loss(y, y_hat):
    ret = -y_hat * np.log(y) - (1 - y_hat) * np.log(1 - y)
    return np.sum(ret)


def random_shuffle(x, y):
    rnd = np.arange(len(x))
    np.random.shuffle(rnd)
    return x[rnd], y[rnd]


def calc_grad(x, y_hat, w, b):
    y = F(x, w, b)
    pred_error = y_hat - y
    return -np.matmul(x.T, pred_error), -np.sum(pred_error)


def calc_acc(y, y_hat):
    return 1 - np.mean(np.abs(y - y_hat))


if __name__ == "__main__":
    x_train_path = "./data/X_train"
    y_train_path = './data/Y_train'

    x_train = read_data(open(x_train_path))
    y_train = read_data(open(y_train_path))
    print("read: completed")

    x_train = normalize(x_train)
    print("normalize: completed")

    x_train, y_train, x_val, y_val = split_data(x_train, y_train, ratio=0.1)
    dim = x_train.shape[1]
    train_siz = x_train.shape[0]
    val_siz = x_val.shape[0]
    print("size of training data:{}".format(train_siz))
    print("size of validation data:{}".format(val_siz))
    print("size of dim:{}".format(dim))

    # initialize weights and bias
    W = np.zeros([dim, 1])
    B = np.zeros([1, 1])

    max_iter = 10
    batch_size = 8
    learning_rate = 0.20

    # for plotting
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    step = 1
    for epoch in range(max_iter):
        x_train, y_train = random_shuffle(x_train, y_train)

        # Mini_batch training
        for idx in range(int(train_siz/batch_size)):
            X = x_train[idx * batch_size: (idx + 1) * batch_size]
            Y = y_train[idx * batch_size: (idx + 1) * batch_size]

            W_grad, B_grad = calc_grad(X, Y, W, B)
            W = W - learning_rate / np.sqrt(step) * W_grad
            B = B - learning_rate / np.sqrt(step) * B_grad
            step = step + 1

        y_train_pred = F(x_train, W, B)
        loss = cross_entropy_loss(y_train_pred, y_train) / train_siz
        print(epoch, "train loss: ", round(loss, 3))
        train_loss.append(loss)  # loss最后取一个平均值
        y_train_acc = np.round(y_train_pred)
        train_acc.append(calc_acc(y_train_acc, y_train))

        y_val_pred = F(x_val, W, B)
        loss = cross_entropy_loss(y_val_pred, y_val) / val_siz
        print(epoch, "validation loss: ", round(loss, 3))
        val_loss.append(loss)  # loss最后取一个平均值
        y_val_acc = np.round(y_val_pred)
        val_acc.append(calc_acc(y_val_acc, y_val))

    # Loss curve
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.title('Loss')
    plt.legend(['train', 'dev'])
    plt.savefig('loss.png')
    plt.show()

    # Accuracy curve
    plt.plot(train_acc)
    plt.plot(val_acc)
    plt.title('Accuracy')
    plt.legend(['train', 'dev'])
    plt.savefig('acc.png')
    plt.show()

    np.save('weights.npy', W)
    np.save('bias.npy', B)
