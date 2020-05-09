from scrapeData import *
import keras
from keras.models import load_model
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Conv2D, BatchNormalization, MaxPooling2D, Flatten


def get_sizes(stocks, split1, split2):
    train_sizes = [0]
    test_sizes = [0]
    trade_sizes = [0]
    for k in range(len(stocks)):
        if stocks[k].ticker[0] != 'No File':
            stock_len = len(stocks[k].metrics[:, 0])
            split_idx1 = round(split1 * stock_len)
            split_idx2 = round(split2 * stock_len)
            train_sizes.append(split_idx1)
            test_sizes.append(split_idx2 - split_idx1)
            trade_sizes.append(stock_len - split_idx2)
    return train_sizes, test_sizes, trade_sizes


def allocate_train_test_trade(stocks, split1, split2, h, cnn):
    train_sizes, test_sizes, trade_sizes = get_sizes(stocks, split1, split2)
    if cnn:
        w = stocks[0].metrics.shape[1]
        train_sizes_idx = list(np.cumsum(np.array(train_sizes)))
        train_data = np.zeros((train_sizes_idx[-1], h, w))
        train_labels = np.zeros((train_sizes_idx[-1], 1))

        test_sizes_idx = list(np.cumsum(np.array(test_sizes)))
        test_data = np.zeros((test_sizes_idx[-1], h, w))
        test_labels = np.zeros((test_sizes_idx[-1], 1))

        trade_sizes_idx = list(np.cumsum(np.array(trade_sizes)))
        trade_data = np.zeros((trade_sizes_idx[-1], h, w))
        trade_labels = np.zeros((trade_sizes_idx[-1], 1))
    else:
        train_sizes_idx = list(np.cumsum(np.array(train_sizes)))
        train_data = np.zeros((train_sizes_idx[-1], stocks[0].metrics.shape[1]))
        train_labels = np.zeros((train_sizes_idx[-1], 1))

        test_sizes_idx = list(np.cumsum(np.array(test_sizes)))
        test_data = np.zeros((test_sizes_idx[-1], stocks[0].metrics.shape[1]))
        test_labels = np.zeros((test_sizes_idx[-1], 1))

        trade_sizes_idx = list(np.cumsum(np.array(trade_sizes)))
        trade_data = np.zeros((trade_sizes_idx[-1], stocks[0].metrics.shape[1]))
        trade_labels = np.zeros((trade_sizes_idx[-1], 1))

    return train_data, train_labels, test_data, test_labels, trade_data, trade_labels


def populate_class(stocks, split1, split2):
    total = len(stocks)
    col_data = stocks[0].names
    for k in range(len(stocks)):
        waitbar(total, int(k))
        print(stocks[k].ticker[0])
        if stocks[k].ticker[0] != 'No File':
            split_idx1 = round(split1 * len(stocks[k].metrics[:, 0]))
            split_idx2 = round(split2 * len(stocks[k].metrics[:, 0]))
            stocks[k].train_data = stocks[k].metrics[:split_idx1, :].copy()
            stocks[k].test_data = stocks[k].metrics[split_idx1:split_idx2, :].copy()
            stocks[k].trade_data = stocks[k].metrics[split_idx2:, :].copy()
            stocks[k].train_labels = stocks[k].label_pg[:split_idx1]
            stocks[k].test_labels = stocks[k].label_pg[split_idx1:split_idx2]
            stocks[k].trade_labels = stocks[k].label_pg[split_idx2:]
    return stocks


def train_test_validate(stocks, split1, split2, h, cnn=True):
    train_sizes, test_sizes, trade_sizes = get_sizes(stocks, split1, split2)
    train_data, train_labels, test_data, test_labels, trade_data, trade_labels = allocate_train_test_trade(
        stocks, split1, split2, h, cnn)

    train_sizes_idx = list(np.cumsum(np.array(train_sizes)))
    test_sizes_idx = list(np.cumsum(np.array(test_sizes)))
    trade_sizes_idx = list(np.cumsum(np.array(trade_sizes)))

    if cnn:
        w = stocks[0].metrics.shape[1]
        for k in range(len(stocks)):
            stocks[k].trade_data_tensor = np.zeros((trade_sizes[k + 1], h, w))
            for n in range(train_sizes[k + 1]):
                if n >= h:
                    tsi = train_sizes_idx[k]
                    train_data[tsi + n, :, :] = stocks[k].train_data[int(n - h):n, :]
            for n in range(test_sizes[k + 1]):
                if n >= h:
                    tsi = test_sizes_idx[k]
                    test_data[tsi + n, :, :] = stocks[k].test_data[int(n - h):n, :]
            for n in range(trade_sizes[k + 1]):
                if n >= h:
                    tsi = trade_sizes_idx[k]
                    trade_data[tsi + n, :, :] = stocks[k].trade_data[int(n - h):n, :]
                    stocks[k].trade_data_tensor[n, :, :] = stocks[k].trade_data[int(n - h):n, :]

        train_labels[train_sizes_idx[k]:train_sizes_idx[k + 1]] = stocks[k].train_labels
        test_labels[test_sizes_idx[k]:test_sizes_idx[k + 1]] = stocks[k].test_labels
        trade_labels[trade_sizes_idx[k]:trade_sizes_idx[k + 1]] = stocks[k].trade_labels

        # train = (train_data, train_labels, train_labels_og)  # test = (test_data, test_labels, test_labels_og)  # validate = (trade_data, trade_labels, trade_labels_og)
    else:
        for k in range(len(stocks)):
            train_data[train_sizes_idx[k]:train_sizes_idx[k + 1]] = stocks[k].train_data
            test_data[test_sizes_idx[k]:test_sizes_idx[k + 1]] = stocks[k].test_data
            trade_data[trade_sizes_idx[k]:trade_sizes_idx[k + 1]] = stocks[k].trade_data
            train_labels[train_sizes_idx[k]:train_sizes_idx[k + 1]] = stocks[k].train_labels
            test_labels[test_sizes_idx[k]:test_sizes_idx[k + 1]] = stocks[k].test_labels
            trade_labels[trade_sizes_idx[k]:trade_sizes_idx[k + 1]] = stocks[k].trade_labels

        # train_labels_og = train_labels  # test_labels_og = test_labels  # trade_labels_og = trade_labels  # buy_mask = np.nonzero(train_labels_og == 1)[0]  # sell_mask = np.nonzero(train_labels_og == 0)[0]  # keep_idx = random.sample(list(sell_mask), len(buy_mask))  # keep = np.hstack((keep_idx, buy_mask))  #  # train_labels = to_categorical(train_labels[keep, :])  # test_labels = to_categorical(test_labels)  # train_data = train_data[keep, :]

    train_labels_og = train_labels
    buy_mask = np.nonzero(train_labels_og == 1)[0]
    sell_mask = np.nonzero(train_labels_og == 0)[0]
    keep_idx = random.sample(list(sell_mask), len(buy_mask))
    keep = np.hstack((keep_idx, buy_mask))
    train_labels = to_categorical(train_labels[keep, :])
    train_data = train_data[keep, :]

    test_labels_og = test_labels
    buy_mask = np.nonzero(test_labels_og == 1)[0]
    sell_mask = np.nonzero(test_labels_og == 0)[0]
    keep_idx = random.sample(list(sell_mask), len(buy_mask))
    keep = np.hstack((keep_idx, buy_mask))
    test_labels = to_categorical(test_labels[keep, :])
    test_data = test_data[keep, :]

    trade_labels_og = trade_labels
    train = (train_data, train_labels, train_labels_og)
    test = (test_data, test_labels, test_labels_og)
    validate = (trade_data, trade_labels, trade_labels_og)
    return train, test, validate


def make_1d_ann(train, test, load_nn_model=0, model_name=''):
    train_data = train[0]
    train_labels = train[1]
    test_data = test[0]
    test_labels = test[1]
    n_inputs = train_data.shape[1]
    n_outputs = 2
    model = Sequential()
    model.add(Dense(n_inputs, input_dim=n_inputs, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(n_outputs))
    omt = keras.optimizers.Adam(lr=0.005)
    loss = 'binary_crossentropy'
    model.compile(loss=loss, optimizer=omt, metrics=['binary_accuracy'])
    es = EarlyStopping(monitor='val_loss', patience=10)
    history = model.fit(train_data, train_labels, epochs=500,
        batch_size=int(len(test_labels) * 0.01), verbose=True, shuffle=True,
        validation_data=(test_data, test_labels), callbacks=[es])
    scoresTest = model.evaluate(test_data, test_labels, verbose=0)
    prd = model.predict_classes(test_data)
    print(np.column_stack((prd, test_labels)))
    print(
        str(model.metrics_names[1]) + ' %.2f%%' % (scoresTest[1] * 100) + ' accuracy on test data')
    if load_nn_model == 1:
        model = load_model(model_name + '.h5')
        scoresTest = model.evaluate(test_data, test_labels, verbose=0)
        prd = model.predict_classes(test_data)
        print(str(model.metrics_names[1]) + ' %.2f%%' % (
                scoresTest[1] * 100) + ' accuracy on test data')
    return model, history


def make_2d_cnn(train, test, h, load_nn_model=0, model_name=''):
    train_data = train[0]
    train_labels = train[1]
    test_data = test[0]
    test_labels = test[1]
    train_data = np.expand_dims(train_data, axis=3)
    test_data = np.expand_dims(test_data, axis=3)
    n_inputs = train_data.shape[1]
    n_outputs = 2
    model = Sequential()
    model.add(Conv2D(8, kernel_size=(10, 10), strides=(1, 1), activation='relu',
        input_shape=(h, n_inputs, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Conv2D(8, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    omt = keras.optimizers.Adam(lr=0.0005)
    loss = 'binary_crossentropy'
    model.compile(loss=loss, optimizer=omt, metrics=['binary_accuracy'])
    es = EarlyStopping(monitor='val_loss', patience=10)
    batch_size = int(len(test_labels) * 0.01) if len(test_labels) > 100 else len(test_labels)
    history = model.fit(train_data, train_labels, epochs=100,
        batch_size=batch_size, verbose=True, shuffle=False,
        validation_data=(test_data, test_labels), callbacks=[es])
    scoresTest = model.evaluate(test_data, test_labels, verbose=0)
    prd = model.predict_classes(test_data)
    print(np.column_stack((prd, test_labels)))
    print(
        str(model.metrics_names[1]) + ' %.2f%%' % (scoresTest[1] * 100) + ' accuracy on test data')
    if load_nn_model == 1:
        model = load_model(model_name + '.h5')
        scoresTest = model.evaluate(test_data, test_labels, verbose=0)
        prd = model.predict_classes(test_data)
        print(str(model.metrics_names[1]) + ' %.2f%%' % (
                scoresTest[1] * 100) + ' accuracy on test data')
    return model, history
