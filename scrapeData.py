import re
import os
import time
import random
import requests
import datetime
import numpy as np
import matplotlib.pyplot as plt
from candle import *
# from matplotlib.finance import candlestick2_ohlc
import keras
from keras.models import load_model
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Conv2D, BatchNormalization, MaxPooling2D, Flatten

csv_location = 'C:/Users/carme/Desktop/TheProverbialCode/StockMarket/CSVFiles/'


# s = 25
# random.seed(6+s)
# tf.set_random_seed(333+s*2)
# np.random.seed(856+s*3)


def split_crumb_store(v):
    if v is None:
        return
    return v.split(':')[2].strip('"')


def find_crumb_store(lines):
    for l in lines:
        if re.findall(r'CrumbStore', l):
            return l
    print("Did not find CrumbStore")


def get_cookie_value(r):
    if not r.cookies:
        return print('No Cookie 42')
    return {'B': r.cookies['B']}


def get_page_data(symbol):
    url = "https://finance.yahoo.com/quote/%s/?p=%s" % (symbol, symbol)
    r = requests.get(url, timeout=10)
    cookie = get_cookie_value(r)
    lines = r.content.decode('latin-1').replace('\\', '')
    lines = lines.replace('}', '\n')
    return cookie, lines.split('\n')


def get_cookie_crumb(symbol):
    cookie, lines = get_page_data(symbol)
    crumb = split_crumb_store(find_crumb_store(lines))
    return cookie, crumb


def get_data(symbol, start_date, end_date, cookie, crumb, append_to_file, tries):
    filename = csv_location + '%s.csv' % (symbol)
    url = "https://query1.finance.yahoo.com/v7/finance/download/%s?period1=%s&period2=%s&interval=1d&events=history&crumb=%s" % (
        symbol, start_date, end_date, crumb)
    response = requests.get(url, cookies=cookie, timeout=10)
    handle = open(filename, 'wb')
    count = 0
    for block in response.iter_content(1024):
        if block[0:1].decode('UTF-8') == '{':
            tries += 1
            print(str(tries) + ' Download Attempts')
            cookie, crumb = get_cookie_crumb(symbol)
            if tries > 5:
                break
            tries = get_data(symbol, start_date, end_date, cookie, crumb, append_to_file, tries)
        else:
            tries = 0
            if count == 0 and append_to_file == 1:
                handle.write(block[42:])
            else:
                handle.write(block)
    handle.close()
    return tries


def get_now_epoch():
    return int(time.time())


def waitbar(total, current):
    current += 1
    percent_complete = 100 * (current / total)
    here_sym = '>'
    complete_sym = '-'
    advance = str(int(np.round((percent_complete / 2) - 1)) * complete_sym + here_sym)
    retreat = str(int(np.round(((100 - percent_complete) / 2) - 1)) * '.')
    print(advance + retreat + ' ' + str(np.round(percent_complete, 3)) + '%')


def download_quotes(symbols):
    num_symbols = len(symbols)
    csv_present = os.listdir(csv_location)
    for i in range(0, num_symbols):
        symbol = symbols[i][0]
        if symbol + '.csv' not in csv_present:
            start_date = 0
            append_to_file = 0
            last_date = 0
            print("--------------------------------------------------")
            print("Downloading %s to %s.csv" % (symbol, symbol))
            waitbar(num_symbols, i)
            end_date = get_now_epoch()
            cookie, crumb = get_cookie_crumb(symbol)
            tries = 0
            tries = get_data(symbol, start_date, end_date, cookie, crumb, append_to_file, tries)
            filename = csv_location + '%s.csv' % (symbol)
            if tries >= 5:
                os.remove(filename)


class StockClass:
    def __init__(self, symbol, sector, filename):
        try:
            data = np.genfromtxt(csv_location + symbol + '.csv', delimiter=',', skip_header=1,
                converters={0: lambda s: (
                    datetime.datetime.strptime(s.decode('ascii'), '%Y-%m-%d').timestamp())})
            self.ticker = [symbol]
            self.sector = [sector]
            data = self.filter_out_nan(data)
            open = data[:, 1]
            open[open == 0] = 0.001
            high = data[:, 2]
            high[high == 0] = 0.001
            low = data[:, 3]
            low[low == 0] = 0.001
            close = data[:, 4]
            close[close == 0] = 0.001
            volume = data[:, 6]
            volume[volume == 0] = 0.001
            data_mean = data[:, 1:5].mean(1)
            open_roll = np.roll(open, 1, 0)
            high_roll = np.roll(high, 1, 0)
            low_roll = np.roll(low, 1, 0)
            close_roll = np.roll(close, 1, 0)
            open_diff = (open - open_roll) / open_roll
            high_diff = (high - high_roll) / high_roll
            low_diff = (low - low_roll) / low_roll
            close_diff = (close - close_roll) / close_roll
            open_diff[0] = open[1]
            high_diff[0] = high[1]
            low_diff[0] = low[1]
            close_diff[0] = close[1]
            open_roll2 = np.roll(open, 2, 0)
            high_roll2 = np.roll(high, 2, 0)
            low_roll2 = np.roll(low, 2, 0)
            close_roll2 = np.roll(close, 2, 0)
            open_diff2 = (open - open_roll2) / open_roll2
            high_diff2 = (high - high_roll2) / high_roll2
            low_diff2 = (low - low_roll2) / low_roll2
            close_diff2 = (close - close_roll2) / close_roll2
            open_diff2[0:2] = open[1:3]
            high_diff2[0:2] = high[1:3]
            low_diff2[0:2] = low[1:3]
            close_diff2[0:2] = close[1:3]
            open_roll3 = np.roll(open, 3, 0)
            high_roll3 = np.roll(high, 3, 0)
            low_roll3 = np.roll(low, 3, 0)
            close_roll3 = np.roll(close, 3, 0)
            open_diff3 = (open - open_roll3) / open_roll3
            high_diff3 = (high - high_roll3) / high_roll3
            low_diff3 = (low - low_roll3) / low_roll3
            close_diff3 = (close - close_roll3) / close_roll3
            open_diff3[0:3] = open[1:4]
            high_diff3[0:3] = high[1:4]
            low_diff3[0:3] = low[1:4]
            close_diff3[0:3] = close[1:4]
            open_roll4 = np.roll(open, 4, 0)
            high_roll4 = np.roll(high, 4, 0)
            low_roll4 = np.roll(low, 4, 0)
            close_roll4 = np.roll(close, 4, 0)
            open_diff4 = (open - open_roll4) / open_roll4
            high_diff4 = (high - high_roll4) / high_roll4
            low_diff4 = (low - low_roll4) / low_roll4
            close_diff4 = (close - close_roll4) / close_roll4
            open_diff4[0:4] = open[1:5]
            high_diff4[0:4] = high[1:5]
            low_diff4[0:4] = low[1:5]
            close_diff4[0:4] = close[1:5]
            percent_change = 100 * (close - open) / open
            percent_change[0] = 0
            roll_1 = np.roll(data_mean, 1, 0)
            roll_2 = np.roll(data_mean, 2, 0)
            roll_3 = np.roll(data_mean, 3, 0)
            vroll_1 = np.roll(data[:, -1], 1, 0)
            vroll_2 = np.roll(data[:, -1], 2, 0)
            vroll_3 = np.roll(data[:, -1], 3, 0)
            d1price = roll_2 - 4 * roll_1 + 3 * data_mean
            d2price = roll_2 - 2 * roll_1 + data_mean
            d3price = -roll_3 + 3 * roll_2 - 3 * roll_1 + data_mean
            d1volume = vroll_2 - 4 * vroll_1 + 3 * data[:, -1]
            d2volume = vroll_2 - 2 * vroll_1 + data[:, -1]
            d3volume = -vroll_3 + 3 * vroll_2 - 3 * vroll_1 + data[:, -1]
            d1price[0:3] = d1price[3]
            d2price[0:3] = d2price[3]
            d3price[0:3] = d3price[3]
            d1volume[0:3] = d1volume[3]
            d2volume[0:3] = d2volume[3]
            d3volume[0:3] = d3volume[3]
            ma5 = self.moving_average(5, data_mean)
            ma10 = self.moving_average(10, data_mean)
            ma15 = self.moving_average(15, data_mean)
            ma20 = self.moving_average(20, data_mean)
            self.metrics = np.column_stack((
                open, high, low, close, volume, data_mean, open_diff, high_diff, low_diff,
                close_diff, percent_change, d1price, d2price, d3price, d1volume, d2volume, d3volume,
                ma5, ma10, ma15, ma20, open_diff2, high_diff2, low_diff2, close_diff2, open_diff3,
                high_diff3, low_diff3, close_diff3, open_diff4, high_diff4, low_diff4, close_diff4))
            self.metrics_column_data = {'open': 0, 'high': 1, 'low': 2, 'close': 3, 'volume': 4,
                                        'ohlc_mean': 5, 'open_pct_diff': 6, 'high_pct_diff': 7,
                                        'low_pct_diff': 8, 'close_pct_diff': 9, 'pct_change': 10,
                                        'd1price': 11, 'd2price': 12, 'd3price': 13, 'd1volume': 14,
                                        'd2volume': 15, 'd3volume': 16, 'ma5': 17, 'ma10': 18,
                                        'ma15': 19, 'ma20': 20, 'open_diff2': 21, 'high_diff2': 22,
                                        'low_diff2': 23, 'close_diff2': 24, 'open_diff3': 25,
                                        'high_diff3': 26, 'low_diff3': 27, 'close_diff3': 28,
                                        'open_diff4': 29, 'high_diff4': 30, 'low_diff4': 31,
                                        'close_diff4': 32}
            print('Adding Data For Stock %s' % symbol)
        except Exception:
            self.metrics = []
            self.ticker = ['No File']
            print('file: %s.csv' % symbol, 'can not add data  - In Class')

    def filter_out_nan(self, data):
        if self.ticker[:][0] != 'No File':
            for rr in range(data.shape[1]):
                mask = np.isnan(data[:, rr])
                data[mask, rr] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask),
                    data[~mask, rr])
            return data

    def moving_average(self, ma, data):
        csum = np.cumsum(np.insert(data, np.ones(ma + 1), 0))
        mavg = (csum[ma:] - csum[:-ma]) / float(ma)
        mavg = np.delete(mavg, 0)
        return mavg


def parse_csv(symbols):
    stock = []
    print(len(symbols))
    for i in range(len(symbols)):
        try:
            filename = csv_location + '%s.csv' % (symbols[i][0])
            if os.path.exists(filename):
                stock.append(StockClass(symbols[i][0], symbols[i][1], filename))
        except Exception:
            print('file: %s.csv' % symbols[i][0], symbols[i][1], 'does not exist - In parse_csv')
    return stock


def gather_tickers(ticker_list):
    tickers = open(ticker_list, 'r')
    tickers = tickers.read()
    tickers = tickers.split('\n')
    for i in range(len(tickers)):
        tickers[i] = tickers[i].split('\t')
    return tickers


def make_labels_percent_gain(stocks, label_pg_crit):
    for i in range(len(stocks)):
        if stocks[i].ticker[0][:] != 'No File':
            idx = stocks[i].metrics_column_data['pct_change']
            stocks[i].label_pg = np.zeros((len(stocks[i].metrics[:, 0]), 1))
            buy_label_idx = np.nonzero(stocks[i].metrics[:, idx] >= label_pg_crit)
            stocks[i].label_pg[buy_label_idx[0][:] - 1, :] = 1
    return stocks


def normalize_data(data):
    try:
        for i in range(data.shape[1]):
            data[:, i] = data[:, i] - np.mean(data[:, i])
            data[:, i] = data[:, i] / np.max(data[:, i])
    except IndexError:
        data[:] = data[:] / np.max(data[:])
    return data


def train_test_validate_2d(stocks):
    col_data = stocks[0].metrics_column_data
    split_percent = 0.80
    split_percent2 = 0.95
    train_sizes = [0]
    test_sizes = [0]
    trade_sizes = [0]
    for k in range(len(stocks)):
        if stocks[k].ticker[0] != 'No File':
            stock_len = len(stocks[k].metrics[:, 0])
            split_idx1 = round(split_percent * stock_len)
            split_idx2 = round(split_percent2 * stock_len)
            train_sizes.append(split_idx1)
            test_sizes.append(split_idx2 - split_idx1)
            trade_sizes.append(stock_len - split_idx2)

    h = 30
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
    total = len(stocks)
    for k in range(len(stocks)):
        waitbar(total, int(k))
        print(stocks[k].ticker[0])
        if stocks[k].ticker[0] != 'No File':
            split_idx1 = round(split_percent * len(stocks[k].metrics[:, 0]))
            split_idx2 = round(split_percent2 * len(stocks[k].metrics[:, 0]))
            stocks[k].train_data = stocks[k].metrics[:split_idx1, :].copy()
            stocks[k].test_data = stocks[k].metrics[split_idx1:split_idx2, :].copy()
            stocks[k].trade_data = stocks[k].metrics[split_idx2:, :].copy()
            norm_except = col_data['pct_change']
            norm_these = np.delete(np.arange(len(col_data)), norm_except)
            stocks[k].train_data[:, norm_these] = normalize_data(
                stocks[k].train_data[:, norm_these])
            stocks[k].test_data[:, norm_these] = normalize_data(stocks[k].test_data[:, norm_these])
            stocks[k].trade_data[:, norm_these] = normalize_data(
                stocks[k].trade_data[:, norm_these])
            stocks[k].train_labels = stocks[k].label_pg[:split_idx1]
            stocks[k].test_labels = stocks[k].label_pg[split_idx1:split_idx2]
            stocks[k].trade_labels = stocks[k].label_pg[split_idx2:]

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

    train_labels_og = train_labels
    test_labels_og = test_labels
    trade_labels_og = trade_labels
    buy_mask = np.nonzero(train_labels_og == 1)[0]
    sell_mask = np.nonzero(train_labels_og == 0)[0]
    keep_idx = random.sample(list(sell_mask), len(buy_mask))
    keep = np.hstack((keep_idx, buy_mask))

    # train_labels = to_categorical(train_labels[keep, :])
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    # train_data = train_data[keep, :]

    train = (train_data, train_labels, train_labels_og)
    test = (test_data, test_labels, test_labels_og)
    validate = (trade_data, trade_labels, trade_labels_og)
    return train, test, validate


def train_test_validate_1d(stocks):
    col_data = stocks[0].metrics_column_data
    split_percent = 0.80
    split_percent2 = 0.95
    train_sizes = [0]
    test_sizes = [0]
    trade_sizes = [0]
    for k in range(len(stocks)):
        if stocks[k].ticker[0] != 'No File':
            stock_len = len(stocks[k].metrics[:, 0])
            split_idx1 = round(split_percent * stock_len)
            split_idx2 = round(split_percent2 * stock_len)
            train_sizes.append(split_idx1)
            test_sizes.append(split_idx2 - split_idx1)
            trade_sizes.append(stock_len - split_idx2)

    train_sizes_idx = list(np.cumsum(np.array(train_sizes)))
    train_data = np.zeros((train_sizes_idx[-1], stocks[0].metrics.shape[1]))
    train_labels = np.zeros((train_sizes_idx[-1], 1))

    test_sizes_idx = list(np.cumsum(np.array(test_sizes)))
    test_data = np.zeros((test_sizes_idx[-1], stocks[0].metrics.shape[1]))
    test_labels = np.zeros((test_sizes_idx[-1], 1))

    trade_sizes_idx = list(np.cumsum(np.array(trade_sizes)))
    trade_data = np.zeros((trade_sizes_idx[-1], stocks[0].metrics.shape[1]))
    trade_labels = np.zeros((trade_sizes_idx[-1], 1))
    total = len(stocks)
    for k in range(len(stocks)):
        waitbar(total, int(k))
        print(stocks[k].ticker[0])
        if stocks[k].ticker[0] != 'No File':
            split_idx1 = round(split_percent * len(stocks[k].metrics[:, 0]))
            split_idx2 = round(split_percent2 * len(stocks[k].metrics[:, 0]))
            stocks[k].train_data = stocks[k].metrics[:split_idx1, :].copy()
            stocks[k].test_data = stocks[k].metrics[split_idx1:split_idx2, :].copy()
            stocks[k].trade_data = stocks[k].metrics[split_idx2:, :].copy()
            norm_except = col_data['pct_change']
            norm_these = np.delete(np.arange(len(col_data)), norm_except)
            stocks[k].train_data[:, norm_these] = normalize_data(
                stocks[k].train_data[:, norm_these])
            stocks[k].test_data[:, norm_these] = normalize_data(stocks[k].test_data[:, norm_these])
            stocks[k].trade_data[:, norm_these] = normalize_data(
                stocks[k].trade_data[:, norm_these])
            stocks[k].train_labels = stocks[k].label_pg[:split_idx1]
            stocks[k].test_labels = stocks[k].label_pg[split_idx1:split_idx2]
            stocks[k].trade_labels = stocks[k].label_pg[split_idx2:]
            train_data[train_sizes_idx[k]:train_sizes_idx[k + 1]] = stocks[k].train_data
            test_data[test_sizes_idx[k]:test_sizes_idx[k + 1]] = stocks[k].test_data
            trade_data[trade_sizes_idx[k]:trade_sizes_idx[k + 1]] = stocks[k].trade_data
            train_labels[train_sizes_idx[k]:train_sizes_idx[k + 1]] = stocks[k].train_labels
            test_labels[test_sizes_idx[k]:test_sizes_idx[k + 1]] = stocks[k].test_labels
            trade_labels[trade_sizes_idx[k]:trade_sizes_idx[k + 1]] = stocks[k].trade_labels

    train_labels_og = train_labels
    test_labels_og = test_labels
    trade_labels_og = trade_labels
    buy_mask = np.nonzero(train_labels_og == 1)[0]
    sell_mask = np.nonzero(train_labels_og == 0)[0]
    keep_idx = random.sample(list(sell_mask), len(buy_mask))
    keep = np.hstack((keep_idx, buy_mask))

    # train_labels = to_categorical(train_labels[keep, :])
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    # train_data = train_data[keep, :]

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
    # model.add(Dense(n_inputs * 2, activation='relu')s)
    # model.add(Dense(n_inputs * 20, activation='relu'))
    # model.add(Dense(n_inputs * 2, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(n_outputs))
    omt = keras.optimizers.Adam(lr=0.00005)
    loss = 'binary_crossentropy'
    model.compile(loss=loss, optimizer=omt, metrics=['binary_accuracy'])
    es = EarlyStopping(monitor='val_loss', patience=10)
    history = model.fit(train_data, train_labels, epochs=500,
        batch_size=int(np.round((len(test_labels) / 100))), verbose=True, shuffle=False,
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
    return model


def make_2d_cnn(train, test, load_nn_model=0, model_name=''):
    train_data = train[0]
    train_labels = train[1]
    test_data = test[0]
    test_labels = test[1]
    train_data = np.expand_dims(train_data, axis=3)
    test_data = np.expand_dims(test_data, axis=3)
    n_inputs = train_data.shape[1]
    n_outputs = 2
    model = Sequential()
    model.add(
        Conv2D(4, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=(30, 33, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(4, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(n_outputs, activation='softmax'))
    omt = keras.optimizers.Adam(lr=0.00005)
    loss = 'binary_crossentropy'
    model.compile(loss=loss, optimizer=omt, metrics=['binary_accuracy'])
    es = EarlyStopping(monitor='val_loss', patience=10)
    history = model.fit(train_data, train_labels, epochs=2, batch_size=128, verbose=True,
        shuffle=False, validation_data=(test_data, test_labels), callbacks=[es])
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
    return model


def test_strat(stocks, model, cnn=False):
    split_percent = 0.85
    split_percent2 = 0.95
    col_data = col_data = stocks[0].metrics_column_data
    # to_show2 = random.sample(range(len(stocks)), 5)
    stats_names = ['Stock', 'APG', 'WP', 'NumTrades', 'AvgHoldLen']
    tkrs = []
    stats2 = np.zeros((0, 4))
    for k in range(len(stocks)):
        if stocks[k].ticker[0] != 'No File':
            split_idx1 = round(split_percent * len(stocks[k].metrics[:, 0]))
            split_idx2 = round(split_percent2 * len(stocks[k].metrics[:, 0]))
            # prd = model.predict_classes(stocks[k].trade_data[:,[0,1,2,3,4]])
            if cnn:
                prd = model.predict_classes(np.expand_dims(stocks[k].trade_data_tensor, axis=3))
            else:
                prd = model.predict_classes(stocks[k].trade_data)
            buy_price = np.array([])
            sell_price = np.array([])
            buy_day = np.array([])
            sell_day = np.array([])
            i = -1
            f_day = len(stocks[k].metrics[:, 0]) - len(prd)
            ts = stocks[k]
            pg = 2
            open_idx = col_data['open']
            close_idx = col_data['close']
            while i < len(prd) - 2:
                i += 1
                if prd[i] == 1:
                    i += 1
                    buy_price = np.append(buy_price, ts.metrics[f_day + i, open_idx])
                    buy_day = np.append(buy_day, f_day + i)
                    pc = 100 * (ts.metrics[f_day + i, close_idx] - buy_price[-1]) / buy_price[-1]
                    if pc > pg:
                        sell_price = np.append(sell_price, ts.metrics[f_day + i, close_idx])
                        sell_day = np.append(sell_day, f_day + i)
                        i -= 1
                    elif pc < -pg:
                        sell_price = np.append(sell_price, ts.metrics[f_day + i, close_idx])
                        sell_day = np.append(sell_day, f_day + i)
                        i -= 1
                    while abs(pc) < pg and i < len(prd) - 1:
                        i += 1
                        pc = 100 * (ts.metrics[f_day + i, close_idx] - buy_price[-1]) / buy_price[
                            -1]
                        if pc > pg:
                            sell_price = np.append(sell_price, ts.metrics[f_day + i, close_idx])
                            sell_day = np.append(sell_day, f_day + i)
                            i -= 1
                        elif pc < -pg:
                            sell_price = np.append(sell_price, ts.metrics[f_day + i, close_idx])
                            sell_day = np.append(sell_day, f_day + i)
                            i -= 1
                            break
            if len(sell_price) != len(buy_price):
                a = (sell_price[:] - buy_price[:len(sell_price)]) / buy_price[:len(sell_price)]
                ahl = np.round(np.mean(sell_day - buy_day[:len(sell_price)]), 3)
            else:
                a = (sell_price[:] - buy_price[:]) / buy_price[:]
                ahl = np.round(np.mean(sell_day - buy_day), 3)
            if len(a) != 0:
                print(2 * '\n' + str(stocks[k].ticker[0]) + ' Win Percent :' + str(
                    np.round(100 * len(np.nonzero(a > 0)[0]) / len(a), 2)))
                apg = np.round((100 * np.mean(a)), 2)
                wp = np.round(100 * len(np.nonzero(a > 0)[0]) / len(a), 2)
                nt = len(sell_day)
                stocks[k].stats2 = np.array((apg, wp, nt, ahl))
                stats2 = np.vstack((stats2, stocks[k].stats2))
                tkrs.append(stocks[k].ticker[0])
            else:
                apg = 0
                wp = 0
                nt = 0
            print(str(stocks[k].ticker[0]) + ' Average Percent Change :' + str(
                np.round((100 * np.mean(a)), 2)))
            print(str(stocks[k].ticker[0]) + ' Number of Trades :' + str(nt))
            print(str(stocks[k].ticker[0]) + ' Average Hold Time :' + str(ahl) + 2 * '\n')
            # if stocks[k].ticker[0] == 'QD':
            if True:
                # if 1==0:
                # if 1==0:
                # if len(buy_day)>5 and apg>0
                # if buy_day[-1]==len(stocks[k].metrics[:, 0])-1:
                plt.figure()
                ax1 = plt.subplot(3, 1, 1)
                high_idx = col_data['high']
                low_idx = col_data['low']
                test_open = stocks[k].metrics[split_idx2:, open_idx]
                test_low = stocks[k].metrics[split_idx2:, low_idx]
                test_high = stocks[k].metrics[split_idx2:, high_idx]
                test_close = stocks[k].metrics[split_idx2:, close_idx]
                candle(ax1, test_open, test_high, test_low, test_close, colorup="green",
                    colordown="red", width=.4)
                plt.plot(buy_day - f_day, ts.metrics[buy_day.astype(int), open_idx], 'g.',
                    markersize=10)
                plt.plot(sell_day - f_day, ts.metrics[sell_day.astype(int), close_idx], 'r.',
                    markersize=10)
                plt.title(stocks[k].ticker[0][:] + '  APG: ' + str(apg) + ' WP: ' + str(
                    wp) + ' NumTrades: ' + str(nt) + ' AHL: ' + str(ahl))
                ax2 = plt.subplot(3, 1, 2, sharex=ax1)
                line2 = plt.plot(np.arange(len(stocks[k].trade_labels)),
                    stocks[k].trade_labels[:, 0], 'k')
                ax3 = plt.subplot(3, 1, 3, sharex=ax1)
                line3 = plt.plot(np.arange(len(prd)), prd, 'k')
    ljn = 13
    print('Avg. % Gain'.ljust(ljn) + ' | ', 'Win %'.ljust(ljn) + ' | ',
          '# Trades'.ljust(ljn) + ' | ', 'Avg. Hold'.ljust(12))
    stat_print = [str(k).ljust(ljn) for k in np.round(np.mean(stats2, 0), 3)]
    print(' |  '.join(stat_print))
    plt.show()
