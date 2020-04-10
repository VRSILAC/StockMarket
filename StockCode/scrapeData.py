import re
import os
import time
import random
import pickle
import requests
import datetime
import numpy as np
from sklearn.externals import joblib

# import keras
# from keras.models import load_model
# from keras.utils import to_categorical
# from keras.callbacks import EarlyStopping
# from keras.models import Model, Sequential
# from keras.layers import Input, Dense, Dropout, Conv2D, BatchNormalization, MaxPooling2D, Flatten

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
    print(advance + retreat + ' ' + str(np.round(percent_complete, 3)) + '%', end='\r')


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
            open, high, low, close, volume = self.ohlcv(data)
            data_mean = data[:, 1:5].mean(1)
            gap = self.gap(open, close, 1)
            percent_change = self.percent_change(open, close, 0)
            d1price, d2price, d3price = self.derivatives(data_mean)
            d1volume, d2volume, d3volume = self.derivatives(volume)
            ma5 = self.moving_average(5, data_mean)
            d1ma5, d2ma5, d3ma5 = self.derivatives(ma5)
            ma10 = self.moving_average(10, data_mean)
            d1ma10, d2ma10, d3ma10 = self.derivatives(ma10)
            ma15 = self.moving_average(15, data_mean)
            d1ma15, d2ma15, d3ma15 = self.derivatives(ma15)
            ma20 = self.moving_average(20, data_mean)
            d1ma20, d2ma20, d3ma20 = self.derivatives(ma20)
            ma50 = self.moving_average(50, data_mean)
            d1ma50, d2ma50, d3ma50 = self.derivatives(ma50)
            ma100 = self.moving_average(100, data_mean)
            d1ma100, d2ma100, d3ma100 = self.derivatives(ma100)
            ma200 = self.moving_average(200, data_mean)
            d1ma200, d2ma200, d3ma200 = self.derivatives(ma200)
            # mb5, mb4, mb3, mb2, mb1 = (self.mbs(data[:, 1:5], data[:, 6], 30, 5)).T
            self.metrics = np.column_stack((
                open, high, low, close, volume, data_mean, gap, percent_change, d1price, d2price,
                d3price, d1volume, d2volume, d3volume, ma5, d1ma5, d2ma5, d3ma5, ma10, d1ma10,
                d2ma10, d3ma10, ma15, d1ma15, d2ma15, d3ma15, ma20, d1ma20, d2ma20, d3ma20, ma50,
                d1ma50, d2ma50, d3ma50, ma100, d1ma100, d2ma100, d3ma100, ma200, d1ma200, d2ma200,
                d3ma200))
            column_name_list = ['open', 'high', 'low', 'close', 'volume', 'data_mean', 'gap',
                                'percent_change', 'd1price', 'd2price', 'd3price', 'd1volume',
                                'd2volume', 'd3volume', 'ma5', 'd1ma5', 'd2ma5', 'd3ma5', 'ma10',
                                'd1ma10', 'd2ma10', 'd3ma10', 'ma15', 'd1ma15', 'd2ma15', 'd3ma15',
                                'ma20', 'd1ma20', 'd2ma20', 'd3ma20', 'ma50', 'd1ma50', 'd2ma50',
                                'd3ma50', 'ma100', 'd1ma100', 'd2ma100', 'd3ma100', 'ma200',
                                'd1ma200', 'd2ma200', 'd3ma200']
            self.names = dict([(name, i) for i, name in enumerate(column_name_list)])
            print('Adding Data For Stock %s' % symbol)
        except Exception:
            self.metrics = []
            self.ticker = ['No File']
            print('file: %s.csv' % symbol, 'can not add data  - In Class')

    def derivatives(self, data):
        roll_1 = np.roll(data, 1, 0)
        roll_2 = np.roll(data, 2, 0)
        roll_3 = np.roll(data, 3, 0)
        d1 = roll_2 - 4 * roll_1 + 3 * data
        d2 = roll_2 - 2 * roll_1 + data
        d3 = -roll_3 + 3 * roll_2 - 3 * roll_1 + data
        d1[0:3] = d1[3]
        d2[0:3] = d2[3]
        d3[0:3] = d3[3]
        return d1, d2, d3

    def flip_data(self, data):
        return data[::-1, :]

    def filter_out_nan(self, data):
        if self.ticker[:][0] != 'No File':
            for rr in range(data.shape[1]):
                mask = np.isnan(data[:, rr])
                data[mask, rr] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask),
                    data[~mask, rr])
            return data

    def gap(self, open, close, days):
        close_roll = np.roll(close, days, 0)
        gap = (open - close_roll) / close_roll
        gap[:days] = gap[days]
        return gap

    def moving_average(self, ma, data):
        csum = np.cumsum(np.insert(data, np.ones(ma + 1), 0))
        mavg = (csum[ma:] - csum[:-ma]) / float(ma)
        mavg = np.delete(mavg, 0)
        mavg[0] = mavg[1]
        return mavg

    def mbs(self, price_ohlc, volume, days, bars):
        price_ohlc = np.round(price_ohlc, 2)
        volume = np.ceil(volume * 100 / np.max(volume)).astype(np.int)
        monkey_bars = np.zeros((len(volume), bars))
        pv = []
        for k in range(len(volume)):
            pv.append(list(price_ohlc[k, :]) * volume[k])
        for k in np.arange(days, len(volume)):
            print(k, end='\r')
            all_data = np.hstack(pv[k - days:k])
            max = np.max(all_data)
            min = np.min(all_data)
            counts, bins = np.histogram(all_data, bins=list(np.arange(min, max, (max - min) / 200)))
            top_five_locations = np.argsort(counts)[-bars:]
            top_five_prices = bins[top_five_locations]
            monkey_bars[k, :] = top_five_prices
        monkey_bars[:30, :] = monkey_bars[30, :]
        return monkey_bars

    def ohlcv(self, data):
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
        return open, high, low, close, volume

    def percent_change(self, open, close, days):
        open_roll = np.roll(open, days, 0)
        percent_change = (close - open_roll) / open_roll
        percent_change[:days] = percent_change[days]
        return percent_change


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


def clean_stock_list(stocks):
    stocks = [stock for stock in stocks if stock.ticker[0][:] != 'No File']
    return stocks


def gather_tickers(ticker_list):
    tickers = open(ticker_list, 'r')
    tickers = tickers.read()
    tickers = tickers.split('\n')
    for i in range(len(tickers)):
        tickers[i] = tickers[i].split('\t')
    return tickers


def save_stocks(stocks, file):
    file = open(file, 'wb')
    joblib.dump(stocks, file)


def load_stocks(file):
    stocks = joblib.load(file)
    return stocks


def make_labels_percent_gain(stocks, label, logic, label_pg_crit=0):
    for i in range(len(stocks)):
        if stocks[i].ticker[0][:] != 'No File':
            idx = stocks[i].names['percent_change']
            if not hasattr(stocks[i], 'label_pg'):
                stocks[i].label_pg = np.zeros((len(stocks[i].metrics[:, 0]), 1))
                stocks[i].label_pg_actual = np.zeros((len(stocks[i].metrics[:, 0]), 1))
            if logic == 'lt':
                label_idx = np.nonzero(stocks[i].metrics[:, idx] < label_pg_crit)
            elif logic == 'le':
                label_idx = np.nonzero(stocks[i].metrics[:, idx] < label_pg_crit)
            elif logic == 'eq':
                label_idx = np.nonzero(stocks[i].metrics[:, idx] == label_pg_crit)
            elif logic == 'ne':
                label_idx = np.nonzero(stocks[i].metrics[:, idx] != label_pg_crit)
            elif logic == 'ge':
                label_idx = np.nonzero(stocks[i].metrics[:, idx] >= label_pg_crit)
            elif logic == 'gt':
                label_idx = np.nonzero(stocks[i].metrics[:, idx] > label_pg_crit)
            else:
                print('Logic Not Supported')
            stocks[i].label_pg[label_idx[0][:] - 1, :] = label
            stocks[i].label_pg_actual[label_idx[0][:], :] = label
    return stocks


def normalize_data(data):
    try:
        for i in range(data.shape[1]):
            data[:, i] = data[:, i] / np.max(np.abs(data[:, i]))
            # data[:, i] = data[:, i] - np.mean(data[:, i])
    except IndexError:
        data[:] = data[:] / np.max(np.abs(data[:]))
        # data[:] = data[:] - np.mean(data[:])
    return data


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
            # norm_except = [col_data['percent_change'], col_data['percent_change1'],
            #                col_data['percent_change2'], col_data['percent_change3'],
            #                col_data['percent_change4'], col_data['open_diff'],
            #                col_data['high_diff'], col_data['low_diff'], col_data['close_diff'],
            #                col_data['open_diff2'], col_data['high_diff2'], col_data['low_diff2'],
            #                col_data['close_diff2'], col_data['open_diff3'], col_data['high_diff3'],
            #                col_data['low_diff3'], col_data['close_diff3'], col_data['open_diff4'],
            #                col_data['high_diff4'], col_data['low_diff4'], col_data['close_diff4'],
            #                col_data['gap1'], col_data['gap2'], col_data['gap3'], col_data['gap4']]
            norm_these = np.delete(np.arange(len(col_data)), norm_except)
            stocks[k].train_data[:, norm_these] = normalize_data(
                stocks[k].train_data[:, norm_these])
            stocks[k].test_data[:, norm_these] = normalize_data(stocks[k].test_data[:, norm_these])
            stocks[k].trade_data[:, norm_these] = normalize_data(
                stocks[k].trade_data[:, norm_these])
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
    # model.add(Dropout(0.4))
    model.add(Dense(n_outputs))
    omt = keras.optimizers.Adam(lr=0.005)
    loss = 'binary_crossentropy'
    model.compile(loss=loss, optimizer=omt, metrics=['binary_accuracy'])
    es = EarlyStopping(monitor='val_loss', patience=10)
    history = model.fit(train_data, train_labels, epochs=500,
        batch_size=int(len(test_labels) * 0.01), verbose=True, shuffle=False,
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
    history = model.fit(train_data, train_labels, epochs=100,
        batch_size=int(len(test_labels) * 0.01), verbose=True, shuffle=False,
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
