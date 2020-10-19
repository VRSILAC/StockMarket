import re
import os
import time
import random
import pickle
import requests
import datetime
import numpy as np
import joblib

csv_location = '/home/carmelo/Documents/StockMarket/CSVFiles/'


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
            # start_date = 1585724400
            append_to_file = 0
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


def dq(symbol_in):
    csv_present = os.listdir(csv_location)
    symbol = symbol_in[0]
    if symbol + '.csv' not in csv_present:
        start_date = 0
        # start_date = 1585724400
        append_to_file = 0
        print("--------------------------------------------------")
        print("Downloading %s to %s.csv" % (symbol, symbol))
        #waitbar(num_symbols, i)
        end_date = get_now_epoch()
        cookie, crumb = get_cookie_crumb(symbol)
        tries = 0
        tries = get_data(symbol, start_date, end_date, cookie, crumb, append_to_file, tries)
        filename = csv_location + '%s.csv' % (symbol)
        if tries >= 5:
            os.remove(filename)


def download_parallel_quotes(symbols):
    import multiprocessing
    pool = multiprocessing.Pool(processes=5)
    output = pool.map(dq, symbols)


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
            idx = stocks[i].names['gap']
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
