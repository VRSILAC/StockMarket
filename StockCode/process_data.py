import re
import os
import time
import random
import pickle
import requests
import datetime
import numpy as np
import joblib
import traceback
import argparse


class StockClass:
    def __init__(self, symbol, csv_location):
        try:
            data = np.genfromtxt(csv_location + symbol + '.csv', delimiter=',', skip_header=1, converters={
                0: lambda s: (datetime.datetime.strptime(s.decode('ascii'), '%Y-%m-%d').timestamp())})
            self.ticker = [symbol]
            data = self.filter_out_nan(data)
            open_, high, low, close, volume = self.ohlcv(data)
            data_mean = data[:, 1:5].mean(1)
            gap = self.gap(open_, close, 1)
            percent_change = self.percent_change(open_, close, 0)
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
                open_, high, low, close, volume, data_mean, gap, percent_change, d1price, d2price,
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
            print('Adding Data For %s' % symbol)
        except Exception:
            print(traceback.format_exc())
            with open('/home/carmelo/Documents/StockMarket/Ticker_Lists/bad_ticker_list.txt', 'a') as bad_tickers:
                bad_tickers.write(symbol + '\n')
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

    def gap(self, open_, close, days):
        close_roll = np.roll(close, days, 0)
        gap = (open_ - close_roll) / close_roll
        gap[:days] = gap[days]
        return gap

    def moving_average(self, ma, data):
        csum = np.cumsum(np.insert(data, np.ones(ma + 1).astype('int'), 0))
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
        open_ = data[:, 1]
        open_[open_ == 0] = 0.001
        high = data[:, 2]
        high[high == 0] = 0.001
        low = data[:, 3]
        low[low == 0] = 0.001
        close = data[:, 4]
        close[close == 0] = 0.001
        volume = data[:, 6]
        volume[volume == 0] = 0.001
        return open_, high, low, close, volume

    def percent_change(self, open_, close, days):
        open_roll = np.roll(open_, days, 0)
        percent_change = (close - open_roll) / open_roll
        percent_change[:days] = percent_change[days]
        return percent_change


def parse_csv(symbols, csv_location):
    stock = []
    print(len(symbols))
    for i in range(len(symbols)):
        try:
            filename = csv_location + '%s.csv' % (symbols[i])
            if os.path.exists(filename):
                stock.append(StockClass(symbols[i], csv_location))
        except Exception:
            print(traceback.format_exc())
            print('file: %s.csv' % symbols[i], 'does not exist - In parse_csv')
    return stock


def clean_stock_list(stocks):
    stocks = [stock for stock in stocks if stock.ticker[0][:] != 'No File']
    return stocks


def gather_tickers(ticker_list):
    tickers = open(ticker_list, 'r')
    tickers = tickers.read()
    tickers = tickers.split('\n')
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


def waitbar(total, current):
    current += 1
    percent_complete = 100 * (current / total)
    here_sym = '>'
    complete_sym = '-'
    advance = str(int(np.round((percent_complete / 2) - 1)) * complete_sym + here_sym)
    retreat = str(int(np.round(((100 - percent_complete) / 2) - 1)) * '.')
    print(advance + retreat + ' ' + str(np.round(percent_complete, 3)) + '%', end='\r')


def parser():
    parser = argparse.ArgumentParser(description='Stock Market Ticker Downloader')
    parser.add_argument("--ticker_location",
                        default='/home/carmelo/Documents/StockMarket/TickerLists/tickers.txt',
                        help="path pointing to a list of tickers to download. must be from text file. tickers seperated by newline")
    parser.add_argument("--csv_location", default='/home/carmelo/Documents/StockMarket/CSVFiles/',
                        help="path pointing to location to save csv files, ex. /home/user/Desktop/CSVFiles/")
    parser.add_argument("--verbose", default=True, type=bool, help="print status of downloading or not")
    return parser.parse_args()


def check_arguments_errors(args):
    if not os.path.exists(args.csv_location):
        raise(ValueError("Invalid csv_location path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.ticker_location):
        raise(ValueError("Invalid ticker_location path {}".format(os.path.abspath(args.weights))))


def main():
    args = parser()
    check_arguments_errors(args)
    tickers = gather_tickers(args.ticker_location)[:-1]
    s = parse_csv(tickers, args.csv_location)
    s = clean_stock_list(s)
    save_stocks(s, 'stocks.obj')
    do_days = [500, 250, 100]
    for day in do_days:
        try:
            stocks = []
            t = []
            for stock in s:
                if stock.ticker[0] not in t and len(stock.metrics[:, 0]) > day:
                    stock.metrics = stock.metrics[-day:, :]
                    stocks.extend([stock])
                    t.append(stock.ticker[0])
                else:
                    if args.verbose: print('Duplicate')
            save_stocks(stocks, 'stocks_' + str(day) + 'd.obj')
        except Exception:
            print(traceback.format_exc())
            pass


if __name__ == '__main__':
    main()
