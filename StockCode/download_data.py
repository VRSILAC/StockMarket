import re
import os
import time
import requests
import datetime
import numpy as np


def split_crumb_store(v):
    if v is None:
        return
    return v.split(':')[2].strip('"')


def find_crumb_store(lines):
    for l in lines:
        if re.findall(r'CrumbStore', l):
            return l
    # print("Did not find CrumbStore")


def get_cookie_value(r):
    if not r.cookies:
        return
        # return print('No Cookie')
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


def get_data(symbol, start_date, end_date, cookie, crumb, append_to_file):
    filename = csv_location + '%s.csv' % (symbol)
    url = "https://query1.finance.yahoo.com/v7/finance/download/%s?period1=%s&period2=%s&interval=1d&events=history&crumb=%s" % (
        symbol, start_date, end_date, crumb)
    response = requests.get(url, cookies=cookie, timeout=10)
    block = response.content[:1].decode('UTF-8')
    if block == '{' or block == '4':
        return False
    if append_to_file:
        for block in response.iter_content(1024):
            pass
        with open(filename, 'r') as open_file:
            new_handle = bytes('\n'.join(open_file.read().split('\n')[:-3]) + '\n', 'utf-8')
        with open(filename, 'wb') as new_csv:
            new_csv.write(new_handle)
            new_csv.write(block[42:])
            return True
    if not append_to_file:
        block = response.content[:1].decode('UTF-8')
        if block == '{' or block == '4':
            yeet = 1
            return False
        with open(filename, 'wb') as handle:
            for block in response.iter_content(1024):
                handle.write(block)
            return True
    return False


def dq(symbol, verbose=False):
    waitbar(len(open(list_location + ticker_list[0], 'r').read().split('\n')),
            len(open(list_location + 'completed_list.txt', 'r').read().split('\n')))
    csv_present = os.listdir(csv_location)
    filename = csv_location + '%s.csv' % (symbol)
    present = symbol + '.csv' in csv_present
    if present:
        if os.path.getsize(filename) < 1000:
            present = False
            os.remove(filename)
    end_date = get_now_epoch()
    cookie, crumb = get_cookie_crumb(symbol)
    if verbose:
        print("--------------------------------------------------")
        print("Downloading %s to %s.csv" % (symbol, symbol))
    if not present:
        append_to_file = False
        start_date = 0
    else:
        append_to_file = True
        last_time = open(csv_location + symbol + '.csv', 'r').read().split('\n')[-3].split(',')[0]
        if '}' in last_time:
            os.remove(filename)
            start_date = 0
            append_to_file = False
        else:
            start_date = int(datetime.datetime.timestamp(datetime.datetime.strptime(last_time, "%Y-%m-%d")))
    data_saved = False
    attempts = 0
    while attempts < 5 and not data_saved:
        data_saved = get_data(symbol, start_date, end_date, cookie, crumb, append_to_file)
        if data_saved == False:
            cookie, crumb = get_cookie_crumb(symbol)
        attempts += 1
    if data_saved:
        if verbose: print(symbol + ' download successful')
        with open(list_location + 'completed_list.txt', 'a') as complete:
            complete.write('\n' + symbol)
    else:
        if verbose: print(symbol + ' download failed')
        with open(list_location + 'failed_list.txt', 'a') as failed:
            failed.write('\n' + symbol)


def gather_tickers(ticker_list):
    tickers = open(ticker_list, 'r')
    tickers = tickers.read()
    tickers = tickers.split('\n')
    return tickers


def download_parallel_quotes(symbols):
    from functools import partial
    import multiprocessing
    with open(list_location + 'completed_list.txt', 'w') as complete:
        complete.write('')
    with open(list_location + 'failed_list.txt', 'w') as failed:
        failed.write('')
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    dfunc = partial(dq, verbose=True)
    output = pool.map(dfunc, symbols)


def download_quotes(symbols):
    for symbol in symbols:
        dq(symbol)


ticker_list = ['tickers.txt', 'test_list.txt']
list_location = '/home/carmelo/Documents/StockMarket/Ticker_Lists/'
csv_location = '/home/carmelo/Documents/StockMarket/CSVFiles/'
download_list = ticker_list[0]
tickers = gather_tickers(list_location + download_list)[:-1]
download_parallel_quotes(tickers)
