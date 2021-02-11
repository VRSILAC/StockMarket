from selenium.webdriver.common.keys import Keys
from selenium import webdriver
from tda.auth import easy_client
from tda.client import Client
from trade_strats import *
from process_data import *
from config import *
import pandas as pd
import traceback
import datetime
import pickle
import tda


# driver_path = '/home/carmelo/Documents/StockMarket/StockCode/TOS/chromedriver_linux64/chromedriver'
# browser = webdriver.Chrome(driver_path)
def get_day_params():
    today = datetime.datetime.now()
    return [today.year, today.month, today.day, today.hour, today.minute]


def waitbar_(current, total):
    current += 1
    done = int(np.round(((100 * (current / total)) / 2) - 1))
    togo = int(np.round(((100 * ((total - current) / total)) / 2)))
    per = str(np.round(100 * current / total, 1))
    print(done * '-' + '>' + togo * '.' + per + '%', end='\r')


def do_sleep(min, max):
    time_now = time.time()
    sleep_for = np.random.randint(min, max, 1)[0]
    waited_time = time.time() - time_now
    print('\n')
    while waited_time < sleep_for:
        time.sleep(1)
        waited_time = time.time() - time_now
        waitbar_(waited_time, sleep_for)


def update_positions(p):
    file = open('positions.pickle', 'wb')
    pickle.dump(p, file)
    file.close()
    return p


def set_positions(d):
    if os.path.exists('positions.pickle'):
        p = get_positions()
    else:
        p = {}
    file = open('positions.pickle', 'wb')
    ticker = list(d.keys())[0]
    p[ticker] = d[ticker]
    pickle.dump(p, file)
    file.close()


def get_positions():
    file = open('positions.pickle', 'rb')
    positions = pickle.load(file)
    file.close()
    return positions


def gen_watchlist(model):
    stocks = load_stocks('/home/carmelo/Documents/StockMarket/StockData/stocks_100d.obj')
    return get_watchlist(stocks, model, save_close=True)


def get_account_deets(c, funds_per_trade):
    account = c.get_account(account_id)
    account = account.json()['securitiesAccount']
    round_trips = account['roundTrips']
    available_funds = account['currentBalances']['availableFunds']
    num_trades_available = int(available_funds // funds_per_trade)
    return account, round_trips, available_funds, num_trades_available


def get_high_last(c, tickers):
    quotes = c.get_quotes(tickers)
    quotes = quotes.json()
    last_price = [quotes[ticker]["lastPrice"] for ticker in tickers]
    high_price = [quotes[ticker]["highPrice"] for ticker in tickers]
    hl = {}
    for idx, ticker in enumerate(tickers):
        hl[ticker] = {'last_price': last_price[idx], 'high_price': high_price[idx]}
    return hl


def get_buy_list(c, model, funds_per_trade, buy_dict={}):
    watchlist = gen_watchlist(model)
    if os.path.exists('positions.pickle'):
        p = get_positions()
        p = [k for k in p.keys()]
        watchlist = [w for w in watchlist if w[0] not in p]
    tickers = [w[0] for w in watchlist]
    close_price = [c[1] for c in watchlist]
    premarket = c.get_quotes(tickers)
    premarket = premarket.json()
    open_ = [premarket[ticker]["mark"] for ticker in tickers]
    gap = np.array(open_) / np.array(close_price) - 1
    for idx, g in enumerate(gap):
        if model == 'SO':
            condition = g >= 0 and g < 0.1
        elif model == 'DTS':
            condition = g < 0 and g > -0.1
        else:
            return print('Model not supported')
        if condition:
            buy_price = premarket[tickers[idx]]["openPrice"]
            if model == 'DTS':
                slp = 0.9
                sell_price = np.round(buy_price * 1.1, 2)
                stop_loss_0 = np.round(buy_price * slp, 2)
                stop_loss_1 = np.round(buy_price * slp, 2)
            elif model == 'SO':
                slp = 0.85
                sell_price = np.round(buy_price * 1.2, 2)
                stop_loss_0 = np.round(buy_price * slp, 2)
                stop_loss_1 = np.round(buy_price * slp, 2)
            shares = int(funds_per_trade / buy_price)
            buy_dict[tickers[idx]] = {'ticker': tickers[idx], 'buy_price': buy_price, 'shares': shares,
                                      'buy_status': 'POSSIBLE', 'sell_status': '', 'sell_price': sell_price,
                                      'stop_loss_0': stop_loss_0,
                                      'stop_loss_1': stop_loss_1, 'buy_order_id': 0, 'sell_order_id': 0,
                                      'high_price': buy_price, 'last_price': buy_price,
                                      'stop_loss_percent': slp, 'days_held': 0, 'watchlist_day': watchlist[idx][-1],
                                      'purchase_date': get_day_params()}
    return buy_dict


def sell(c, stock, p):
    b = p[stock]
    spec = tda.orders.equities.equity_sell_limit(stock, b['shares'], b['sell_price'])
    sell_order = c.place_order(account_id, spec)
    order_id = tda.utils.Utils(c, account_id).extract_order_id(sell_order)
    if order_id is not None:
        p[stock]['sell_order_id'] = order_id
        p[stock]['sell_status'] = 'WAITING_TO_SELL'
    return p


def stop_loss_sell(c, stock, p, stop_loss):
    b = p[stock]
    spec = tda.orders.equities.equity_sell_limit(stock, b['shares'], b[stop_loss])
    sell_order = c.place_order(account_id, spec)
    order_id = tda.utils.Utils(c, account_id).extract_order_id(sell_order)
    if order_id is not None:
        p[stock]['sell_order_id'] = order_id
        p[stock]['sell_status'] = 'WAITING_TO_SELL_STOP_LOSS'
    return p


def buy(c, buy_dict, buy_today):
    for stock in buy_today:
        day = get_day_params()
        if buy_dict[stock]['watchlist_day'][1] + 10 + buy_dict[stock]['watchlist_day'][2] < day[1] + 10 + day[2]:
            b = buy_dict[stock]
            spec = tda.orders.equities.equity_buy_limit(stock, b['shares'], b['buy_price'])
            buy_order = c.place_order(account_id, spec)
            order_id = tda.utils.Utils(c, account_id).extract_order_id(buy_order)
            if order_id is not None:
                buy_dict[stock]['buy_order_id'] = order_id
                order = c.get_order(order_id, account_id).json()
                buy_dict[stock]['buy_status'] = order['status']
                set_positions({stock: buy_dict[stock]})
    positions = get_positions()
    return positions


def update_fill_price(c, p):
    order = c.get_order(p['buy_order_id'], account_id).json()
    p['actual_buy_price'] = order['orderActivityCollection'][0]['executionLegs'][0]['price']
    return p


def monitor(c, p):
    pop_ticker = []
    for stock in p.keys():
        p[stock]['days_held'] = get_day_params()[2] - p[stock]['purchase_date'][2]
        b_order = c.get_order(p[stock]['buy_order_id'], account_id).json()
        if p[stock]['buy_order_id'] != 0 and p[stock]['buy_status'] != 'FILLED':
            p[stock]['buy_status'] = b_order['status']
        if p[stock]['buy_order_id'] == 'CANCELED':
            pop_ticker.append(stock)
        if p[stock]['sell_order_id'] != 0:
            s_order = c.get_order(p[stock]['sell_order_id'], account_id).json()
            if s_order['status'] == 'FILLED':
                pop_ticker.append(stock)
    if len(pop_ticker) > 0:
        for closed in pop_ticker:
            p.pop(closed)
        p = update_positions(p)
    tickers = list(p.keys())
    hl = get_high_last(c, tickers)
    for stock in p.keys():
        high_price = hl[stock]['high_price']
        last_price = hl[stock]['last_price']
        p[stock]['last_price'] = last_price
        if p[stock]['high_price'] < high_price:
            p[stock]['high_price'] = hl[stock]['high_price']
            p[stock]['stop_loss_1'] = np.round(high_price * p[stock]['stop_loss_percent'], 2)
        if p[stock]['days_held'] > 0:
            if last_price - p[stock]['stop_loss_1'] < 0.01:
                c.cancel_order(account_id, p[stock]['sell_order_id'])
                p = stop_loss_sell(c, stock, p, 'stop_loss_1')
        else:
            if last_price - p[stock]['stop_loss_0'] < 0.01:
                c.cancel_order(account_id, p[stock]['sell_order_id'])
                p = stop_loss_sell(c, stock, p, 'stop_loss_0')
    p = update_positions(p)
    return p


def main():
    c = easy_client(api_key=client_id,
                    redirect_uri=redirect_uri,
                    token_path='/home/carmelo/Documents/StockMarket/StockCode/TOS/token.pickle')
    fpt = 300
    buy_dict = get_buy_list(c, 'SO', fpt)
    if os.path.exists('positions.pickle'): p = get_positions()
    while True:
        try:
            day = get_day_params()
            account, round_trips, available_funds, num_trades_available = get_account_deets(c, funds_per_trade=fpt)
            day_condition = all([day[3] >= 6, day[4] > 30, day[3] < 13])
            # download_condition = all()
            if num_trades_available > 0 and len(buy_dict) > 0 and day_condition:
                buy_today = random.sample(list(buy_dict.keys()), num_trades_available)
                p = buy(c, buy_dict, buy_today)
            if os.path.exists('positions.pickle'):
                for stock in p.keys():
                    if p[stock]['buy_status'] == 'FILLED' and p[stock]['sell_order_id'] == 0:
                        p[stock] = update_fill_price(c, p[stock])
                        sell(c, stock, p)
                p = monitor(c, p)
                df = pd.DataFrame(p)
                pd.set_option('display.max_columns', None)
                print(df, end='\r')
        except Exception:
            print(traceback.format_exc())
            pass
        do_sleep(44, 45)


main()
########################################################
########################################################
########################################################
########################################################
# history = c.get_price_history('AAPL',
#                            period_type=Client.PriceHistory.PeriodType.YEAR,
#                            period=Client.PriceHistory.Period.TWENTY_YEARS,
#                            frequency_type=Client.PriceHistory.FrequencyType.DAILY,
#                            frequency=Client.PriceHistory.Frequency.DAILY)

# r = c.get_orders_by_query(from_entered_datetime=datetime.datetime(2021, 2, ))
# assert r.status_code == httpx.codes.OK, r.raise_for_status()
# print(r.json())
