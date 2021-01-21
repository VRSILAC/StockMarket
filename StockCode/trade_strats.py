from multiprocessing import Pool
import matplotlib.pyplot as plt
from matplotlib import gridspec
from process_data import *
from candle import *
import numpy as np
import warnings
import random
import traceback

old_settings = np.seterr(all='ignore')
np.seterr(invalid='ignore')
warnings.filterwarnings('ignore')


# np.set_printoptions(suppress=True)

def compute_stats(buy_day, sell_day, buy_price, sell_price, max_pc):
    try:
        buy_day = np.array(buy_day)
        sell_day = np.array(sell_day)
        buy_price = np.array(buy_price)
        sell_price = np.array(sell_price)
        apg_ = 100 * (sell_price - buy_price[:len(sell_price)]) / buy_price[:len(sell_price)]
        apg_ = np.array([a if a<max_pc else max_pc for a in apg_])
        apg = np.round(np.mean(apg_), 2)
        wp = np.round(len(np.nonzero(apg_ > 0)[0]) / len(buy_day), 2)
        ahl = np.round(np.mean(sell_day - buy_day[:len(sell_day)]), 1)
        history = [compute_profit(apg_), compute_profit(random.sample(list(apg_), len(apg_)))]
    except Exception:
        apg_ = apg = wp = history = ahl = 0
    return apg_, apg, wp, history, ahl


def compute_profit(apg_):
    profit = 0
    account_history = [1000]
    history = []
    for k in apg_:
        if account_history[-1] < 500:
            profit += -500
            history[-1] = history[-1] - 500
            account_history = [1000]
        investment = 1000 if account_history[-1] > 1000 else account_history[-1]
        trade_profit = (investment) * (k / 100)
        profit += trade_profit
        history.append(profit)
        account_history.append(account_history[-1] + trade_profit)

    return history


def get_conditions(stock, conditions, day):
    condition = []
    for k in range(len(conditions)):
        metric = conditions[k][0]
        logic = conditions[k][1]
        value = conditions[k][2]
        back = conditions[k][3]
        if logic == 'lt':
            b_val = stock.metrics[day - back, stock.names[metric]] < value
            condition.append(b_val)
        elif logic == 'le':
            b_val = stock.metrics[day - back, stock.names[metric]] <= value
            condition.append(b_val)
        elif logic == 'eq':
            b_val = stock.metrics[day - back, stock.names[metric]] == value
            condition.append(b_val)
        elif logic == 'ne':
            b_val = stock.metrics[day - back, stock.names[metric]] != value
            condition.append(b_val)
        elif logic == 'ge':
            b_val = stock.metrics[day - back, stock.names[metric]] >= value
            condition.append(b_val)
        elif logic == 'gt':
            b_val = stock.metrics[day - back, stock.names[metric]] > value
            condition.append(b_val)
        else:
            print('Logic Not Supported')
    return condition


def plot_from_strat(buy_days, sell_days, buy_prices, sell_prices, stocks, stocks_traded_idx, max_pc):
    plot_random = random.sample(stocks_traded_idx, 5) if len(stocks_traded_idx) > 5 else random.sample(
        stocks_traded_idx, len(stocks_traded_idx))
    for idx in plot_random:
        stock = stocks[idx]
        buy_day = buy_days[idx]
        sell_day = sell_days[idx]
        buy_price = buy_prices[idx]
        sell_price = sell_prices[idx]
        apg_, apg, wp, history, ahl = compute_stats(buy_day, sell_day, buy_price, sell_price, max_pc)
        if idx in plot_random:
            look_back = len(stock.metrics[:, 0])
            fig = plt.figure(figsize=(12, 6))
            fig.patch.set_facecolor('xkcd:gray')
            gs = gridspec.GridSpec(4, 1)
            ax1 = plt.subplot(gs[:4])
            ax1.set_facecolor('xkcd:gray')
            open = stock.metrics[-look_back:, 0]
            high = stock.metrics[-look_back:, 1]
            low = stock.metrics[-look_back:, 2]
            close = stock.metrics[-look_back:, 3]
            candle(ax1, open, high, low, close, colorup="cyan", colordown="red", width=.4)
            plt.plot(np.array(buy_day), np.array(buy_price), 'g.', markersize='15')
            plt.plot(np.array(sell_day), np.array(sell_price), 'r.', markersize='15')
            try:
                ax1.set_title(stock.ticker[0][:] + '  APG: ' + str(apg) + ' WP: ' + str(wp) + ' NumTrades: ' + str(
                    len(buy_day)) + ' AHL: ' + str(ahl) + ' $' + str('{:.2f}'.format(history[0][-1])))
            except Exception:
                ax1.set_title('Current Trade: ' + stock.ticker[0][:])
            plt.subplots_adjust(hspace=0.4)
    plt.show()


# Excellent Long Strategy for Covid Market Conditions.
def get_custom_condition(stock, day):
    c0 = np.mean(np.abs(stock.metrics[day - 10:day, stock.names['percent_change']])) > 0.05
    c1 = stock.metrics[day, stock.names['d2price']] < 0
    c2 = stock.metrics[day, stock.names['d2ma5']] < 0
    c3 = stock.metrics[day, stock.names['d3ma5']] > 0
    c4 = stock.metrics[day, stock.names['d2ma20']] < 0
    c5 = stock.metrics[day, stock.names['percent_change']] < 0
    c6 = stock.metrics[day + 1, stock.names['gap']] < 0
    c7 = stock.metrics[day + 1, stock.names['gap']] > -0.1
    c8 = stock.metrics[day, stock.names['open']] < 10

    return [c1, c2, c3, c4, c5, c6, c7, c8, c0]


# def get_custom_condition(stock, day):
#     opens = stock.metrics[day - 15:day, stock.names['open']]
#     close = stock.metrics[day, stock.names['close']]
#     pc1 = (close - opens) / opens
#     avg_pc1 = np.mean(np.abs(pc1))
#     avg_pc1_con = -0.3  # * avg_pc1
#     c1 = any(pc1 < avg_pc1_con)
#     c2 = stock.metrics[day, stock.names['gap']] > 0
#     c3 = stock.metrics[day, stock.names['percent_change']] < 0
#     c4 = stock.metrics[day, stock.names['gap']] < 0.1
#     c5 = avg_pc1 > 0.05
#     c6 = stock.metrics[day + 1, stock.names['gap']] >= 0
#     c7 = stock.moving_average(10, stock.metrics[day - 10:day, stock.names['volume']])[-1] > 1E6
#     c8 = stock.metrics[day + 1, stock.names['gap']] < 0.1
#     ##
#     c9 = stock.metrics[day, stock.names['open']] < 10
#     c10 = stock.metrics[day, stock.names['open']] > 0.05
#     ##
#     # c11 = stock.metrics[day, stock.names['ma5']] < stock.metrics[day, stock.names['ma50']]
#     # c12 = any(stock.metrics[day-15:day, stock.names['gap']] < -avg_pc1)
#     # c11 = np.mean(np.abs(pc1[-3:])) < avg_pc1
#     return [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10]


def get_ohlc(stock, day):
    open = stock.metrics[day, stock.names['open']]
    high = stock.metrics[day, stock.names['high']]
    low = stock.metrics[day, stock.names['low']]
    close = stock.metrics[day, stock.names['close']]
    return open, high, low, close


def swing_(stocks, day_trade=False, hard_stop=True, model_type='Custom', model=False, days_back=20):
    max_pc = 20
    stats = []
    tickers = []
    buy_days = []
    sell_days = []
    buy_prices = []
    sell_prices = []
    current_trades = []
    current_trades_idx = []
    stocks_traded_idx = []
    total = len(stocks)
    for idx, stock in enumerate(stocks):
        waitbar(total, idx)
        if model_type == 'ANN':
            look_back = len(stock.trade_data[:, 0])
            stock.metrics = stock.metrics[-look_back:, :]
        elif model_type == '2DCNN':
            look_back = len(stock.trade_data_tensor[:, 0])
            stock.metrics = stock.metrics[-look_back:, :]
        else:
            look_back = len(stock.metrics[:, 0])
        day = -look_back + 11 if days_back == -1 else -days_back
        buy_day = []
        sell_day = []
        buy_price = []
        sell_price = []
        while day < -1:
            pct_range = np.mean(np.abs(stock.metrics[day - 10:day, stock.names['percent_change']]))
            take_gain_percent = np.min([1 + pct_range * 3, 1.2])
            stop_loss_percent = np.max([1 - pct_range * 2, 0.9])
            stop_loss_percent2 = np.max([1 - pct_range * 2, 0.9])
            if model_type == '2DCNN':
                prd = model.predict_classes(
                    np.expand_dims(np.expand_dims(stock.trade_data_tensor[day, :, :], axis=2), axis=0))
                condition = [True] if prd == 1 else [False]
            elif model_type == 'ANN':
                prd = model.predict_classes(np.expand_dims(stock.trade_data[day, :], axis=0))
                condition = [True] if prd == 1 else [False]
            else:
                condition = get_custom_condition(stock, day)
            if all(condition):
                # if np.count_nonzero(condition) >= len(condition) - 1:
                day += 1
                open_, high, low, close = get_ohlc(stock, day)
                buy_day.append(look_back + day)
                buy_price.append(open_)
                try:
                    hard_stop_val = np.min(stock.metrics[day - 10:day, stock.names['low']])
                except:
                    hard_stop_val = np.min(stock.metrics[day - 10:, stock.names['low']])
                stop_loss = hard_stop_val if hard_stop else stop_loss_percent * buy_price[-1]
                if low < stop_loss:
                    sell_price.append(stop_loss)
                    sell_day.append(look_back + day)
                elif high >= take_gain_percent * buy_price[-1]:
                    sell_price.append(take_gain_percent * buy_price[-1])
                    sell_day.append(look_back + day)
                elif day_trade:
                    sell_price.append(close)
                    sell_day.append(look_back + day)
                else:
                    current_high = np.max([open_, close])
                    stop_loss = hard_stop_val if hard_stop else stop_loss_percent2 * current_high
                    while True:
                        day += 1
                        open_, high, low, close = get_ohlc(stock, day)
                        if not day < -0:
                            current_trades_idx.append(idx)
                            break
                        # condition = get_custom_condition(stock, day)
                        if open_ <= stop_loss:
                            sell_price.append(open_)
                            sell_day.append(look_back + day)
                            break
                        if open_ >= buy_price[-1] * take_gain_percent:
                            sell_price.append(open_)
                            sell_day.append(look_back + day)
                            break
                        if low <= stop_loss:
                            sell_price.append(stop_loss)
                            sell_day.append(look_back + day)
                            break
                        if high >= buy_price[-1] * take_gain_percent:
                            sell_price.append(buy_price[-1] * take_gain_percent)
                            sell_day.append(look_back + day)
                            break
                        if open_ > current_high:
                            current_high = open_
                        if close > current_high:
                            current_high = close
                        stop_loss = hard_stop_val if hard_stop else stop_loss_percent2 * current_high
            else:
                day += 1
        buy_days.append(buy_day)
        sell_days.append(sell_day)
        buy_prices.append(buy_price)
        sell_prices.append(sell_price)
        apg_, apg, wp, history, ahl = compute_stats(buy_day, sell_day, buy_price, sell_price, max_pc)
        try:
            if all(apg_ < 100):
                stats.append([apg, wp, len(buy_day), ahl, np.round(history[0][-1], 2)])
            else:
                stats.append([0, 0, 0, 0, 0])
        except Exception:
            stats.append([0, 0, 0, 0, 0])
        tickers.append(stock.ticker[0][:])
        if len(sell_day) > 0:
            stocks_traded_idx.append(idx)
    current_trades = [stocks[cti].ticker[0] + '  ' for cti in current_trades_idx]
    print_stats(stats, tickers)
    print('Current Trades: ' + ''.join(current_trades))
    plot_from_strat(buy_days, sell_days, buy_prices, sell_prices, stocks, stocks_traded_idx, max_pc)
    # plot_from_strat(buy_days, sell_days, buy_prices, sell_prices, stocks, current_trades_idx, max_pc)
    return buy_days, sell_days, buy_prices, sell_prices, stats


def print_stats(stats, tickers):
    stats = np.array(stats)
    zero_mask = np.all(stats == 0, axis=1)
    stats = stats[~np.all(stats == 0, axis=1)]
    tickers = [ticker for idx, ticker in enumerate(tickers) if zero_mask[idx] == False]
    ljn = 8
    print('Ticker'.ljust(ljn) + ' | ', 'APG'.ljust(ljn) + ' | ', 'Win %'.ljust(ljn) + ' | ',
          '# Trades'.ljust(ljn) + ' | ', 'AH'.ljust(ljn) + ' | ', 'Profit'.ljust(ljn))
    for k in range(len(tickers)):
        stat_print = [tickers[k].ljust(ljn)]
        stat_print.extend([str(k).ljust(ljn) for k in stats[k, :]])
        print(' |  '.join(stat_print))
    stat_print = ['Avgs'.ljust(ljn)]
    stat_print.extend([str(k).ljust(ljn) for k in list(np.round(np.mean(stats, 0), 2))])
    print(' |  '.join(stat_print))
    print('Ticker'.ljust(ljn) + ' | ', 'APG'.ljust(ljn) + ' | ', 'Win %'.ljust(ljn) + ' | ',
          '# Trades'.ljust(ljn) + ' | ', 'AH'.ljust(ljn) + ' | ', 'Profit'.ljust(ljn))
    print('Total Number Trades: ' + str(np.sum(stats[:, 2])))
