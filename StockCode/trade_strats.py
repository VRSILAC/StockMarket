from multiprocessing import Pool
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scrapeData import *
from candle import *
import numpy as np
import warnings
import random
import traceback

old_settings = np.seterr(all='ignore')
np.seterr(invalid='ignore')
warnings.filterwarnings('ignore')


# np.set_printoptions(suppress=True)

def compute_stats(buy_day, sell_day, buy_price, sell_price):
    try:
        buy_day = np.array(buy_day)
        sell_day = np.array(sell_day)
        buy_price = np.array(buy_price)
        sell_price = np.array(sell_price)
        apg_ = 100 * (sell_price - buy_price[:len(sell_price)]) / buy_price[:len(sell_price)]
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


def plot_from_strat(buy_days, sell_days, buy_prices, sell_prices, stocks, stocks_traded_idx):
    plot_random = random.sample(stocks_traded_idx, 5) if len(stocks_traded_idx) > 5 else random.sample(
        stocks_traded_idx, len(stocks_traded_idx))
    for idx in plot_random:
        stock = stocks[idx]
        # if stock.ticker[0][:] not in ['APT','DVAX','IG','PLAY','SGBX','VRTX']:
        #     continue
        buy_day = buy_days[idx]
        sell_day = sell_days[idx]
        buy_price = buy_prices[idx]
        sell_price = sell_prices[idx]
        apg_, apg, wp, history, ahl = compute_stats(buy_day, sell_day, buy_price, sell_price)
        if idx in plot_random:
            # look_back = 1000 if len(stock.metrics[:, 0]) > 1000 else len(stock.metrics[:, 0])
            look_back = len(stock.metrics[:, 0])
            fig = plt.figure(figsize=(12, 6))
            fig.patch.set_facecolor('xkcd:gray')
            gs = gridspec.GridSpec(4, 1)
            ax1 = plt.subplot(gs[:3])
            ax1.set_facecolor('xkcd:gray')
            open = stock.metrics[-look_back:, 0]
            high = stock.metrics[-look_back:, 1]
            low = stock.metrics[-look_back:, 2]
            close = stock.metrics[-look_back:, 3]
            candle(ax1, open, high, low, close, colorup="cyan", colordown="red", width=.4)
            plt.plot(np.array(buy_day[:len(sell_day)]), np.array(buy_price[:len(sell_day)]), 'g.',
                     markersize='15')
            plt.plot(np.array(sell_day), np.array(sell_price), 'r.', markersize='15')
            ax1.set_title(stock.ticker[0][:] + '  APG: ' + str(apg) + ' WP: ' + str(
                wp) + ' NumTrades: ' + str(len(buy_day)) + ' AHL: ' + str(ahl) + ' $' + str(
                '{:.2f}'.format(history[0][-1])))
            # ax2 = plt.subplot(gs[2])
            # ax2.set_facecolor('xkcd:gray')
            # plt.plot(apg_, 'k')
            [c1, c2, c3, c4, c5, c6, c7] = get_custom_condition_vals(stock, buy_day[-1] - 1)
            ax3 = plt.subplot(gs[3])
            ax3.set_facecolor('xkcd:gray')
            plt.plot(c1, 'k', label=str(c1))
            plt.plot(c2, 'r', label=str(c2))
            plt.plot(c3, 'b', label=str(c3))
            plt.plot(c4, 'g', label=str(c4))
            plt.plot(c5, 'm', label=str(c5))
            plt.plot(c6, 'y', label=str(c6))
            plt.plot(c7, 'o', label=str(c7))
            plt.legend()
            # plt.plot(history[1], 'b')
            plt.subplots_adjust(hspace=0.4)
    plt.show()


## Excellent Long Strategy for Covid Market Conditions.
# def get_custom_condition(stock, day):
#     opens = stock.metrics[day - 15:day, stock.names['open']]
#     close = stock.metrics[day, stock.names['close']]
#     pc1 = (close - opens) / opens
#     c1 = any(pc1 < -0.5)
#     c2 = stock.metrics[day, stock.names['gap']] > 0
#     c3 = stock.metrics[day, stock.names['percent_change']] < 0
#     c4 = stock.metrics[day, stock.names['gap']] < 0.1
#     c5 = np.mean(np.abs(pc1)) > 0.05
#     c6 = stock.metrics[day + 1, stock.names['gap']] >= 0
#     c7 = stock.moving_average(10, stock.metrics[day - 10:day, stock.names['volume']])[-1] > 1E6
#     return [c1, c2, c3, c4, c5, c6, c7]

def get_custom_condition_vals(stock, day):
    opens = stock.metrics[day - 15:day, stock.names['open']]
    close = stock.metrics[day, stock.names['close']]
    pc1 = (close - opens) / opens
    avg_pc1 = np.mean(np.abs(pc1))
    c1 = any(pc1 < -3 * avg_pc1)
    c2 = stock.metrics[day, stock.names['gap']]
    c4 = stock.metrics[day, stock.names['percent_change']]
    c3 = stock.metrics[day, stock.names['gap']]
    c5 = avg_pc1
    c6 = stock.metrics[day + 1, stock.names['gap']]
    c7 = stock.moving_average(10, stock.metrics[day - 10:day, stock.names['volume']])[-1] > 1E6
    # c8 = stock.metrics[day, stock.names['open']] < stock.metrics[day,stock.names['ma20']]
    return [c1, c2, c3, c4, c5, c6, c7]


def get_custom_condition(stock, day):
    opens = stock.metrics[day - 15:day, stock.names['open']]
    close = stock.metrics[day, stock.names['close']]
    pc1 = (close - opens) / opens
    avg_pc1 = np.mean(np.abs(pc1))
    c1 = any(pc1 < -3 * avg_pc1)
    c2 = stock.metrics[day, stock.names['gap']] > 0
    c3 = stock.metrics[day, stock.names['percent_change']] < 0
    c4 = stock.metrics[day, stock.names['gap']] < 0.1
    c5 = avg_pc1 > 0.05
    c6 = stock.metrics[day + 1, stock.names['gap']] >= 0
    c7 = stock.moving_average(10, stock.metrics[day - 10:day, stock.names['volume']])[-1] > 1E6
    c8 = stock.metrics[day, stock.names['open']] < stock.metrics[day, stock.names['ma50']]
    return [c1, c2, c3, c4, c5, c6, c7, c8]


def get_ohlc(stock, day):
    open = stock.metrics[day, stock.names['open']]
    high = stock.metrics[day, stock.names['high']]
    low = stock.metrics[day, stock.names['low']]
    close = stock.metrics[day, stock.names['close']]
    return open, high, low, close


def swing_(stocks, day_trade=False, hard_stop=True, model_type='Custom', model=False):
    stats = []
    tickers = []
    buy_days = []
    sell_days = []
    buy_prices = []
    sell_prices = []
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
        day = -look_back + 60
        buy_day = []
        sell_day = []
        buy_price = []
        sell_price = []
        while day < -2:
            pct_range = np.mean(np.abs(stock.metrics[day - 10:day, stock.names['percent_change']]))
            take_gain_percent = 1 + pct_range * 3
            stop_loss_percent = 1 - pct_range * 3
            stop_loss_percent2 = 1 - pct_range * 3
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
                open, high, low, close = get_ohlc(stock, day)
                buy_day.append(look_back + day)
                buy_price.append(open)
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
                    sell_price.append(stock.metrics[day, close])
                    sell_day.append(look_back + day)
                else:
                    current_high = np.max([open, close])
                    stop_loss = hard_stop_val if hard_stop else stop_loss_percent2 * current_high
                    while True:
                        day += 1
                        open, high, low, close = get_ohlc(stock, day)
                        if not day < -1:
                            break
                        # condition = get_custom_condition(stock, day)
                        if open <= stop_loss:
                            sell_price.append(open)
                            sell_day.append(look_back + day)
                            break
                        if open >= buy_price[-1] * take_gain_percent:
                            sell_price.append(open)
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
                        if open > current_high:
                            current_high = open
                        if close > current_high:
                            current_high = close
                        stop_loss = hard_stop_val if hard_stop else stop_loss_percent2 * current_high
            else:
                day += 1
        buy_days.append(buy_day)
        sell_days.append(sell_day)
        buy_prices.append(buy_price)
        sell_prices.append(sell_price)
        apg_, apg, wp, history, ahl = compute_stats(buy_day, sell_day, buy_price, sell_price)
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
    print_stats(stats, tickers)
    plot_from_strat(buy_days, sell_days, buy_prices, sell_prices, stocks, stocks_traded_idx)
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
