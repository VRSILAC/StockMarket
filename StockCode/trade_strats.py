import matplotlib.pyplot as plt
from matplotlib import gridspec
from candle import *
import numpy as np
import random


def compute_stats(buy_day, sell_day, buy_price, sell_price):
    buy_day = np.array(buy_day)
    sell_day = np.array(sell_day)
    buy_price = np.array(buy_price)
    sell_price = np.array(sell_price)
    apg_ = 100 * (sell_price - buy_price[:len(sell_price)]) / buy_price[:len(sell_price)]
    apg = np.round(np.mean(apg_), 2)
    wp = np.round(len(np.nonzero(apg_ > 0)[0]) / len(buy_day), 2)
    ahl = np.round(np.mean(sell_day - buy_day[:len(sell_day)]), 1)
    history = [compute_profit(apg_), compute_profit(random.sample(list(apg_), len(apg_)))]
    return apg_, apg, wp, history, ahl


def compute_profit(apg_):
    investment = 1000
    profit = 0
    account_history = [1000]
    history = []
    for k in apg_:
        # investment = investment+0.5*profit
        investment = investment if account_history[-1] > 1000 else account_history[-1]
        trade_profit = (investment) * (k / 100)
        profit += trade_profit
        history.append(profit)
        account_history.append(account_history[-1] + trade_profit)
        if account_history[-1] < 500:
            profit += -500
            history[-1] = history[-1] - 500
            account_history = [1000]
        print(
            'I: ' + str(np.round(investment, 2)) + ' P: ' + str(np.round(profit, 2)) + 'AH: ' + str(
                np.round(account_history[-1], 2)))
    return history


def get_conditions(stock, conditions, day):
    condition = []
    for k in range(len(conditions)):
        metric = conditions[k][0]
        logic = conditions[k][1]
        value = conditions[k][2]
        if logic == 'lt':
            b_val = stock.metrics[day, stock.names[metric]] < value
            condition.append(b_val)
        elif logic == 'le':
            b_val = stock.metrics[day, stock.names[metric]] <= value
            condition.append(b_val)
        elif logic == 'eq':
            b_val = stock.metrics[day, stock.names[metric]] == value
            condition.append(b_val)
        elif logic == 'ne':
            b_val = stock.metrics[day, stock.names[metric]] != value
            condition.append(b_val)
        elif logic == 'ge':
            b_val = stock.metrics[day, stock.names[metric]] >= value
            condition.append(b_val)
        elif logic == 'gt':
            b_val = stock.metrics[day, stock.names[metric]] > value
            condition.append(b_val)
        else:
            print('Logic Not Supported')
    return condition


def day_trade(stocks, conditions):
    lengths = [len(stocks[k].metrics[:, 0]) for k in range(len(stocks))]
    look_back = np.min(lengths) if np.min(lengths) < 1000 else 1000
    buy_days = []
    sell_days = []
    buy_prices = []
    sell_prices = []
    for stock in stocks:
        day = -look_back
        buy_day = []
        sell_day = []
        buy_price = []
        sell_price = []
        print(stock.ticker[0])
        while day < -1:
            condition = get_conditions(stock, conditions, day)
            if all(condition):
                day += 1
                buy_day.append(look_back + day)
                buy_price.append(stock.metrics[day, 0])
                if stock.metrics[day, 2] <= 0.95 * buy_price[-1]:
                    sell_price.append(0.95 * buy_price[-1])
                    sell_day.append(look_back + day)
                else:
                    sell_price.append(stock.metrics[day, 3])
                    sell_day.append(look_back + day)
            else:
                day += 1
        buy_days.append(buy_day)
        sell_days.append(sell_day)
        buy_prices.append(buy_price)
        sell_prices.append(sell_price)
        apg_, apg, wp, history, ahl = compute_stats(buy_day, sell_day, buy_price, sell_price)
        fig = plt.figure()
        fig.patch.set_facecolor('xkcd:gray')
        gs = gridspec.GridSpec(4, 1)
        ax1 = plt.subplot(gs[:2])
        ax1.set_facecolor('xkcd:gray')
        open = stock.metrics[-look_back:, 0]
        high = stock.metrics[-look_back:, 1]
        low = stock.metrics[-look_back:, 2]
        close = stock.metrics[-look_back:, 3]
        candle(ax1, open, high, low, close, colorup="cyan", colordown="red", width=.4)
        plt.plot(np.array(buy_day), np.array(buy_price), 'g.', markersize='10')
        plt.plot(np.array(buy_day), np.array(sell_price), 'r.', markersize='10')
        ax1.set_title(
            stock.ticker[0][:] + '  APG: ' + str(apg) + ' WP: ' + str(wp) + ' NumTrades: ' + str(
                len(buy_day)) + ' AHL: ' + str(ahl) + ' $' + str('{:.2f}'.format(history[0][-1])))
        ax2 = plt.subplot(gs[2])
        ax2.set_facecolor('xkcd:gray')
        plt.plot(apg_, 'k')
        # ax2.ylabel('Percent_Gain')
        ax3 = plt.subplot(gs[3])
        ax3.set_facecolor('xkcd:gray')
        plt.plot(history[0], 'k')
        plt.plot(history[1], 'b')
        # ax3.ylabel('Account Value')
        plt.subplots_adjust(hspace=0.4)
    plt.show()
    return buy_days, sell_days, buy_prices, sell_prices


def swing_with_stop(stocks, conditions):
    lengths = [len(stocks[k].metrics[:, 0]) for k in range(len(stocks))]
    look_back = np.min(lengths) if np.min(lengths) < 1000 else 1000
    stop_loss_percent = 0.92
    buy_days = []
    sell_days = []
    buy_prices = []
    sell_prices = []
    for stock in stocks:
        day = -look_back
        buy_day = []
        sell_day = []
        buy_price = []
        sell_price = []
        print(stock.ticker[0])
        while day < -1:
            condition = get_conditions(stock, conditions, day)
            if all(condition):
                day += 1
                buy_day.append(look_back + day)
                buy_price.append(stock.metrics[day, 0])
                if stock.metrics[day, 2] <= 0.95 * buy_price[-1]:
                    sell_price.append(stop_loss_percent * buy_price[-1])
                    sell_day.append(look_back + day)
                else:
                    current_high = np.max([stock.metrics[day, 0], stock.metrics[day, 3]])
                    stop_loss = stop_loss_percent * current_high
                    while True:
                        day += 1
                        if not day < -1:
                            break
                        open = stock.metrics[day, 0]
                        low = stock.metrics[day, 2]
                        close = stock.metrics[day, 3]
                        if open <= stop_loss:
                            sell_price.append(open)
                            sell_day.append(look_back + day)
                            break
                        if open > current_high:
                            current_high = open
                            stop_loss = current_high * stop_loss_percent
                        if low <= stop_loss:
                            sell_price.append(stop_loss)
                            sell_day.append(look_back + day)
                            break
                        if close > current_high:
                            current_high = close
                            stop_loss = current_high * stop_loss_percent
            else:
                day += 1
        buy_days.append(buy_day)
        sell_days.append(sell_day)
        buy_prices.append(buy_price)
        sell_prices.append(sell_price)
        apg_, apg, wp, history, ahl = compute_stats(buy_day, sell_day, buy_price, sell_price)
        fig = plt.figure()
        fig.patch.set_facecolor('xkcd:gray')
        gs = gridspec.GridSpec(4, 1)
        ax1 = plt.subplot(gs[:2])
        ax1.set_facecolor('xkcd:gray')
        open = stock.metrics[-look_back:, 0]
        high = stock.metrics[-look_back:, 1]
        low = stock.metrics[-look_back:, 2]
        close = stock.metrics[-look_back:, 3]
        candle(ax1, open, high, low, close, colorup="cyan", colordown="red", width=.4)
        plt.plot(np.array(buy_day[:len(sell_day)]), np.array(buy_price[:len(sell_day)]), 'g.',
            markersize='10')
        plt.plot(np.array(sell_day), np.array(sell_price), 'r.', markersize='10')
        ax1.set_title(
            stock.ticker[0][:] + '  APG: ' + str(apg) + ' WP: ' + str(wp) + ' NumTrades: ' + str(
                len(buy_day)) + ' AHL: ' + str(ahl) + ' $' + str('{:.2f}'.format(history[0][-1])))
        ax2 = plt.subplot(gs[2])
        ax2.set_facecolor('xkcd:gray')
        plt.plot(apg_, 'k')
        # ax2.ylabel('Percent_Gain')
        ax3 = plt.subplot(gs[3])
        ax3.set_facecolor('xkcd:gray')
        plt.plot(history[0], 'k')
        plt.plot(history[1], 'b')
        # ax3.ylabel('Account Value')
        plt.subplots_adjust(hspace=0.4)
    plt.show()
    return buy_days, sell_days, buy_prices, sell_prices
