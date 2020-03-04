import random
import numpy as np
from candle import *
import matplotlib.pyplot as plt


def test_strat(stocks, split1, split2, model, cnn=False):
    col_data = col_data = stocks[0].metrics_column_data
    to_show = random.sample(range(len(stocks)), 5)
    stats_names = ['Stock', 'APG', 'WP', 'NumTrades', 'AvgHoldLen']
    tkrs = []
    stats2 = np.zeros((0, 4))
    for k in range(len(stocks)):
        if stocks[k].ticker[0] != 'No File':
            split_idx1 = round(split1 * len(stocks[k].metrics[:, 0]))
            split_idx2 = round(split2 * len(stocks[k].metrics[:, 0]))
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
            # if True:
            if k in to_show:
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
