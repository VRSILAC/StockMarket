# from trade_strats import *
# from scrapeData import *
#
# # stocks = load_stocks('stocks_500d.obj')
# ticker_list = ['SP500_Labels.txt', 'Penny.txt', 'mid_cap.txt', 'test_list.txt', 'gold.txt',
#                'spy.txt']
# prefix = 'C:\\Users\\carme\\Desktop\\TheProverbialCode\\StockMarket\\Ticker_Lists\\'
# tickers = gather_tickers(prefix + ticker_list[-1])
# download_quotes(tickers)
# stocks = parse_csv(tickers)
# stocks = clean_stock_list(stocks)
#
# def draw_box_plot(data, offset, edge_color, fill_color):
#     pos = np.arange(data.shape[1]) + offset
#     bp = ax.boxplot(data, positions=pos, widths=0.3, patch_artist=True, manage_ticks=False)
#     for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
#         plt.setp(bp[element], color=edge_color)
#     for patch in bp['boxes']:
#         patch.set(facecolor=fill_color)
#
#
# percents = [-0.02, 0.02]
# stocks = make_labels_percent_gain(stocks, percents[0], 'le', label_pg_crit=percents[0])
# stocks = make_labels_percent_gain(stocks, percents[1], 'ge', label_pg_crit=percents[1])
# col_data = stocks[0].names
# for idx, stock in enumerate(stocks):
#     stock.metrics_normalized = stock.metrics.copy()
#     stock.metrics_normalized = normalize_data(stock.metrics_normalized)
#     up_days = np.nonzero(stock.label_pg_actual == percents[1])[0]
#     down_days = np.nonzero(stock.label_pg_actual == percents[0])[0]
#     # up_days = np.nonzero(stock.label_pg == percents[1])[0]
#     # down_days = np.nonzero(stock.label_pg == percents[0])[0]
#     if idx == 0:
#         up_metrics = stock.metrics_normalized[up_days, :]
#         down_metrics = stock.metrics_normalized[down_days, :]
#     else:
#         up_metrics = np.vstack((up_metrics, stock.metrics_normalized[up_days, :]))
#         down_metrics = np.vstack((down_metrics, stock.metrics_normalized[down_days, :]))
#
# fig, ax = plt.subplots(figsize=(12, 5))
# draw_box_plot(up_metrics, -0.2, 'green', 'white')
# draw_box_plot(down_metrics, 0.2, 'red', 'white')
# plt.xticks(xrange(up_metrics.shape[1]), labels=col_data, rotation=90)
# plt.subplots_adjust(left=0.05, bottom=0.28, right=0.95)
# plt.ylim([-1.1, 1.1])
# plt.plot(np.zeros(len(col_data)), 'k')
# plt.show()
# plt.close()


from trade_strats import *
from process_data import *
from sklearn.preprocessing import StandardScaler

# stocks = load_stocks('stocks_500d.obj')
ticker_list = ['tickers.txt']
prefix = '/home/carmelo/Documents/StockMarket/Ticker_Lists/'
tickers = gather_tickers(prefix + ticker_list[-1])


def draw_box_plot(data, offset, edge_color, fill_color):
    pos = np.arange(data.shape[1]) + offset
    bp = ax.boxplot(data, positions=pos, widths=0.3, patch_artist=True, manage_ticks=False)
    for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)
    for patch in bp['boxes']:
        patch.set(facecolor=fill_color)


stocks = load_stocks('stocks_100d.obj')

data = []
for stock in stocks:
    open = stock.metrics[:, stock.names['open']]
    close = stock.metrics[:, stock.names['close']]
    for day in range(11, len(open) - 3):
        pc = (close[day + 3] - open[day]) / open[day]
        a_range = 10
        percentage_change = a_range * np.mean(np.abs(stock.metrics[day - 10:day, stock.names['percent_change']]))
        big_drop = any(stock.metrics[day:day + 3, stock.names['percent_change']] < -a_range * percentage_change)
        # Buy = 0, Sell = 1, Hold = 2, Nothing = 3
        if pc > percentage_change and not big_drop:
            action = 0
        elif pc > 0 and not big_drop:
            action = 2
        elif pc < percentage_change or big_drop:
            action = 1
        else:
            action = 3
        day_data = [action]
        day_data.extend([np.mean(stock.metrics[day - 2:day, stock.names[key]]) for key in stock.names.keys()])
        data.append(day_data)

data = np.array(data)


# np.save('data', data)
col_data = stocks[0].names

# data = np.load('data.npy')
# data[:, 1:] = normalize_data(data[:, 1:])

# data[:, 1:] = StandardScaler().fit_transform(data[:, 1:])

up_days = np.nonzero(data[:, 0] == 0)[0]
down_days = np.nonzero(data[:, 0] == 1)[0]
up_metrics = data[up_days, 1:]
down_metrics = data[down_days, 1:]

# for idx, stock in enumerate(stocks):

# up_days = np.nonzero(stock.label_pg_actual == percents[1])[0]
# down_days = np.nonzero(stock.label_pg_actual == percents[0])[0]
# up_days = np.nonzero(stock.label_pg == percents[1])[0]
# down_days = np.nonzero(stock.label_pg == percents[0])[0]
# if idx == 0:
#     up_metrics = stock.metrics_normalized[up_days, :]
#     down_metrics = stock.metrics_normalized[down_days, :]
# else:
#     up_metrics = np.vstack((up_metrics, stock.metrics_normalized[up_days, :]))
#     down_metrics = np.vstack((down_metrics, stock.metrics_normalized[down_days, :]))

fig, ax = plt.subplots(figsize=(12, 5))
draw_box_plot(up_metrics, -0.2, 'green', 'white')
draw_box_plot(down_metrics, 0.2, 'red', 'white')
plt.xticks(xrange(up_metrics.shape[1]), labels=col_data, rotation=90)
plt.subplots_adjust(left=0.05, bottom=0.28, right=0.95)
plt.ylim([-1.1, 1.1])
plt.plot(np.zeros(len(col_data)), 'k')
plt.show()
plt.close()
