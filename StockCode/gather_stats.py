from trade_strats import *
from scrapeData import *


def draw_box_plot(data, offset, edge_color, fill_color):
    pos = np.arange(data.shape[1]) + offset
    bp = ax.boxplot(data, positions=pos, widths=0.3, patch_artist=True, manage_ticks=False)
    for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)
    for patch in bp['boxes']:
        patch.set(facecolor=fill_color)


ticker_list = ['SP500_Labels.txt', 'Penny.txt', 'mid_cap.txt', 'test_list.txt', 'gold.txt']
prefix = 'C:\\Users\\carme\\Desktop\\TheProverbialCode\\StockMarket\\Ticker_Lists\\'
tickers = gather_tickers(prefix + ticker_list[-1])

### Workflow
# Inside scrapeData.py
download_quotes(tickers)
stocks = parse_csv(tickers)
stocks = make_labels_percent_gain(stocks, -10, 'le', label_pg_crit=-5)
stocks = make_labels_percent_gain(stocks, 10, 'ge', label_pg_crit=5)
col_data = stocks[0].names

norm_except = [col_data['percent_change'], col_data['percent_change1'], col_data['percent_change2'],
               col_data['percent_change3'], col_data['percent_change4'], col_data['open_diff'],
               col_data['high_diff'], col_data['low_diff'], col_data['close_diff'],
               col_data['open_diff2'], col_data['high_diff2'], col_data['low_diff2'],
               col_data['close_diff2'], col_data['open_diff3'], col_data['high_diff3'],
               col_data['low_diff3'], col_data['close_diff3'], col_data['open_diff4'],
               col_data['high_diff4'], col_data['low_diff4'], col_data['close_diff4'],
               col_data['gap1'], col_data['gap2'], col_data['gap3'], col_data['gap4'],
               col_data['ma5_10'], col_data['ma5_15'], col_data['ma5_20'], col_data['ma5_50'],
               col_data['ma5_100'], col_data['ma5_200'], col_data['ma10_15'], col_data['ma10_20'],
               col_data['ma10_50'], col_data['ma10_100'], col_data['ma10_200'], col_data['ma15_20'],
               col_data['ma15_50'], col_data['ma15_100'], col_data['ma15_200'], col_data['ma20_50'],
               col_data['ma20_100'], col_data['ma20_200'], col_data['ma50_100'],
               col_data['ma50_200'], col_data['ma100_200'], col_data['dm_ma5'], col_data['dm_ma10'],
               col_data['dm_ma15'], col_data['dm_ma20'], col_data['dm_ma50'], col_data['dm_ma100'],
               col_data['dm_ma200'], col_data['mb1_p'], col_data['mb2_p'], col_data['mb3_p'],
               col_data['mb4_p'], col_data['mb5_p']]

norm_these = np.delete(np.arange(len(col_data)), norm_except)

for idx, stock in enumerate(stocks):
    stock.metrics_normalized = stock.metrics.copy()
    stock.metrics_normalized[:, norm_these] = normalize_data(
        stock.metrics_normalized[:, norm_these])
    up_days = np.nonzero(stock.label_pg_actual == 10)[0]
    down_days = np.nonzero(stock.label_pg_actual == -10)[0]
    if idx == 0:
        up_metrics = stock.metrics_normalized[up_days, :]
        down_metrics = stock.metrics_normalized[down_days, :]
    else:
        up_metrics = np.vstack((up_metrics, stock.metrics_normalized[up_days, :]))
        down_metrics = np.vstack((down_metrics, stock.metrics_normalized[down_days, :]))

fig, ax = plt.subplots(figsize=(12, 5))
draw_box_plot(up_metrics, -0.2, 'green', 'white')
draw_box_plot(down_metrics, 0.2, 'red', 'white')
plt.xticks(xrange(up_metrics.shape[1]), labels=col_data, rotation=90)
plt.subplots_adjust(left=0.05, bottom=0.28, right=0.95)
plt.ylim([-10, 10])
plt.show()
plt.close()
