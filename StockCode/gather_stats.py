from trade_strats import *
from scrapeData import *

stocks = load_stocks('stocks_1000d.obj')


def draw_box_plot(data, offset, edge_color, fill_color):
    pos = np.arange(data.shape[1]) + offset
    bp = ax.boxplot(data, positions=pos, widths=0.3, patch_artist=True, manage_ticks=False)
    for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)
    for patch in bp['boxes']:
        patch.set(facecolor=fill_color)


percents = [-0.2, 0.2]
stocks = make_labels_percent_gain(stocks, percents[0], 'le', label_pg_crit=percents[0])
stocks = make_labels_percent_gain(stocks, percents[1], 'ge', label_pg_crit=percents[1])
col_data = stocks[0].names
for idx, stock in enumerate(stocks):
    stock.metrics_normalized = stock.metrics.copy()
    stock.metrics_normalized = normalize_data(stock.metrics_normalized)
    # up_days = np.nonzero(stock.label_pg_actual == percents[1])[0]
    # down_days = np.nonzero(stock.label_pg_actual == percents[0])[0]
    up_days = np.nonzero(stock.label_pg == percents[1])[0]
    down_days = np.nonzero(stock.label_pg == percents[0])[0]
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
plt.ylim([-1.1, 1.1])
plt.plot(np.zeros(len(col_data)), 'k')
plt.show()
plt.close()
