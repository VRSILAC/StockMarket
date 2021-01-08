from scrapeData import *
import traceback
ticker_list = ['SP500_Labels.txt', 'Penny.txt', 'mid_cap.txt', 'fat_list.txt', 'spy.txt']
prefix = '/home/carmelo/Documents/StockMarket/Ticker_Lists/'
tickers = gather_tickers(prefix + ticker_list[-2])
# download_parallel_quotes(tickers)
s = parse_csv(tickers)
s = clean_stock_list(s)
save_stocks(s, 'stocks.obj')

try:
    stocks = []
    t = []
    for stock in s:
        if stock.ticker[0] not in t and len(stock.metrics[:, 0]) > 500:
            stock.metrics = stock.metrics[-500:, :]
            stocks.extend([stock])
            t.append(stock.ticker[0])
        else:
            print('Duplicate')

    save_stocks(stocks, 'stocks_500d.obj')
except Exception:
    print(traceback.format_exc())
    pass

try:
    stocks = []
    t = []
    for stock in s:
        if stock.ticker[0] not in t and len(stock.metrics[:, 0]) > 100:
            stock.metrics = stock.metrics[-100:, :]
            stocks.extend([stock])
            t.append(stock.ticker[0])
        else:
            print('Duplicate')

    save_stocks(stocks, 'stocks_100d.obj')
except Exception:
    print(traceback.format_exc())
    pass