from trade_strats import *
from scrapeData import *

load = True

if load:
    stocks = load_stocks('stocks_100d.obj')
else:
    ticker_list = ['SP500_Labels.txt', 'Penny.txt', 'mid_cap.txt', 'fat_list.txt', 'spy.txt']
    prefix = '/home/carmelo/Documents/StockMarket/Ticker_Lists/'
    tickers = gather_tickers(prefix + ticker_list[-2])
    download_parallel_quotes(tickers)
    s = parse_csv(tickers)
    s = clean_stock_list(s)
    save_stocks(s, 'stocks.obj')
    stocks = s
bd, sd, bp, sp, stats = swing_(stocks, day_trade=False, hard_stop=False, model_type='Custom')




