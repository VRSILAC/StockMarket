from trade_strats import *
from scrapeData import *

load = True

if load:
    stocks = load_stocks('stocks_200d.obj')
else:
    ticker_list = ['SP500_Labels.txt', 'Penny.txt', 'mid_cap.txt', 'test_list.txt', 'gold.txt']
    prefix = 'C:\\Users\\carme\\Desktop\\TheProverbialCode\\StockMarket\\Ticker_Lists\\'
    tickers = gather_tickers(prefix + ticker_list[3])
    # download_quotes(tickers)
    stocks = parse_csv(tickers)
    stocks = clean_stock_list(stocks)

bd, sd, bp, sp, stats = swing_(stocks, day_trade=False, hard_stop=False)
# bd, sd, bp, sp, stats = swing_with_stop(stocks, day_trade=False)



