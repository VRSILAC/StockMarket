from trade_strats import *
from scrapeData import *

ticker_list = ['SP500_Labels.txt', 'Penny.txt', 'mid_cap.txt', 'test_list.txt', 'gold.txt']
prefix = 'C:\\Users\\carme\\Desktop\\TheProverbialCode\\StockMarket\\Ticker_Lists\\'
tl = gather_tickers(prefix + ticker_list[2])
# r = np.random.randint(0, len(tl), 20)
# tickers = [tl[k] for k in r]
tickers = tl
### Workflow
# Inside scrapeData.py
download_quotes(tickers)
stocks = parse_csv(tickers)
stocks = clean_stock_list(stocks)
# save_stocks(stocks)
# stocks = load_stocks()
conditions = [('d2price', 'ge', 0, 0), ('d2price', 'ge', 0, 1), ('d2price', 'le', 0, 2),
              ('gap1', 'ge', 0, -1)]
buy_days, sell_days, buy_prices, sell_prices, stats = swing_with_stop(stocks, conditions)
