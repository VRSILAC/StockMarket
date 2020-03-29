from trade_strats import *
from scrapeData import *

ticker_list = ['SP500_Labels.txt', 'Penny.txt', 'Single.txt', 'test_list.txt', 'gold.txt']
prefix = 'C:\\Users\\carme\\Desktop\\TheProverbialCode\\StockMarket\\Ticker_Lists\\'
tickers = gather_tickers(prefix + ticker_list[-2])

### Workflow
# Inside scrapeData.py
download_quotes(tickers)
stocks = parse_csv(tickers)

conditions = [('gap1', 'lt', 0.05), ('gap2', 'lt', 0.10), ('percent_change', 'ge', 0),
              ('d1price', 'ge', 0), ('d2price', 'ge', 0.1), ('percent_change1', 'ge',0)]


# buy_days, sell_days, buy_prices, sell_prices = day_trade(stocks, conditions)
buy_days, sell_days, buy_prices, sell_prices = swing_with_stop(stocks, conditions)