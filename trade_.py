from scrapeData import *

ticker_list = ['SP500_Labels.txt', 'Penny.txt', 'Single.txt', 'test_list.txt']
tickers = gather_tickers(ticker_list[3])

### Workflow
download_quotes(tickers)
stocks = parse_csv(tickers)
label_pg_crit = 1
stocks = make_labels_percent_gain(stocks, label_pg_crit)
train, test, validate = train_test_validate_2d(stocks)
# train, test, validate = train_test_validate_1d(stocks)
# model = make_1d_ann(train, test)
model = make_2d_cnn(train, test)
test_strat(stocks,model, cnn=True)






