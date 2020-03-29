from test_strat import *
from scrapeData import *

ticker_list = ['SP500_Labels.txt', 'Penny.txt', 'Single.txt', 'test_list.txt', 'gold.txt']
tickers = gather_tickers(ticker_list[-1])

### Workflow
# Inside scrapeData.py
download_quotes(tickers)
stocks = parse_csv(tickers)
stocks = make_labels_percent_gain(stocks, label_pg_crit=1)
(split1, split2) = (0.6, 0.8)
stocks = populate_class(stocks, split1, split2)
h = 30
cnn = False
train, test, validate = train_test_validate(stocks, split1, split2, h, cnn=cnn)
if cnn:
    model, history = make_2d_cnn(train, test, h)
else:
    model, history = make_1d_ann(train, test)

# Inside test_strat.py
test_strat(stocks, split1, split2, model, cnn=cnn)

# train, test, validate = train_test_validate_1d(stocks)
# model = make_1d_ann(train, test)

lengths = [len(stocks[k].metrics[:,0]) for k in range(len(stocks))]
look_back = np.min(lengths)
days = np.arange(-look_back,0)

for k in
for day in days:
    if
