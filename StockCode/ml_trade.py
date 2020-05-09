from scrapeData import *
from ml_frame import *
from trade_strats import *

stocks = load_stocks('stocks_1000d.obj')
# stocks = stocks[::100]
stocks = make_labels_percent_gain(stocks, 1, 'ge', label_pg_crit=0.05)
(split1, split2) = (0.6, 0.8)
stocks = populate_class(stocks, split1, split2)
h = len(stocks[0].metrics[0, :])
cnn = True
train, test, validate = train_test_validate(stocks, split1, split2, h, cnn=cnn)
if cnn:
    model, history = make_2d_cnn(train, test, h)
    model_type = '2DCNN'
else:
    model, history = make_1d_ann(train, test)
    model_type = 'ANN'

swing_(stocks, model_type=model_type, model=model)

# # Inside test_strat.py
# test_strat(stocks, split1, split2, model, cnn=cnn)
