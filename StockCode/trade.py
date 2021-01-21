from trade_strats import *
from process_data import *
stocks = load_stocks('stocks_100d.obj')
# use days_back=-1 for full range
bd, sd, bp, sp, stats = swing_(stocks, day_trade=False, hard_stop=False, model_type='Custom', days_back=20)


