from trade_strats import *
from process_data import *


def parser():
    parser = argparse.ArgumentParser(description='Stock Market Ticker Downloader')
    parser.add_argument("--day_trade", default=False, type=bool, help="sell stocks on same day")
    parser.add_argument("--hard_stop", default=False, type=bool,
                        help="use hard_stop to use stop loss as min price over last 10 days")
    parser.add_argument("--model", default='Custom', type=str,
                        help="use model_type 'Custom' always (for now)")
    parser.add_argument("--days_back", default=-1, type=int, help="use days_back=-1 for full range")
    parser.add_argument("--dataset", default='/home/carmelo/Documents/StockMarket/StockCode/stocks_100d.obj', type=str,
                        help="path pointing to processed stock data")
    return parser.parse_args()


def check_arguments_errors(args):
    if not os.path.exists(args.dataset):
        raise (ValueError("Invalid dataset path {}".format(os.path.abspath(args.config_file))))


def main():
    args = parser()
    check_arguments_errors(args)
    stocks = load_stocks(args.dataset)
    bd, sd, bp, sp, stats = swing_(stocks, day_trade=args.day_trade, hard_stop=args.hard_stop, model_type=args.model,
                                   days_back=args.days_back)
    trades = []
    for sidx in range(len(bd)):
        if bd[sidx] != [] and len(bd[sidx]) == len(sd[sidx]):
            for t_idx, bday in enumerate(bd[sidx]):
                trades.append([sidx, bday, sd[sidx][t_idx], bp[sidx][t_idx], sp[sidx][t_idx]])

    trades = np.array(trades)
    idx = trades[:, 1].argsort(axis=0)
    trades = trades[idx, :]
    np.save('trades', trades)


if __name__ == '__main__':
    main()
