from pandas import DataFrame
from get_all_tickers import get_tickers as gt
from get_all_tickers.get_tickers import Region
from statsmodels.tsa.stattools import coint
from yfinance import Ticker
import numpy as np
import concurrent.futures

bad_data=set()
def engle_list(tickers):
    result = englegranger(tickers[0], tickers[1])
    return [tickers[0], tickers[1], result[0], result[1]]

def englegranger(ticker1, ticker2, prnt=False):
    """Engle-Granger
        params:
            ticker1: the first ticker in our pair
            ticker2: the second ticker in our pair
            start: the start date for our analysis
            end: the end date for our analysis
        returns:
            (5% confidence level - coint score) / coint score
    """
    if ticker1 in bad_data or ticker2 in bad_data:
        return -1*float('inf'), 0
    def get_data(ticker, volume=1000000):
        data = Ticker(ticker)
        if data.info['averageVolume10days'] < volume:
            return []
        log_price = np.log(data.history(period = "3y")['Close'].values)
        log_price = log_price[~np.isnan(log_price)]
        log_price = log_price[~np.isinf(log_price)]
        return log_price
    logpricet1 = get_data(ticker1)
    logpricet2 = get_data(ticker2)
    if len(logpricet1) < 100:
        bad_data.add(ticker1)
    if len(logpricet2) < 100:
        bad_data.add(ticker2)
    size = min(len(logpricet1), len(logpricet2))
    if size < 100:
        return -1*float('inf'), 0
    logpricet1 = logpricet1[:size]
    logpricet2 = logpricet2[:size]


    score, pvalue, levels = coint(logpricet1, logpricet2)
    """ calculate number of standard errors to each level of confidence """
    if prnt:
        print ('coint score =',score,'\npvalue =', pvalue,
            '\n1% 5% & 10% = ',levels)
        print("measurement: ", (levels[1] - score) / score)
    return levels[1], (levels[1] - score) / score

def run_coint_test():

    tickers = gt.get_tickers(AMEX=False, NASDAQ=False)
    results = []
    ticker_pairs = [[ticker1, ticker2] for ticker1 in tickers for ticker2 in tickers if ticker1 != ticker2]
    print('lists made')
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(engle_list, pair) for pair in ticker_pairs]
        print('executor set up')
        for i, fut in enumerate(concurrent.futures.as_completed(futures)):
            if i%1000 == 0:
                print(i)
                results_df = DataFrame(results, columns={'Ticker 1', 'Ticker 2', '5% Level', 'Normalized Score'})
                results_df.to_csv('parallel_coint_scores.csv', index=False, encoding='utf-8')
            if fut.result()[2] != -1*float('inf'):
                results.append(fut.result())
            
    return results

results = run_coint_test()
results_df = DataFrame(results, columns={'Ticker 1', 'Ticker 2', '5% Level', 'Normalized Score'})
results_df.to_csv('parallel_coint_scores.csv', index=False, encoding='utf-8')
