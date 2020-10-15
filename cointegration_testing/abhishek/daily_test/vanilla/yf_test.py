from pandas import DataFrame
from get_all_tickers import get_tickers as gt
from get_all_tickers.get_tickers import Region
from statsmodels.tsa.stattools import coint
from yfinance import Ticker
import numpy as np


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
    def get_data(ticker):
        data =  Ticker(ticker).history(period = "3y")['Close']
        log_price = np.log(data.values)
    t2data = Ticker(ticker2).history(period = "3y")['Close']
    logpricet2 = np.log(t2data.values)

    #Clean NaN and Inf Vals
    
    logpricet1 = logpricet1[~np.isinf(logpricet1)]
    logpricet2 = logpricet2[~np.isnan(logpricet2)]
    logpricet2 = logpricet2[~np.isinf(logpricet2)]
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
    seen_pairs = set()
    results = []
    i = 0
    for ticker1 in tickers:
        for ticker2 in tickers:
            ordered_pair =(min(ticker1, ticker2), max(ticker1, ticker2))
            if ticker1  == ticker2 or ordered_pair in seen_pairs:
                continue
            seen_pairs.add(ordered_pair)
            ordered_pair = list(ordered_pair)
            level, score = englegranger(ticker1, ticker2)
            if level == -1*float('inf'):
                continue
            ordered_pair.append(level)
            ordered_pair.append(score)
            results.append(ordered_pair)
            i += 1
            if i % 1000 == 0:
                print(i)
                results_df = DataFrame(results, columns={'Ticker 1', 'Ticker 2', '5% Level', 'Normalized Score'})
                results_df.to_csv('coint_scores.csv', index=False, encoding='utf-8')


    return results

results = run_coint_test()
results.sort(key=lambda x: x[2])
results_df = DataFrame(results, columns={'Ticker 1', 'Ticker 2', '5% Level', 'Normalized Score'})
results_df.to_csv('coint_scores.csv', index=False, encoding='utf-8')
