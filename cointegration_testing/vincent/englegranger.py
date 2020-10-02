from statsmodels.tsa.stattools import coint
import yfinance as yf
import numpy as np

def englegranger(ticker1, ticker2, start, end):
    """Engle-Granger
        params:
            ticker1: the first ticker in our pair
            ticker2: the second ticker in our pair
            start: the start date for our analysis
            end: the end date for our analysis
        returns:
            (5% confidence level - coint score) / coint score
    """
    t1data = yf.download(tickers = ticker1, start = start, end = end, interval = "1d")['Close']
    t2data = yf.download(tickers = ticker2, start = start, end = end, interval = "1d")['Close']
    logpricet1 = np.log(t1data.values)
    logpricet2 = np.log(t2data.values)
    score, pvalue, levels = coint(logpricet1, logpricet2)
    """ calculate number of standard errors to each level of confidence """
    print ('coint score =',score,'\npvalue =', pvalue,
       '\n1% 5% & 10% = ',levels)
    print("measurement: ", (levels[1] - score) / score)

def main():
    englegranger("KO", "PEP", "2018-09-01", "2020-09-01")
    englegranger("GLD", "GDX", "2018-09-01", "2020-09-01")
    englegranger("GS", "C", "2018-09-01", "2020-09-01")

if __name__ == "__main__":
    main()
