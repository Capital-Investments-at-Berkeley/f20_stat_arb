import yfinance as yf
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm
import numpy as np
import itertools
import pandas as pd
import datetime as dt
from pykalman import KalmanFilter
import mlfinlab.data_structures.time_data_structures as time_data_structures

def runCointTestIndividual(etf, tickers, start, end):
    coint_data = pd.DataFrame(columns=['ticker', 't-stat', 'pval'])
    
    etf_data = yf.download(etf, start = start, end = end)
    etf_data = etf_data[['Close']]
    etfLogPrice = np.log(etf_data['Close'].values)

    for i, t in enumerate(tickers):
        t_data = yf.download(t, start = start, end = end)['Close']
        tLogPrice = np.log(t_data.values)
        if t_data.isnull().values.any() or len(etfLogPrice) != len(tLogPrice):
            print('err')
            continue
        t_stat, pval, crit_values = coint(etfLogPrice, tLogPrice)
        coint_data.loc[i] = [t, t_stat, pval]
        print(t)
    return coint_data.sort_values(by='t-stat', ascending=True)

def runCointTestBaskets(etf, tickers, start, end):
    coint_data = pd.DataFrame(columns=['ticker', 't-stat', 'pval'])
    etf_data = yf.download(etf, start = start, end = end)
    etf_data = etf_data[['Close']]
    etfLogPrice = np.log(etf_data['Close'].values)
    
    tickers_data = yf.download(tickers, start = start, end = end)
    
    tickers_subsets = []
    for i in range(2, len(tickers) + 1):
        for subset in itertools.combinations(tickers, i):
            tickers_subsets += [subset]
    
    for i, t_list in enumerate(tickers_subsets):
        if i % 500 == 0:
            print(i, "done, out of a total of ", len(tickers_subsets))
        basket_sum = sum([tickers_data['Close'][t] for t in t_list])
        tLogPrice = np.log(basket_sum.values)
        if basket_sum.isnull().values.any() or len(etfLogPrice) != len(tLogPrice):
            print('err')
            continue
        t_stat, pval, crit_values = coint(etfLogPrice, tLogPrice)
        coint_data.loc[i] = [t_list, t_stat, pval]
    return coint_data.sort_values(by='t-stat', ascending=True)

def cleanTickData(df):
    year = df['DATE'] // 10000
    month = df['DATE'] // 100 % 100
    day = df['DATE'] % 100
    date_time = month.astype(str) + '/' + day.astype(str) + '/' + year.astype(str) + ' ' + df['TIME_M']
    new_data = pd.concat([pd.to_datetime(date_time, infer_datetime_format=True), df['PRICE'], df['SIZE']], axis=1)
    new_data.columns = ['date', 'price', 'volume']
    print('DONE!')
    return new_data

def getMinuteData(df):
    minuteDf = time_data_structures.get_time_bars(df, resolution='MIN', batch_size=10000000)
    minuteDf['date_time'] = pd.to_datetime(minuteDf['date_time'], unit='s')
    return minuteDf

def univariateKalmanFilter(priceX, priceY):
    obs_mat = sm.add_constant(priceX, prepend=False)[:, np.newaxis]
    delta = 1e-5
    trans_cov = delta / (1 - delta) * np.eye(2)
    firstY = np.log(priceY[0])
    firstX = np.log(priceX[0])
    kf = KalmanFilter(n_dim_obs = 1, n_dim_state = 2,
                    initial_state_mean = [firstY / firstX, 0.0],
                    initial_state_covariance = np.ones((2,2)),
                    transition_matrices = np.eye(2), 
                    observation_matrices = obs_mat,
                    observation_covariance = 0.5,
                    transition_covariance= trans_cov)
    return kf

