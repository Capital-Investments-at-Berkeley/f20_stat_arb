import yfinance as yf
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import statsmodels.api as sm
import numpy as np
import itertools
import pandas as pd
import datetime as dt
from pykalman import KalmanFilter
import mlfinlab.data_structures.time_data_structures as time_data_structures
import operator
from mlfinlab.optimal_mean_reversion import OrnsteinUhlenbeck

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

def runCointTestBasketsEG(etf, tickers, start, end):
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

def runCointTestBasketsJoh(etf, tickers, start, end):
    coint_data = pd.DataFrame(columns=['ticker', 'critical-values', 'trace-stat'])
    etf_data = yf.download(etf, start = start, end = end)
    etf_data = etf_data[['Close']]
    etfLogPrice = np.log(etf_data['Close'].values)
    
    tickers_data = yf.download(tickers, start = start, end = end)
    
    tickers_subsets = []
    for i in range(2, len(tickers) + 1):
        for subset in itertools.combinations(tickers, i):
            tickers_subsets.append(list(subset))
    
    for i, t_list in enumerate(tickers_subsets):
        if i % 500 == 0:
            print(i, "done, out of a total of ", len(tickers_subsets))
        df = tickers_data['Close'][t_list]
        df = df.apply(np.log)
        df['etf'] = etfLogPrice
        if df.isnull().values.any():
            print('err')
            continue
        jres = coint_johansen(df, det_order=0, k_ar_diff=1)
        coint_data.loc[i] = [t_list, jres.trace_stat_crit_vals[-1], jres.trace_stat[-1]]
    return coint_data.sort_values(by='trace-stat', ascending=True)

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

def multivariateKalmanFilter(pricesX: pd.DataFrame, priceY):
    basket = pricesX.to_numpy()
    obs_mat = sm.add_constant(basket, prepend=False)[:, np.newaxis]
    basket_size = obs_mat[0].shape[1]
    delta = 1e-5
    trans_cov = delta / (1 - delta) * np.eye(basket_size)
    init_mean = basket[0]/priceY[0]
    init_mean = np.append(init_mean, [0])
    kf = KalmanFilter(n_dim_obs=1, n_dim_state=basket_size,
                    initial_state_mean = init_mean,
                    initial_state_covariance = np.ones((basket_size,basket_size)),
                    transition_matrices = np.eye(basket_size), 
                    observation_matrices = obs_mat,
                    observation_covariance = 0.5,
                    transition_covariance= trans_cov)
    return kf

def createBands(data, lookback, z_threshold=1):
    data['ema'] = data['spread'].ewm(span=lookback).mean()
    rolling_std = data['spread'].rolling(lookback).std()
    data['upperband'] = data['ema'] + (z_threshold * rolling_std)
    data['lowerband'] = data['ema'] - (z_threshold * rolling_std)
    data = data.dropna()
    data = data.reset_index()
    return data

def createBars(data, lower, upper, avg):
    data['upperband'] = upper
    data['lowerband'] = lower
    data['ema'] = avg
    data = data.dropna()
    data = data.reset_index()
    return data

def createSignal(data):
    data['aboveOrBelowEMA'] = np.where(data['spread'] > data['ema'], 1, -1)
    data['signal'] = np.where(data['spread'] > data['upperband'], -1, np.nan)
    data['signal'] = np.where(data['spread'] < data['lowerband'], 1, data['signal'])
    data['signal'] = np.where(data['aboveOrBelowEMA'] * data['aboveOrBelowEMA'].shift(1) < 0, 
                                     0, data['signal'])
    data['signal'] = data['signal'].ffill().fillna(0)
    return data

def createPositions(data):
    dataWithPosition = pd.DataFrame()
    data['datetime'] = pd.to_datetime(data['datetime'])
    dfList = [group[1] for group in data.groupby(data['datetime'].dt.date)]
    for day in dfList:        
        day['aboveOrBelowEMA'] = np.where(day['spread'] > day['ema'], 1, -1)
        day['signal'] = np.where((day['spread'] > day['upperband']), -1, np.nan)
        day['signal'] = np.where((day['spread'] < day['lowerband']), 1, day['signal'])
        day['signal'] = np.where(day['aboveOrBelowEMA'] * day['aboveOrBelowEMA'].shift(1) < 0, 
                                         0, day['signal'])
        day['signal'] = day['signal'].ffill().fillna(0)
        day['position'] = day['signal'].shift(1).fillna(0)
        if day['position'].iloc[-1] != 0:
            day['position'].iloc[-1] = 0
        dataWithPosition = dataWithPosition.append(day)      
    return dataWithPosition

def calculateDiffThresh(data, q=0.1):
    spread_diff = np.ediff1d(data['spread'])
    spread_diff = np.insert(spread_diff, 0, 0)
    data['spread_diff'] = 100 * (abs(spread_diff) / abs(data['spread'].shift(-1)))
    return data['spread_diff'].quantile(q)

def createOptimalPositions(data, threshold=.0001):
    spread_diff = np.ediff1d(data['spread'])
    spread_diff = np.insert(spread_diff, 0, 0)
    data['spread_diff'] = 100 * (abs(spread_diff) / abs(data['spread'].shift(-1)))
    dataWithPosition = pd.DataFrame()
    data['datetime'] = pd.to_datetime(data['datetime'])
    dfList = [group[1] for group in data.groupby(data['datetime'].dt.date)]
    for day in dfList:        
        day['aboveOrBelowEMA'] = np.where(day['spread'] > day['ema'], 1, -1)
        day['signal'] = np.where((day['spread'] > day['upperband']) & (day['spread_diff'].abs() < threshold), -1, np.nan)
        day['signal'] = np.where((day['spread'] < day['lowerband']) & (day['spread_diff'].abs() < threshold), 1, day['signal'])
        day['signal'] = np.where(day['aboveOrBelowEMA'] * day['aboveOrBelowEMA'].shift(1) < 0, 
                                         0, day['signal'])
        day['signal'] = day['signal'].ffill().fillna(0)
        day['position'] = day['signal'].shift(1).fillna(0)
        if day['position'].iloc[-1] != 0:
            day['position'].iloc[-1] = 0
        dataWithPosition = dataWithPosition.append(day)      
    return dataWithPosition

def createDerivativePositions(data):
    spread_diff = np.ediff1d(data['spread'])
    spread_diff = np.insert(spread_diff, 0, 0)
    data['spread_diff'] = spread_diff
    dataWithPosition = pd.DataFrame()
    data['datetime'] = pd.to_datetime(data['datetime'])
    dfList = [group[1] for group in data.groupby(data['datetime'].dt.date)]
    for day in dfList:        
        day['aboveOrBelowEMA'] = np.where(day['spread'] > day['ema'], 1, -1)
        day['signal'] = np.where(day['spread'] + day['spread_diff'] > day['upperband'], -1, np.nan)
        day['signal'] = np.where(day['spread'] + day['spread_diff'] < day['lowerband'], 1, day['signal'])
        day['signal'] = np.where(day['aboveOrBelowEMA'] * day['aboveOrBelowEMA'].shift(1) < 0, 
                                         0, day['signal'])
        day['signal'] = day['signal'].ffill().fillna(0)
        day['position'] = day['signal'].shift(1).fillna(0)
        if day['position'].iloc[-1] != 0:
            day['position'].iloc[-1] = 0
        dataWithPosition = dataWithPosition.append(day)      
    return dataWithPosition

def constructTradeLog(datetime, positions, priceY, priceX, hedgeRatios, stoploss = None, lot_size = 1):
    logdictlist = []
    minutedictlist = []
    
    curr_position = 0
    basketSize = len(priceX[0])
    holdingPeriod = 0
    cumulative_tradeprofit = 0
    stoppedOut = False

    tradeDict = {}
    minuteDict = {}
    
    for i, p in enumerate(positions):
        minuteDict['datetime'] = datetime[i]        
        if (curr_position == 0 and p == 1) or (curr_position == 0 and p == -1): # open position; long = long Y, short X
            tradeDict['start'] = datetime[i]
            tradeDict['positionSizeY'] = lot_size
            tradeDict['entryY'] = priceY[i]
            tradeDict['positionSizeX'] = [hedgeRatios[i][j] * lot_size for j in range(basketSize)]
            tradeDict['entryX'] = [priceX[i][j] for j in range(basketSize)]
            tradeDict['initialPortfolioValue'] = (tradeDict['positionSizeY'] * tradeDict['entryY']) + \
                                                    sum([tradeDict['positionSizeX'][j] * priceX[i][j] for j in range(basketSize)])
            if p == 1:
                tradeDict['position'] = 'long'
            else:
                tradeDict['position'] = 'short'
            
            minuteDict['profit'] = 0
            minuteDict['returns'] = 0
            
            holdingPeriod += 1

        elif curr_position * p == 1:
            profit = 0
            if stoppedOut == False:
                if p == 1:
                    profitY = tradeDict['positionSizeY'] * (priceY[i] - priceY[i - 1])
                    profitX = sum([tradeDict['positionSizeX'][j] * (priceX[i - 1][j] - priceX[i][j]) for j in range(basketSize)])
                else:
                    profitY = tradeDict['positionSizeY'] * (priceY[i - 1] - priceY[i])
                    profitX = sum([tradeDict['positionSizeX'][j] * (priceX[i][j] - priceX[i - 1][j]) for j in range(basketSize)])

                profit = profitY + profitX
                cumulative_tradeprofit += profit

                if stoploss != None:
                    if cumulative_tradeprofit / tradeDict['initialPortfolioValue'] <= stoploss:
                        stoppedOut = True
                        tradeDict['end'] = datetime[i]
                        tradeDict['holdingPeriod'] = holdingPeriod
                        tradeDict['exitY'] = priceY[i]
                        tradeDict['exitX'] = [priceX[i][j] for j in range(basketSize)]
                        tradeDict['trade_profit'] = cumulative_tradeprofit
                        tradeDict['trade_returns'] = cumulative_tradeprofit / tradeDict['initialPortfolioValue']
                       
                holdingPeriod += 1
            
            minuteDict['profit'] = profit
            minuteDict['returns'] = profit / tradeDict['initialPortfolioValue']
            
        elif (curr_position == 1 and p == 0) or (curr_position == -1 and p == 0): # close
            profit = 0
            if stoppedOut == False:
                if curr_position == 1:
                    profitY = tradeDict['positionSizeY'] * (priceY[i] - priceY[i - 1])
                    profitX = sum([tradeDict['positionSizeX'][j] * (priceX[i - 1][j] - priceX[i][j]) for j in range(basketSize)])
                else:
                    profitY = tradeDict['positionSizeY'] * (priceY[i - 1] - priceY[i])
                    profitX = sum([tradeDict['positionSizeX'][j] * (priceX[i][j] - priceX[i - 1][j]) for j in range(basketSize)])

                profit = profitY + profitX
                cumulative_tradeprofit += profit
                
                tradeDict['end'] = datetime[i]
                tradeDict['holdingPeriod'] = holdingPeriod
                tradeDict['exitY'] = priceY[i]
                tradeDict['exitX'] = [priceX[i][j] for j in range(basketSize)]
                tradeDict['trade_profit'] = cumulative_tradeprofit
                tradeDict['trade_returns'] = cumulative_tradeprofit / tradeDict['initialPortfolioValue']
                
            minuteDict['profit'] = profit
            minuteDict['returns'] = profit / tradeDict['initialPortfolioValue']
            
            logdictlist.append(tradeDict)
            stoppedOut = False
            holdingPeriod = 0
            cumulative_tradeprofit = 0
            tradeDict = {}
            
        elif curr_position == 0 and p == 0:
            minuteDict['profit'] = 0
            minuteDict['returns'] = 0
        
        curr_position = p
        minutedictlist.append(minuteDict)
        minuteDict = {}
        
    clist = ['start', 'end', 'holdingPeriod', 'position', 'positionSizeY', 'entryY', 'exitY', 'positionSizeX', 'entryX', 'exitX', 
             'initialPortfolioValue', 'trade_profit', 'trade_returns']
    return pd.DataFrame(logdictlist, columns=clist), pd.DataFrame(minutedictlist)

def tuneBBParameters(data, lookbacks, z_threshs, ticker_list, stoploss = None):
    size = len(lookbacks) * len(z_threshs)
    results = {}
    counter = 0
    for lookback in lookbacks:
        for z_thresh in z_threshs:
            counter += 1
            dataTemp = data.copy()
            syntheticAssetLogPrice = dataTemp[ticker_list].apply(np.log)
            qqqLogPrice = np.log(dataTemp['qqqclose'].values)
            
            kf = multivariateKalmanFilter(syntheticAssetLogPrice, qqqLogPrice)
            state_means, state_covs = kf.filter(qqqLogPrice)
            slopes = state_means[:, np.arange(0, len(ticker_list), 1)]
            #intercept = state_means[:, len(ticker_list)]
            
            syntheticAssetEstimate = [np.dot(slopes[i], syntheticAssetLogPrice.values[i].T) for i in range(len(slopes))]
            spread_ts = qqqLogPrice - syntheticAssetEstimate
            
            dataTemp.reset_index(inplace=True)
            dataTemp['logspread'] = spread_ts
            dataTemp['spread'] = np.exp(spread_ts)
            dataTemp = dataTemp.rename(columns={'index': 'datetime'})
            
            price_data = dataTemp[['datetime']]
            price_data['logspread'] = spread_ts
            price_data['spread'] = np.exp(spread_ts)
            price_data['qqqclose'] = dataTemp['qqqclose']
            price_data[ticker_list] = dataTemp[ticker_list]
            
            backtest_data = createBands(price_data, lookback, z_thresh)
            backtest_data = createPositions(backtest_data)
            #backtest_data = createOptimalPositions(backtest_data)
            hedge_ratios = np.asarray([slopes.T[i][lookback - 1:] for i in range(len(slopes.T))]).T
            
            tradeLog, minuteDf = constructTradeLog(backtest_data['datetime'], 
                                                   backtest_data['position'].values, 
                                                   backtest_data['qqqclose'].values, 
                                                   backtest_data[ticker_list].values, 
                                                   hedge_ratios.round(3), stoploss = stoploss,
                                                   lot_size = 1000)
            
            cumulative_profit = minuteDf['profit'].sum()
            cumulative_returns = np.cumprod(1 + minuteDf['returns']).iloc[-1]
            
            minuteDf['datetime'] = pd.to_datetime(minuteDf['datetime'])
            dailyReturns = calculateDailyReturns(minuteDf[['datetime', 'returns']])
            annualizedSharpe = calculateAnnualizedSharpeRatio(dailyReturns)
            
            results[(lookback, z_thresh)] = [cumulative_profit, cumulative_returns, annualizedSharpe]

            if counter % 10 == 0 or counter == size:
                print(counter, "done, out of", size)
                
    return dict(sorted(results.items(), key=lambda item: item[1][2], reverse=True))

def calculateDailyReturns(minuteRets):
    minuteRets['dayperiod'] = minuteRets['datetime'].dt.to_period('D')
    minuteRets['returns'] = minuteRets['returns'] + 1
    dailyreturns = minuteRets.groupby('dayperiod')['returns'].apply(lambda x: x.cumprod().iloc[-1] - 1)
    return dailyreturns

def calculateAnnualizedSharpeRatio(dailyRets):
    return np.sqrt(252) * (dailyRets.mean() / dailyRets.std())
