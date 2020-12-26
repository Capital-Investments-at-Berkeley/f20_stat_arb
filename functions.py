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

def createSignal(data):
    data['aboveOrBelowEMA'] = np.where(data['spread'] > data['ema'], 1, -1)
    data['signal'] = np.where(data['spread'] > data['upperband'], -1, np.nan)
    data['signal'] = np.where(data['spread'] < data['lowerband'], 1, data['signal'])
    data['signal'] = np.where(data['aboveOrBelowEMA'] * data['aboveOrBelowEMA'].shift(1) < 0, 
                                     0, data['signal'])
    data['signal'] = data['signal'].ffill().fillna(0)
    return data

def constructTradeLog(datetime, positions, priceY, priceX, hedgeRatios, lot_size = 1):
    col_list = ['start', 'end', 'holdingPeriod', 'positionSizeY', 'entryY', 'exitY', 
                'positionSizeX', 'entryX', 'exitX', 'position', 'profit', 'returns']
    df = pd.DataFrame(columns=col_list)
    curr_position = 0
    basketSize = len(priceX[0])
    holdingPeriod = 0
    tradeLog = [None] * 12
    for i, p in enumerate(positions):
        if curr_position == 0 and p == 1: # open long (long Y, short X)
            tradeLog[0] = datetime[i]
            tradeLog[3] = lot_size
            tradeLog[4] = priceY[i]
            tradeLog[6] = [hedgeRatios[i][j] * lot_size for j in range(basketSize)]
            tradeLog[7] = [priceX[i][j] for j in range(basketSize)]
            tradeLog[9] = p
            holdingPeriod += 1
        elif curr_position == 0 and p == -1: # open short
            tradeLog[0] = datetime[i]
            tradeLog[3] = lot_size
            tradeLog[4] = priceY[i]
            tradeLog[6] = [hedgeRatios[i][j] * lot_size for j in range(basketSize)]
            tradeLog[7] = [priceX[i][j] for j in range(basketSize)]
            tradeLog[9] = p
            holdingPeriod += 1
        elif curr_position * p == 1:
            holdingPeriod += 1
        elif curr_position == 1 and p == 0: # close long
            tradeLog[1] = datetime[i]
            tradeLog[2] = holdingPeriod
            tradeLog[5] = priceY[i]
            tradeLog[8] = [priceX[i][j] for j in range(basketSize)]
            profitY = tradeLog[3] * (tradeLog[5] -tradeLog[4])
            profitX = sum([tradeLog[6][j] * (tradeLog[7][j] - tradeLog[8][j]) for j in range(basketSize)])
            tradeLog[10] = profitY + profitX
            tradeLog[11] = tradeLog[10] / ((tradeLog[3] * tradeLog[4]) + sum([tradeLog[6][j] * tradeLog[7][j] for j in range(basketSize)]))
            df.loc[-1] = tradeLog  # adding a row
            df.index = df.index + 1  # shifting index
            df = df.sort_index()  # sorting by index
            tradeLog = [None] * 12
            holdingPeriod = 0
        elif curr_position == -1 and p == 0: # close short
            tradeLog[1] = datetime[i]
            tradeLog[2] = holdingPeriod
            tradeLog[5] = priceY[i]
            tradeLog[8] = [priceX[i][j] for j in range(basketSize)]
            profitY = tradeLog[3] * (tradeLog[4] -tradeLog[5])
            profitX = sum([tradeLog[6][j] * (tradeLog[8][j] - tradeLog[7][j]) for j in range(basketSize)])
            tradeLog[10] = profitY + profitX
            tradeLog[11] = tradeLog[10] / ((tradeLog[3] * tradeLog[4]) + sum([tradeLog[6][j] * tradeLog[7][j] for j in range(basketSize)]))
            df.loc[-1] = tradeLog  # adding a row
            df.index = df.index + 1  # shifting index
            df = df.sort_index()  # sorting by index
            tradeLog = [None] * 12
            holdingPeriod = 0
        curr_position = p
    return df#.reindex(index=df.index[::-1])

def calculateDollarProfit(positions, priceY, priceX, hedgeRatios, lot_size = 1):
    profitY = []
    profitX = []
    curr_position = 0
    openIndex = 0
    position_sizes = []
    for i, p in enumerate(positions):
        if curr_position == 0 and p == 0:
            profitY.append(0)
            profitX.append(0)
            position_sizes.append(1)
        
        elif curr_position == 0 and p == 1: # open long (long Y, short X)
            profitY.append(0)
            profitX.append(0)
            position_sizes.append(1)
            openIndex = i
        
        elif curr_position == 0 and p == -1: # open short
            profitY.append(0)
            profitX.append(0)
            position_sizes.append(1)
            openIndex = i
            
        elif curr_position == 1 and p == 1: # calculate profit from previous bar
            profitY.append(priceY[i] - priceY[i - 1])
            profitX.append(sum([priceX[i - 1][j] * hedgeRatios[openIndex][j] - 
                               priceX[i][j] * hedgeRatios[openIndex][j] for j in range(len(hedgeRatios[openIndex]))]))
            position_sizes.append(priceY[i - 1] + sum([priceX[i - 1][j] * hedgeRatios[openIndex][j] 
                                                   for j in range(len(hedgeRatios[openIndex]))]))
        
        elif curr_position == -1 and p == -1: # calculate profit from previous bar
            profitY.append(priceY[i - 1] - priceY[i])
            profitX.append(sum([priceX[i][j] * hedgeRatios[openIndex][j] - 
                               priceX[i - 1][j] * hedgeRatios[openIndex][j] for j in range(len(hedgeRatios[openIndex]))]))
            position_sizes.append(priceY[i - 1] + sum([priceX[i - 1][j] * hedgeRatios[openIndex][j] 
                                                   for j in range(len(hedgeRatios[openIndex]))]))
            
        elif curr_position == 1 and p == 0: # close long
            profitY.append(priceY[i] - priceY[i - 1])
            profitX.append(sum([priceX[i - 1][j] * hedgeRatios[openIndex][j] - 
                               priceX[i][j] * hedgeRatios[openIndex][j] for j in range(len(hedgeRatios[openIndex]))]))
            position_sizes.append(priceY[i - 1] + sum([priceX[i - 1][j] * hedgeRatios[openIndex][j] 
                                                   for j in range(len(hedgeRatios[openIndex]))]))
            openIndex = 0
        
        elif curr_position == -1 and p == 0: # close short
            profitY.append(priceY[i - 1] - priceY[i])
            profitX.append(sum([priceX[i][j] * hedgeRatios[openIndex][j] - 
                               priceX[i - 1][j] * hedgeRatios[openIndex][j] for j in range(len(hedgeRatios[openIndex]))]))
            position_sizes.append(priceY[i - 1] + sum([priceX[i - 1][j] * hedgeRatios[openIndex][j] 
                                                   for j in range(len(hedgeRatios[openIndex]))]))
            openIndex = 0
        
        curr_position = p
   
    return pd.Series(profitY) * lot_size, pd.Series(profitX) * lot_size, pd.Series(position_sizes) * lot_size

def calculateCumulativeProfit(profit1, profit2):
    return np.cumsum(profit1 + profit2)

def tuneBBParameters(data, lookbacks, z_threshs, ticker_list):
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
            slopes=state_means[:, np.arange(0, len(ticker_list), 1)]
            
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
            backtest_data = createSignal(backtest_data)
            backtest_data['position'] = backtest_data['signal'].shift(1).fillna(0)
            hedge_ratios = np.asarray([slopes.T[i][lookback - 1:] for i in range(len(slopes.T))])
            
            profitY, profitX, pos_size = calculateDollarProfit(backtest_data['position'].values, backtest_data['qqqclose'].values, 
                                backtest_data[ticker_list].values, 
                                hedge_ratios.T.round(3), 
                                lot_size = 1000)
            
            cumulative_profit = calculateCumulativeProfit(profitY, profitX).iloc[-1]
            backtest_data['returns'] = (profitY + profitX) / pos_size
            cumulative_returns = np.cumprod(1 + backtest_data['returns']).iloc[-1]
            
            backtest_data['datetime'] = pd.to_datetime(backtest_data['datetime'])
            dailyReturns = calculateDailyReturns(backtest_data[['datetime', 'returns']])
            annualizedSharpe = calculateAnnualizedSharpeRatio(dailyReturns)
            
            results[(lookback, z_thresh)] = [cumulative_profit, cumulative_returns, annualizedSharpe]

            if counter % 10 == 0 or counter == size:
                print(counter, "done, out of", size)
                
    return dict(sorted(results.items(), key=lambda item: item[1][1], reverse=True)) # sort by annualized Sharpe

def calculateDailyReturns(minuteRets):
    minuteRets['dayperiod'] = minuteRets['datetime'].dt.to_period('D')
    minuteRets['returns'] = minuteRets['returns'] + 1
    dailyreturns = minuteRets.groupby('dayperiod')['returns'].apply(lambda x: x.cumprod().iloc[-1] - 1)
    return dailyreturns

def calculateAnnualizedSharpeRatio(dailyRets):
    return np.sqrt(252) * (dailyRets.mean() / dailyRets.std())
