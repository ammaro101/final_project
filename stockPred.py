# https://pypi.org/project/yfinance/ (""" it's an open-source tool that uses Yahoo's publicly available APIs, and is intended for research and educational purposes. """)
# import yfinance, our data source
import yfinance as yf

# import pandas and numpy
import pandas as pd 
import numpy as np

# import from tensorflow
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import SimpleRNN, Dense, LSTM, Input, GRU, SeparableConv1D, BatchNormalization, MaxPooling1D, add, Layer, concatenate
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.saving import register_keras_serializable

# import from keras_tuner
from keras_tuner import HyperModel, Hyperband, RandomSearch, Tuner, Oracle

# import from scikit-learn
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, r2_score, confusion_matrix, ConfusionMatrixDisplay

# https://pypi.org/project/pandas-ta/ ("""An easy to use Python 3 Pandas Extension with 130+ Technical Analysis Indicators. Can be called from a Pandas DataFrame or standalone""")
# import pandas-ta
import pandas_ta as ta

# import matplotlib for data visualisation
import matplotlib.pyplot as plt

# import from IPython library to be able to display images
from IPython.display import Image, display


# this library allow us to calculate how long a process would take 
from datetime import datetime

###########################################################################################################
# insert the stock symbols into a list
symbols_list = ['PFE', 'ROP', 'XYL', 'CPAY', 'INCY']


# define a function to load the data from source (yfinance API), and save it as a csv to local storage
def loadData(symbols=symbols_list, period='10y', interval='1wk'):
    
    # the timestamp column name is different depending on the interval, this will set the timestamp to the appropriate value based on the interval
#     if interval in ['1d', '5d', '1wk', '1mo', '3mo']: 
#         timestamp = 'Date'
#     else:
#         timestamp = 'Datetime'
    
    try:
        # load the the dataframe from the csv file if it already exist
        df = pd.read_csv(f'{period}_{interval}_stocks_data.csv')
        
        # the timestamp column name is different depending on the interval, this will set the timestamp to the appropriate value based on the interval
        if 'Date' in df.columns:
            df.set_index(['Date', 'Ticker'], inplace=True)
        else:
            df.set_index(['Datetime', 'Ticker'], inplace=True)
        
        print("Data loaded from directory")
        
    except FileNotFoundError:
        # print a message stating the data does not already exists and need to be downloaded from yfinance
        print(f"There is no {period}_{interval}_stocks_data.csv. Data will be downloaded from yfinance.")
        
        # download the data from source and store it in the stock_data variable which will hold the data as a pandas dataframe
        stocks_data =  yf.download(symbols, period=period, interval=interval)

        # reshape the dataframe as a multi-level index dataframe
        stocks_data = stocks_data.stack()

        # source: https://www.statology.org/pandas-change-column-names-to-lowercase/
        # convert column names to lowercase
        stocks_data.columns = stocks_data.columns.str.lower()

        # save the dataframe to a csv file (Save the data to a CSV so we don't have to make any extra unnecessary requests to the API every time we reload the notebook)
        stocks_data.to_csv(f'{period}_{interval}_stocks_data.csv', index=True)

        # load the the dataframe from the csv file
        df = pd.read_csv(f'{period}_{interval}_stocks_data.csv')
        
        if 'Date' in df.columns:
            df.set_index(['Date', 'Ticker'], inplace=True)
        else:
            df.set_index(['Datetime', 'Ticker'], inplace=True)

    finally: 
        # create a dict to store the dataframe of each unique symbol where keys are symbol, values are dataframes
        df_dict = {}

        # iterate over the symbols
        for symbol in symbols:

            # source of inspiration https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.xs.html [11]
            # extract the specific stock data at the 'Ticker' level of this multi index dataframe and save it as a dataframe
            symbol_df = df.xs(symbol, axis=0, level='Ticker', drop_level=True)

            # store the datafram into the df_dict
            df_dict[symbol] = symbol_df

        # return the dictionary
        return df_dict
    
###########################################################################################################
# create a function that takes a dataframe and create 'next_close' column based on its 'close' column
def get_next_close(_df):
    
    # create the 'next_close' column to be equal to the next closing price
    # this can be accomplished easily by shifting the close column backward by 1
    return _df['close'].shift(-1)

# create a function that returns 1 if the the next closing price is higher than current closing price and 0 otherwise.
def assign_trend(row):
    if row['next_close'] > row['close']:
        return 1
    elif row['next_close'] < row['close']:
        return 0
    else: # if the next value is missing then return NaN
        return np.nan

# create a function that add the target columns to the dataframe
def add_targets(_df):
    
    # add the next_close column to the dataframe
    _df['next_close'] = get_next_close(_df)
    
    # add the trend column to the dataframe
    _df['trend'] = _df.apply(assign_trend, axis=1)
    
    # drop the NaN values
    _df.dropna(inplace=True)
    
    # fix the 'trend' data type to be int
    _df = _df.astype({'trend': int})
    
    return _df

###########################################################################################################
# for the time being let's create a function that add all the technical indicators we want to a df
def add_technical_indicators(_df):
    
    ##### indicators based on the closing price ##### index range: 6:36
    # apply macd on the close column in a df and add it to the dataframe    
    macd = ta.macd(_df['close'])
    # The MACD (Moving Average Convergence/Divergence) is a popular indicator to that is used to identify a trend
    _df.insert(6, 'macd', macd.iloc[:,0])
    # Histogram is the difference of MACD and Signal
    _df.insert(7, 'macd_histogram', macd.iloc[:,1])
    # Signal is an EMA (exponential moving average) of MACD
    _df.insert(8, 'macd_signal', macd.iloc[:,2])
    
    # apply RSI on the Close column in a df and add it to the dataframe    
    # RSI (Relative Strength Index) is popular momentum oscillator. Measures velocity and magnitude a trend
    rsi = ta.rsi(_df['close'])
    _df.insert(9, 'rsi', rsi)

    # apply SMA on the Close column in a df and add it to the dataframe    
    # SMA (Simple Moving Average) is the classic moving average that is the equally weighted average over n periods.
    sma = ta.sma(_df['close'])
    _df.insert(10, 'sma', sma)

    # apply EMA on the Close column in a df and add it to the dataframe    
    # EMA (Exponential Moving Average). The weights are determined by alpha which is proportional to it's length.
    ema = ta.ema(_df['close'])
    _df.insert(11, 'ema', ema)
    
    ######## repeat the same proccess for all the technical indicators we want to include ##########
    # bbands: A popular volatility indicator by John Bollinger.
    bbands = ta.bbands(_df['close'])
    _df.insert(12, 'bbands_lower', bbands.iloc[:,0])
    _df.insert(13, 'bbands_mid', bbands.iloc[:,1])
    _df.insert(14, 'bbands_upper', bbands.iloc[:,2])
    _df.insert(15, 'bbands_bandwidth', bbands.iloc[:,3])
    _df.insert(16, 'bbands_percent', bbands.iloc[:,4])
    
    # dema: The Double Exponential Moving Average attempts to a smoother average with less lag than the normal Exponential Moving Average (EMA).
    dema = ta.dema(_df['close'])
    _df.insert(17, 'dema', dema)
    
    # tema: A less laggy Exponential Moving Average.
    tema = ta.tema(_df['close'])
    _df.insert(18, 'tema', tema)

    # roc: Rate of Change is an indicator is also referred to as Momentum. It is a pure momentum oscillator that measures the percent change in price with the previous price 'n' (or length) periods ago.
    roc = ta.roc(_df['close'])
    _df.insert(19, 'roc', roc)
    
    # mom: Momentum is an indicator used to measure a security's speed (or strength) of movement.  Or simply the change in price.
    mom = ta.mom(_df['close'])
    _df.insert(20, 'mom', mom)
    
    # kama: Developed by Perry Kaufman, Kaufman's Adaptive Moving Average (KAMA) is a moving average designed to account for market noise or volatility. KAMA will closely follow prices when the price swings are relatively small and the noise is low. KAMA will adjust when the price swings widen and follow prices from a greater distance. This trend-following indicator can be used to identify the overall trend, time turning points and filter price movements.
    kama = ta.kama(_df['close'])
    _df.insert(21, 'kama', kama)
                       
    # trix: is a momentum oscillator to identify divergences.
    trix = ta.trix(_df['close'])
    _df.insert(22, 'trix', trix.iloc[:,0])
    _df.insert(23, 'trixs', trix.iloc[:,1])
    
    # hma: The Hull Exponential Moving Average attempts to reduce or remove lag in moving averages.
    hma = ta.hma(_df['close'])
    _df.insert(24, 'hma', hma)
    
    # alma: The ALMA moving average uses the curve of the Normal (Gauss) distribution, which can be shifted from 0 to 1. This allows regulating the smoothness and high sensitivity of the indicator. Sigma is another parameter that is responsible for the shape of the curve coefficients. This moving average reduces lag of the data in conjunction with smoothing to reduce noise.
    alma = ta.alma(_df['close'])
    _df.insert(25, 'alma', alma)
    
    # apo: The Absolute Price Oscillator is an indicator used to measure a security's momentum.  It is simply the difference of two Exponential Moving Averages (EMA) of two different periods. Note: APO and MACD lines are equivalent.
    apo = ta.apo(_df['close'])
    _df.insert(26, 'apo', apo)
    
    # cfo: The Forecast Oscillator calculates the percentage difference between the actualprice and the Time Series Forecast (the endpoint of a linear regression line).
    cfo = ta.cfo(_df['close'])
    _df.insert(27, 'cfo', cfo)
    
    # cg: The Center of Gravity Indicator by John Ehlers attempts to identify turning points while exhibiting zero lag and smoothing.
    cg = ta.cg(_df['close'])
    _df.insert(28, 'cg', cg)
    
    # cmo: Attempts to capture the momentum of an asset with overbought at 50 and oversold at -50.
    cmo = ta.cmo(_df['close'])
    _df.insert(29, 'cmo', cmo)
    
    # coppock: Coppock Curve (originally called the "Trendex Model") is a momentum indicator is designed for use on a monthly time scale.  Although designed for monthly use, a daily calculation over the same period can be made, converting the periods to 294-day and 231-day rate of changes, and a 210-day weighted moving average.
    coppock = ta.coppock(_df['close'])
    _df.insert(30, 'coppock', coppock)
    
    # cti: The Correlation Trend Indicator is an oscillator created by John Ehler in 2020. It assigns a value depending on how close prices in that range are to following a positively- or negatively-sloping straight line. Values range from -1 to 1. This is a wrapper for ta.linreg(close, r=True).
    cti = ta.cti(_df['close'])
    _df.insert(31, 'cti', cti)
    
    # decay: Creates a decay moving forward from prior signals like crosses. The default is "linear". Exponential is optional as "exponential" or "exp".
    decay = ta.decay(_df['close'])
    _df.insert(32, 'decay', decay)
    
    # decreasing: Returns True if the series is decreasing over a period, False otherwise. If the kwarg 'strict' is True, it returns True if it is continuously decreasing over the period. When using the kwarg 'asint', then it returns 1 for True or 0 for False.
    decreasing = ta.decreasing(_df['close'])
    _df.insert(33, 'decreasing', decreasing)
    
    # ebsw: This indicator measures market cycles and uses a low pass filter to remove noise. Its output is bound signal between -1 and 1 and the maximum length of a detected trend is limited by its length input.
    ebsw = ta.ebsw(_df['close'])
    _df.insert(34, 'ebsw', ebsw)
    
    # entropy: Introduced by Claude Shannon in 1948, entropy measures the unpredictability of the data, or equivalently, of its average information. A die has higher entropy (p=1/6) versus a coin (p=1/2).
    entropy = ta.entropy(_df['close'])
    _df.insert(35, 'entropy', entropy)
    
    
    ##### indicators based on the high and lows of the price ##### range= 36:67
    
    # aberration: A volatility indicator
    aberration = ta.aberration(_df['high'], _df['low'], _df['close'])
    _df.insert(36, 'aberration_zg', aberration.iloc[:,0])
    _df.insert(37, 'aberration_sg', aberration.iloc[:,1])
    _df.insert(38, 'aberration_xg', aberration.iloc[:,2])
    _df.insert(39, 'aberration_atr', aberration.iloc[:,3])
    
    # adx:  Average Directional Movement is meant to quantify trend strength by measuring the amount of movement in a single direction.    
    adx = ta.adx(_df['high'], _df['low'], _df['close'])
    _df.insert(40, 'adx_adx', adx.iloc[:,0])
    _df.insert(41, 'adx_dmp', adx.iloc[:,1])
    _df.insert(42, 'adx_dmn', adx.iloc[:,2])

    # atr: Averge True Range is used to measure volatility, especially volatility caused by gaps or limit moves.
    atr = ta.atr(_df['high'], _df['low'], _df['close'])
    _df.insert(43, 'atr', atr)
    
    # stoch: The Stochastic Oscillator (STOCH) was developed by George Lane in the 1950's. He believed this indicator was a good way to measure momentum because changes in momentum precede changes in price.
    stoch = ta.stoch(_df['high'], _df['low'], _df['close'])
    _df.insert(44, 'stoch_k', stoch.iloc[:,0])
    _df.insert(45, 'stoch_d', stoch.iloc[:,1])
    
    # Supertrend: is an overlap indicator. It is used to help identify trend direction, setting stop loss, identify support and resistance, and/or generate buy & sell signals.
    supertrend = ta.supertrend(_df['high'], _df['low'], _df['close'])
    _df.insert(46, 'supertrend_trend', supertrend.iloc[:,0])
    _df.insert(47, 'supertrend_direction', supertrend.iloc[:,1])
    
    # cci: Commodity Channel Index is a momentum oscillator used to primarily identify overbought and oversold levels relative to a mean.
    cci = ta.cci(_df['high'], _df['low'], _df['close'])
    _df.insert(48, 'cci', cci)
    
    # aroon: attempts to identify if a security is trending and how strong.
    aroon = ta.aroon(_df['high'], _df['low'])
    _df.insert(49, 'aroon_up', aroon.iloc[:,0])
    _df.insert(50, 'aroon_down', aroon.iloc[:,1])
    _df.insert(51, 'aroon_osc', aroon.iloc[:,2])
    
    # natr: Normalized Average True Range attempt to normalize the average true range.
    natr = ta.natr(_df['high'], _df['low'], _df['close'])
    _df.insert(52, 'natr', natr)
    
    # William's Percent R is a momentum oscillator similar to the RSI that attempts to identify overbought and oversold conditions.
    willr = ta.willr(_df['high'], _df['low'], _df['close'])
    _df.insert(53, 'willr', willr)
    
    # vortex: Two oscillators that capture positive and negative trend movement.
    vortex = ta.vortex(_df['high'], _df['low'], _df['close'])
    _df.insert(54, 'vortex_vip', vortex.iloc[:,0])
    _df.insert(55, 'vortex_vim', vortex.iloc[:,1])
    
    # hlc3: the average of high, low, and close prices
    hlc3 = ta.hlc3(_df['high'], _df['low'], _df['close'])
    _df.insert(56, 'hlc3', hlc3)
    
    # ohlc4: the average of open, high, low, and close prices
    ohlc4 = ta.ohlc4(_df['open'], _df['high'], _df['low'], _df['close'])
    _df.insert(57, 'ohlc4', ohlc4)
    
    # accbands: Acceleration Bands created by Price Headley plots upper and lower envelope bands around a simple moving average.
    accbands = ta.accbands(_df['high'], _df['low'], _df['close'])
    _df.insert(58, 'accbands_lower', accbands.iloc[:,0])
    _df.insert(59, 'accbands_mid', accbands.iloc[:,1])
    _df.insert(60, 'accbands_upper', accbands.iloc[:,2])

    # chop: The Choppiness Index was created by Australian commodity trader E.W. Dreiss and is designed to determine if the market is choppy (trading sideways) or not choppy (trading within a trend in either direction). Values closer to 100 implies the underlying is choppier whereas values closer to 0 implies the underlying is trending.
    chop = ta.chop(_df['high'], _df['low'], _df['close'])
    _df.insert(61, 'chop', chop)
    
    # dm: The Directional Movement was developed by J. Welles Wilder in 1978 attempts to determine which direction the price of an asset is moving. It compares prior highs and lows to yield to two series +DM and -DM.
    dm = ta.dm(_df['high'], _df['low'])
    _df.insert(62, 'dm_positive', dm.iloc[:,0])
    _df.insert(63, 'dm_negative', dm.iloc[:,1])

    # donchian: Donchian Channels are used to measure volatility, similar to Bollinger Bands and Keltner Channels.
    donchian = ta.donchian(_df['high'], _df['low'])
    _df.insert(64, 'donchian_lower', donchian.iloc[:,0])
    _df.insert(65, 'donchian_mid', donchian.iloc[:,1])
    _df.insert(66, 'donchian_upper', donchian.iloc[:,2])
    
    
    ##### indicators based on the volume of the price ##### range= 67:72
    
    # obv: On Balance Volume is a cumulative indicator to measure buying and selling pressure.
    obv = ta.obv(_df['close'], _df['volume'])
    _df.insert(67, 'obv', obv)
    
    # vwma: Volume Weighted Moving Average.
    vwma = ta.vwma(_df['close'], _df['volume'])
    _df.insert(68, 'vwma', vwma)
    
    # adosc: Accumulation/Distribution Oscillator indicator utilizes Accumulation/Distribution and treats it similarily to MACD or APO.
    adosc = ta.adosc(_df['high'], _df['low'], _df['close'], _df['volume'])
    _df.insert(69, 'adosc', adosc)
    
    # cmf: Chailin Money Flow measures the amount of money flow volume over a specific period in conjunction with Accumulation/Distribution.
    cmf = ta.cmf(_df['high'], _df['low'], _df['close'], _df['volume'])
    _df.insert(70, 'cmf', cmf)
    
    # efi: Elder's Force Index measures the power behind a price movement using price and volume as well as potential reversals and price corrections.
    efi = ta.efi(_df['close'], _df['volume'])
    _df.insert(71, 'efi', efi)


    #### we can add more technical indicators if we want using the same process ####
    
    # remove the NaN values and return the new dataframe
    _df.dropna(inplace=True)
    
    return _df


###########################################################################################################

# create a function that takes a dictionary of dataframes as input and add the targets and features to them 
def add_targets_and_indicators(_dfs):
    
    # iterate over the dataframes in the dictionary
    for symbol in _dfs.keys():
        
        # copy the dataframe
        _df = _dfs[symbol].copy(deep=True)
        
        # add target columns to the copied dataframe
        _df = add_targets(_df)
        
        # add technical indicators to the copied dataframe
        _df = add_technical_indicators(_df)
        
        # replace the original dataframe with the new dataframe
        _dfs[symbol] = _df
    
    # return the new dataframes dictionary
    return _dfs

###########################################################################################################
# create a function to computes the ratio of trend = 1 for each individual dataframe in a dictionary 
def calculate_data_balance(_dfs):
    
    # store the total ratio of trend 1 of all the dataframes
    total = 0
    
    # iterate over the dataframes in the dictionary
    for symbol in _dfs.keys():
        
        # get the number of values where trend = 1
        trend_1 = _dfs[symbol]['trend'].value_counts()[1]
        
        # get the total number of rows in the dataframe
        row_num = _dfs[symbol].shape[0]
        
        # percentage of 'trend up' to the whole column
        trend_1_ratio = trend_1/row_num
        
        # print the ratio to the screen
        print(f"The Trend up ratio of {symbol} is {trend_1_ratio} for {row_num} rows")
        
        # add the ratio to total
        total += trend_1_ratio
        
    # get the average trend up ratio
    average = total / len(_dfs.keys())
    
    # print the average ratio
    print(f"The Average Trend up ratio is: {average}")
    
###########################################################################################################
def calculate_common_sense_baseline(_dfs):
    
    # store the total common_sense_score for all the dataframes
    total = 0
    
    # iterate over the dataframes in the dictionary
    for symbol in _dfs.keys():
        
        # since the common sense will be to assume the trend next is going to be the same as the trend now, we will shift the trend
        # forward by one, this will give us a column that matches the common sense assumption we set
        common_sense = _dfs[symbol]['trend'].shift(1)

        # measure the average of when the common sense (naive) prediction matches the actual 'trend'
        common_sense_score = (common_sense == _dfs[symbol]['trend']).mean()
        
        # print the score to the screen
        print(f"The common sense score of {symbol} is: {common_sense_score}")
        
        # add the score to the total
        total += common_sense_score
    
    # get the average score
    average = total / len(_dfs.keys())
    
    # print the average score
    print(f"The Average common sense score is: {average}")
    
    

###########################################################################################################
# create a function to apply a given scaler to the features
def apply_scaler(scaler, features):
    
    # set the training and test ratio to be 70-30
    training_ratio = int(len(features) * 0.8)

    # devide the feature set into training and test set
    X_train, X_test = features[:training_ratio], features[training_ratio:]
    
    # apply a scaler on the training and test sets in isolation so we don't allow the test set to influence the scaling process, which reduces the likelihood of overfitting 
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # concat the two scaled sets into one
    X = np.concatenate((X_train_scaled, X_test_scaled), axis=0)

    # return the scaled features
    return X


# source of isnpiration: https://stackoverflow.com/questions/47945512/how-to-reshape-input-for-keras-lstm?rq=4 [13]
# create a function to reshape X and y into sequences of x timesteps
def create_seqs(features, target, num_rows):
    # create 2 empty lists to store the newly shaped features and target lists
    X, y = [], []
    
    # iterate over the features
    for i in range(len(features) - num_rows):
        # create indexes of the start and end of each sequence
        seq_s = i
        seq_e = i + num_rows
        
        # the ith sequence will be a slice of the features between the indexes, create it and add it to X
        xi = features[seq_s : seq_e]
        X.append(xi)
        
        # do the same for the target and add it to y
        yi = target[seq_e]
        y.append(yi)
    
    # return the X and y as numpy arraies
    return np.array(X), np.array(y)


# create a function to convert a dataframe into training, validation and test sets
def create_train_vald_test_sets(_df, scaler, target="classification", timesteps=6):

    # reset the index
    _df.reset_index(inplace = True)
    
    if 'Date' in _df.columns:
        # drop the Date column as it's not necessary for now
        _df.drop(['Date'], axis=1, inplace=True)
    else:
        # drop the Datetime column as it's not necessary for now
        _df.drop(['Datetime'], axis=1, inplace=True)

    # set the features set
    X = _df.iloc[:, :-2]
    
    # set the target 
    if (target == "classification"):
        # trend is the target for classification
        y = _df.iloc[:, -1]
    else:
        # next_close is the target for regression
        y = _df.iloc[:, -2]

    # apply a scaler on the features set
    X = apply_scaler(scaler, X)
    
    # create sequences
    X_seq, y_seq = create_seqs(X, y, timesteps)
    
    # source of inspiration: https://www.tensorflow.org/api_docs/python/tf/keras/utils/to_categorical [14]
    # use to_categorical from tf to converts the target (trend) to binary class matrix, this will help us assign confidences to the classification prediction
    if (target == "classification"):
        y_seq = to_categorical(y_seq)

    # devide the data into a training set and a test set in 70-30 ratio
    training_ratio = int(len(X) * 0.8)
    
    # add a vaidation ratio at 20% of the data, this will leave 10% as test
#     validation_ratio = int(len(X) * 0.2)
    
#     X_train, X_vald, X_test = X_seq[:training_ratio], X_seq[training_ratio:training_ratio + validation_ratio], X_seq[training_ratio + validation_ratio:]
#     y_train, y_vald, y_test = y_seq[:training_ratio], y_seq[training_ratio:training_ratio + validation_ratio], y_seq[training_ratio + validation_ratio:]

    X_train, X_test = X_seq[:training_ratio], X_seq[training_ratio:]
    y_train, y_test = y_seq[:training_ratio], y_seq[training_ratio:]

    # return the sets and the last_date
#     return X_train, X_vald, X_test, y_train, y_vald, y_test
    return X_train, X_test, y_train, y_test



# create a function that takes a dict of dataframes, and return a dict of training, validation and testing datasets
def prepare_data_to_train(dfs_dict, scaler, target, timesteps):
    
    # create a dict of dicts to store training, validation and test sets for each stock
    sets_dict = {}
    
    # iterate over each dataframe in the dictionary
    for symbol in dfs_dict.keys():
        
        # convert the dataframe into training, validation and test sets
#         X_train, X_vald, X_test, y_train, y_vald, y_test = create_train_vald_test_sets(dfs_dict[symbol].copy(deep=True), scaler, target, timesteps)
        X_train, X_test, y_train, y_test = create_train_vald_test_sets(dfs_dict[symbol].copy(deep=True), scaler, target, timesteps)
        
        # create a dict of the sets and add it to the sets_dict
#         sets_dict[symbol] = {
#             'X_train': X_train, 'X_vald': X_vald, 'X_test': X_test, 
#             'y_train': y_train, 'y_vald': y_vald, 'y_test': y_test
#         }
        sets_dict[symbol] = {
            'X_train': X_train, 'X_test': X_test, 
            'y_train': y_train, 'y_test': y_test
        }
    
    # return the sets
    return sets_dict


###########################################################################################################
# helper function to measure how long a process would take
def get_time():
    return datetime.now()

# source of inspiration: Introduction to the Keras Tuner, https://www.tensorflow.org/tutorials/keras/keras_tuner
# create a function to train static and hyperparameters optimized models on multiple datasets and archive them
def create_models_archive(_create_model, _datasets_dict, _model_type='classification', _tuner=None, _epochs=50, _model_name='model', _project_name='proj'):
    
    # get time before the training
    start = get_time()
    
    # create the models archive dictionary
    archive = {}
    
    # iterate over the symbols in the dictionary
    for symbol in _datasets_dict.keys():
        
        # initiate a dict for the symbol
        archive[symbol] = {}
        
        # setup the data to be passed to the model
        X_train, y_train = _datasets_dict[symbol]['X_train'], _datasets_dict[symbol]['y_train']
#         X_vald, y_vald = _datasets_dict[symbol]['X_vald'], _datasets_dict[symbol]['y_vald']
        X_test, y_test = _datasets_dict[symbol]['X_test'], _datasets_dict[symbol]['y_test']

        # source: 7.2. Inspecting and monitoring deep-learning models using Keras callba- acks and TensorBoard
        # source: EarlyStopping, https://keras.io/api/callbacks/early_stopping/
        # Stop training when a monitored metric has stopped improving.
        # monitor: Quantity to be monitored.
        # min_delta: Minimum change in the monitored quantity to qualify as an improvement (we will use the default value)
        # patience: Number of epochs with no improvement after which training will be stopped.
#         stop_early = EarlyStopping(monitor='accuracy', 
#                                    patience=20)
        stop_early = EarlyStopping(monitor='val_loss', 
                                   patience=50)
        
        # source: ReduceLROnPlateau, https://keras.io/api/callbacks/reduce_lr_on_plateau/
        # Reduce learning rate when a metric has stopped improving.
#         reduce_lr =  ReduceLROnPlateau(monitor='accuracy', 
#                                        factor=0.1, 
#                                        patience=10)
        reduce_lr =  ReduceLROnPlateau(monitor='val_loss', 
                                       factor=0.1, 
                                       patience=30)
        
        # initialize the model
#         model = _create_model(X_train.shape)   
        model = _create_model        
        
        # source: Hyperband Tuner, https://keras.io/api/keras_tuner/tuners/hyperband/
        ### Instantiate the tuner and perform hypertuning
        # pass the model which is an Instance of HyperModel class (or callable that takes hyperparameters and returns a Model instance)
        # objective is the direction of the optimization
        # max_epochs is the maximum number of epochs to train one model, it is recommended to set this to a value slightly higher than expected then use early stopping callback during training (we will use the default value)
        # factor: the reduction factor for the number of epochs and number of models for each bracket (we will use the default value)
        # hyperband_iterations: the number of times to iterate over the full Hyperband algorithm. One iteration will run approximately max_epochs * (math.log(max_epochs, factor) ** 2) cumulative epochs across all trials. (we will use the default value)
        # we will set the seed to make our work easier to replicate by others 
        # directory is and project name is the path where it will store the trails data results, this will make it much faster to rerun the training process if we need to
        if _tuner:
            tuner = _tuner(model, 
                        objective='val_mae', 
                        max_epochs=10, 
                        factor=3, 
                        hyperband_iterations=1, 
                        seed=101, 
                        directory='keras_tuner_models', 
                        project_name=f'{_project_name}/{symbol}')

            # Run the hyperparameter search. The arguments for the search method are the same as those used for tf.keras.model.fit in addition to the callback above.
#             tuner.search(X_train, y_train, 
#                          epochs=_epochs, 
#                          validation_data=(X_vald, y_vald), 
#                          callbacks=[stop_early, reduce_lr], 
#                         verbose=0)
            
            tuner.search(X_train, y_train, 
                         validation_split=0.2, 
                         epochs=_epochs, 
                         callbacks=[stop_early])

            # source: The base Tuner class, https://keras.io/api/keras_tuner/tuners/base_tuner/
            # get_best_hyperparameters Returns the best hyperparameters, as determined by the objective as a list sorted from the best to the worst.
            # Get the optimal hyperparameters
            best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

            # Display tuning results summary. prints a summary of the search results including the hyperparameter values and evaluation results for each trial.
            results_summary = tuner.results_summary()
        
            # Build the model with the optimal hyperparameters and train it on the data for 50 epochs
            model = tuner.hypermodel.build(best_hps)
            
            print("best_hps: ", best_hps.values)
            print("model summary")
            model.summary()
                    
        
        # fit the model
#         history = model.fit(X_train, y_train, 
#                             epochs=_epochs, 
#                             batch_size=32, 
#                             validation_data=(X_vald, y_vald), 
#                             callbacks=[stop_early, reduce_lr], 
#                             verbose=0)    
        history = model.fit(X_train, y_train, 
                            epochs=_epochs, 
                            batch_size=32, 
                            validation_split=0.2, 
                            callbacks=[stop_early, reduce_lr], 
                            verbose=1)
        
        # get the history of accuracy during training as a list
#         if _model_type == "classification":
#             val_prec_per_epoch = history.history['val_accuracy']
#         else:
#             val_prec_per_epoch = history.history['val_loss']

#         # get the index of the highest val_precision from this list, we will use this index to set the epochs values during the training
#         best_epoch = val_prec_per_epoch.index(max(val_prec_per_epoch)) + 1
#         print('Best epoch: %d' % (best_epoch,))

#         ### train the model based on the results of the hyperparameter optimization process 
#         # Re-instantiate the hypermodel and train it with the optimal number of epochs from above.
#         if _tuner:
#             hypermodel = tuner.hypermodel.build(best_hps)
#         else:
# #             hypermodel = _create_model(X_train.shape)
#             hypermodel = model

#         # Retrain the model
#         hypermodel_history = hypermodel.fit(X_train, y_train, 
#                                             validation_data=(X_vald, y_vald), 
#                                             epochs=best_epoch, 
#                                             verbose=0)

        # evaluate the model
        hypermodel = model
        model_evaluation = hypermodel.evaluate(X_test, y_test, verbose=0)
#         model_evaluation = model.evaluate(X_test, y_test, verbose=0)
            
        # get predictions from the model given the test and validation set
        y_pred = hypermodel.predict(X_test)
#         y_pred_vald = hypermodel.predict(X_vald)

        # evaluate the model whether the model type is classification or regression
        if _model_type == "classification":     
            # convert the predictions and test set to be in the shape of a vector of labels
            y_pred_labels = np.argmax(y_pred, axis=1)
            y_test_labels = np.argmax(y_test, axis=1)
            
            # do the same for the validation set
#             y_pred_vald_labels = np.argmax(y_pred_vald, axis=1)
#             y_vald_labels = np.argmax(y_vald, axis=1)
        else:
            # get the R2 of the model
            r2 = r2_score(y_test, y_pred)
        
        # source of inspiration: https://www.tensorflow.org/tutorials/keras/save_and_load
        # save model to device
        hypermodel.save(f'models/{_model_name}_{symbol}.keras')        
        
        # store the model in the associated symbol dict
        archive[symbol]['model'] = load_model(f'models/{_model_name}_{symbol}.keras')
        
        # evaluate the model on the test_set and store it in the associated symbol dict
        archive[symbol]['evaluation'] = model_evaluation
        
        # store the best model hyperparameters in the associated symbol dict
        archive[symbol]['hyperparameters'] = best_hps if _tuner else None
        
        # store the the best model training and validation accuracy history
#         archive[symbol]['training_history'] = hypermodel_history
        archive[symbol]['training_history'] = history

        # store the the best model prediction labels and true labels
        archive[symbol]['y_pred_labels'] = y_pred_labels if _model_type == "classification" else None
        archive[symbol]['y_test_labels'] = y_test_labels if _model_type == "classification" else None
#         archive[symbol]['y_pred_vald_labels'] = y_pred_vald_labels if _model_type == "classification" else None
#         archive[symbol]['y_vald_labels'] = y_vald_labels if _model_type == "classification" else None
        
        # store the R2 score for regression model
        archive[symbol]['r2'] = r2 if _model_type == "regression" else None
        
        # store the the tunning proccess results_summary
        archive[symbol]['results_summary'] = results_summary if _tuner else None
        
        # get time after the training
        end = get_time()
        
        # print the trainig duration
        print(f"training for the {symbol} model was done in: {end - start}")
        
    return archive


###########################################################################################################
# create a function to calculate the precision, recall, and accuracy of a model
def calculate_precision_recall_fscore(y_test, y_pred):
    # source of inspiration: https://stackoverflow.com/questions/48987959/classification-metrics-cant-handle-a-mix-of-continuous-multioutput-and-multi-la [15]
    # get precision, recall, and fscore and return them
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
    return precision, recall, fscore

# create a function that produce a confusion matrix of a model
def create_confusion_matrix(y_test, y_pred):
    # source of inspiration: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html [16]
    conf_mat = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(conf_mat)
    disp.plot()
    plt.show()
       
# create a function that graph the traing vs validation loss
def create_train_vald_graph(training_history): 
    # source of the code snippet[17]
    # get the training and validation loss
    loss = training_history['loss']
#     val_loss = training_history['val_loss']
    accuracy = training_history['accuracy']
    
    # we can get the number of epochs simply from the length of the loss list
    epochs = range(1, len(loss) + 1)
    
    ### plot
    plt.figure()
    
    # plot the training loss against the epochs
    plt.plot(epochs, loss, 'bo', label='Training loss')
    
    # plot the validation loss against the epochs
#     plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.plot(epochs, accuracy, 'b', label='accuracy loss')
    
    # add title and legend
    plt.title('Training and accuracy loss')
    plt.legend()
    plt.show()


# create a function that evaluate a dictionary of models
def evaluate_models_archive(_models_archive):
    
    # initialize the total metrics variables set them to 0
    total_precision = 0
    total_recall = 0
    total_fscore = 0
    total_accuracy = 0
#     total_vald_precision = 0
#     total_vald_recall = 0
#     total_vald_fscore = 0
#     total_vald_accuracy = 0
    
    # iterate over the symbols of the dictionary
    for symbol in _models_archive.keys():
        
        # get the model y_test and y_pred
        y_test = _models_archive[symbol]['y_test_labels']
        y_pred = _models_archive[symbol]['y_pred_labels']
        
        # get the model y_vald and y_vald_pred
#         y_vald = _models_archive[symbol]['y_vald_labels']
#         y_vald_pred = _models_archive[symbol]['y_pred_vald_labels']
        
        # model summary
        _models_archive[symbol]['model'].summary()
        
        # plot the model and save it to an image
        plot_path = f"{symbol}_model_plot.png"
        plot_model(_models_archive[symbol]['model'], show_shapes=True, dpi=80, to_file=plot_path)
        # disply the saved plot image
        display(Image(plot_path))
        
        # classification model metrics for test set
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, fscore = calculate_precision_recall_fscore(y_test, y_pred)
        
        # print the metrics for each model
        print(f"The {symbol} Model Classification Metrics for test set:")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F-Score: {fscore}")
        print("--------------------------------------------------------------")
        
        # add the scores for this model to the total scores
        total_precision += precision
        total_recall += recall
        total_fscore += fscore
        total_accuracy += accuracy
        
        # create confusion matrix
        create_confusion_matrix(y_test, y_pred)
        
        ##########################################
        # classification model metrics for test set
#         accuracy = accuracy_score(y_vald, y_vald_pred)
#         precision, recall, fscore = calculate_precision_recall_fscore(y_vald, y_vald_pred)
        
#         # print the metrics for each model
#         print(f"The {symbol} Model Classification Metrics for validation set:")
#         print(f"Accuracy: {accuracy}")
#         print(f"Precision: {precision}")
#         print(f"Recall: {recall}")
#         print(f"F-Score: {fscore}")
#         print("--------------------------------------------------------------")
        
#         # add the scores for this model to the total scores
#         total_vald_precision += precision
#         total_vald_recall += recall
#         total_vald_fscore += fscore
#         total_vald_accuracy += accuracy
        
#         # create confusion matrix
#         create_confusion_matrix(y_vald, y_vald_pred)
        
        ################################################################################
        
        # plot training vs validation loss
        create_train_vald_graph(_models_archive[symbol]['training_history'].history)
        
    # calculate average metrics for test set
    models_num = len(_models_archive.keys())
    average_precision = total_precision / models_num
    average_recall = total_recall / models_num
    average_fscore = total_fscore / models_num
    average_accuracy = total_accuracy / models_num

    # print the average metrics
    print(f"Average Classification Metrics for All Models on test set:")
    print(f"Average Accuracy: {average_accuracy}")
    print(f"Average Precision: {average_precision}")
    print(f"Average Recall: {average_recall}")
    print(f"Average F-Score: {average_fscore}")
    
    # calculate average metrics for validation set
#     models_num = len(_models_archive.keys())
#     average_vald_precision = total_vald_precision / models_num
#     average_vald_recall = total_vald_recall / models_num
#     average_vald_fscore = total_vald_fscore / models_num
#     average_vald_accuracy = total_vald_accuracy / models_num

#     # print the average metrics
#     print(f"Average Classification Metrics for All Models on validation set:")
#     print(f"Average Accuracy: {average_vald_accuracy}")
#     print(f"Average Precision: {average_vald_precision}")
#     print(f"Average Recall: {average_vald_recall}")
#     print(f"Average F-Score: {average_vald_fscore}")
    
    
###########################################################################################################

### create the baseline model
# source of inspiration: François Chollet (11, 2017), “Deep Learning with Python” chapter 6 [8]
# define a model class that allow us to build SimpleRNN, LSTM, or GRU models for classification or regression approches
class RNNModel:
    def __init__(self, X_train_shape, model_type='classification', layer_type='SimpleRNN'):
        self.X_train_shape = X_train_shape
        self.model_type = model_type
        self.layer_type = layer_type

    def build(self):
        # initialize a sequential model
        model = Sequential()

        # add the model layers
        # input layer
        model.add(Input(shape=(self.X_train_shape[1], self.X_train_shape[2])))

        # RNN layers
        # set the layer type
        if self.layer_type == 'SimpleRNN':
            RNN_layer = SimpleRNN
        elif self.layer_type == 'LSTM':
            RNN_layer = LSTM
        elif self.layer_type == 'GRU':
            RNN_layer = GRU
        else:
            raise ValueError('Wrong layer type provided!!! only support (SimpleRNN, LSTM, GRU)')

        # add the selected layers to the model
        model.add(RNN_layer(64, return_sequences=True))
        model.add(RNN_layer(64, return_sequences=False))

        # set the model type and compile it
        if self.model_type == 'classification':
            model.add(Dense(2, activation='softmax')) # output layer
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # compile the model
        elif self.model_type == 'regression':
            model.add(Dense(1)) # output layer
            model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae']) # compile the model
        else:
            raise ValueError('Wrong model type provided!!! only support (classification, regression)')

        return model
    
###########################################################################################################
# source of inspiration: Introduction to the Keras Tuner, https://www.tensorflow.org/tutorials/keras/keras_tuner
# constuct the model which will perform hyperparameter optimization to choose layers count, neurons counts, recurrent_dropout, optimizer_type, optimizer learning rate
class HP_RNNModel(HyperModel):

    # initialize the model upon creating a class instance
    # using a class structure instead of a function to construct the model will allow us to pass variables to it before passing it to keras tuner
    # source of inspiration on how to pass variables to the model before passing the model to keras tuner: https://github.com/JulieProst/keras-tuner-tutorial/blob/master/hypermodels.py
    def __init__(self, X_train_shape, layer_type='SimpleRNN'):
        self.X_train_shape = X_train_shape
        self.layer_type = layer_type
    
    # build the model
    def build(self, hp):
        # initialize a sequential model
        model = Sequential()

        ### add the model layers (Model hyperparameters optimization)
        # input layer
        model.add(Input(shape=(self.X_train_shape[1], self.X_train_shape[2])))

        # source: Int method, https://keras.io/api/keras_tuner/hyperparameters/
        # dynamically optimize the number of layers
        hp_layers = hp.Int(name='hp_layers', 
                           min_value=2, 
                           max_value=4, 
                           step=2)

        # for each optimized layer
        for i in range(hp_layers):
            
#             # optimize the layer type
#             hp_layer_type = hp.Choice(f'RNN_layer_{i}_type', values=['SimpleRNN', 'LSTM', 'GRU'])
#             if hp_layer_type == 'SimpleRNN':
#                 layer_type = SimpleRNN
#             elif hp_layer_type == 'LSTM':
#                 layer_type = LSTM
#             else:
#                 layer_type = GRU
                
            # select the layer type based on input
            RNN_layer = self.layer_type
            if RNN_layer == 'SimpleRNN':
                RNN_layer = SimpleRNN
            elif RNN_layer == 'LSTM':
                RNN_layer = LSTM
            else:
                RNN_layer = GRU

            # dynamically tune the number of units in each layer, select a value between 64-128
            hp_units = hp.Int(name=f'hp_units_at_hp_layer_{i}', 
                              min_value=64, 
                              max_value=128, 
                              step=64)
            
            # source: SimpleRNN layer, https://keras.io/api/layers/recurrent_layers/simple_rnn/
            # return_sequences: Boolean. Whether to return the last output in the output sequence, or the full sequence. Default: False.
            # set the return_sequences parameter to true unless it's the last layer, set it to false
            return_sequences_boolean = i != (hp_layers - 1)

            # source: Float method, https://keras.io/api/keras_tuner/hyperparameters/
            # source: SimpleRNN layer, https://keras.io/api/layers/recurrent_layers/simple_rnn/
            # recurrent_dropout: Float between 0 and 1. Fraction of the units to drop for the linear transformation of the recurrent state.
            # dynamically tune the recurrent_dropout float value
            recurrent_dropout = hp.Float(name=f'recurrent_dropout_{i}', 
                                         min_value=0.0, 
                                         max_value=0.5, 
                                         step=0.1)

            # add a simpleRNN layer and pass the optimized number of unites, recurrent_dropout, and the return_sequences boolean
            layer = RNN_layer(units=hp_units, 
                              return_sequences=return_sequences_boolean, 
                              recurrent_dropout=recurrent_dropout)
            model.add(layer)


        # add the output layer
        model.add(Dense(2, activation='softmax'))

        ### the model compiler (Algorithm hyperparameters optimization)
        # source: Choice method, https://keras.io/api/keras_tuner/hyperparameters/
        # dynamically optimize the optimizer type
#         hp_optimizer_type = hp.Choice('optimizer_type', values=['Adam', 'RMSprop', 'SGD'])
#         if hp_optimizer_type == 'Adam':
#             optimizer = Adam
#         elif hp_optimizer_type == 'RMSprop':
#             optimizer = RMSprop
#         else:
#             optimizer = SGD
            
        # set the optimizer type 
        optimizer = Adam
        

        # dynamically tune the learning rate for the optimizer
        # When sampling="log", the step is multiplied between samples.
        hp_lr = hp.Float('learning_rate', 
                         min_value=0.0001, 
                         max_value=0.01, 
                         sampling='LOG')
        hp_optimizer = optimizer(learning_rate=hp_lr)

        # compile the model
        model.compile(optimizer=hp_optimizer, 
                      loss='categorical_crossentropy', # this is the most suitable one for predictions of one-hot encoded labels
                      metrics=['accuracy'])

        # return the model
        return model
    
    # source: omalleyt12, https://github.com/keras-team/keras-tuner/issues/122
    # define a fit function which will allow us to pass an optimized value for batch_size
    # *args and **kwargs are the ones we pass through tuner.search()
    def fit(self, hp, model, *args, **kwargs):
        
        # dynamically optimize the batch_size for the training process
        hp_batch_size = hp.Choice("batch_size", [32, 64])
        return model.fit(
            *args,
            batch_size=hp_batch_size,
            **kwargs,
        )
    

###########################################################################################################