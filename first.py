#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import ta
from ta.utils import dropna
from ta.volatility import BollingerBands
from ta import add_all_ta_features
from datetime import datetime
import statistics 
from sklearn import tree
import itertools


# #### keys[0] = api_key
# #### keys[1] = secret_key

# In[2]:


keys = [] 
with open('keys.txt','r') as f:
    for row in f:
        keys.append(row)
f.close
keys = keys[0].split('-')


# In[3]:


from binance.client import Client
client = Client(keys[0], keys[1])


#  KLINE_INTERVAL_12HOUR = '12h'
#  KLINE_INTERVAL_15MINUTE = '15m'
#  KLINE_INTERVAL_1DAY = '1d'
#  KLINE_INTERVAL_1HOUR = '1h'
#  KLINE_INTERVAL_1MINUTE = '1m'
#  KLINE_INTERVAL_1MONTH = '1M'
#  KLINE_INTERVAL_1WEEK = '1w'
#  KLINE_INTERVAL_2HOUR = '2h'
#  KLINE_INTERVAL_30MINUTE = '30m'
#  KLINE_INTERVAL_3DAY = '3d'
#  KLINE_INTERVAL_3MINUTE = '3m'
#  KLINE_INTERVAL_4HOUR = '4h'
#  KLINE_INTERVAL_5MINUTE = '5m'
#  KLINE_INTERVAL_6HOUR = '6h'
#  KLINE_INTERVAL_8HOUR = '8h'

# In[4]:


def FormatTime(candles_df_date,candles_df_close_time):
    final_date = []
    final_close_time = []

    for time in candles_df_date.unique():
        final_date.append(datetime.fromtimestamp(int(time/1000)))

    for time in candles_df_close_time.unique():
        final_close_time.append(datetime.fromtimestamp(int(time/1000)))
    return final_date,final_close_time


# In[5]:


def getMetaData(name):
    candles =  client.get_klines(symbol=name,interval=Client.KLINE_INTERVAL_4HOUR)
    candles_df = pd.DataFrame(candles)
    
    final_date,final_close_time = FormatTime(candles_df[0],candles_df[6])
    
    candles_df.pop(11)
    candles_df.pop(6)
    candles_df.pop(0)
    
    candles_df.insert(0,'Date',final_date,True)
    candles_df.insert(6,'Close_time',final_close_time,True)
    
    candles_df =candles_df.rename(columns={1: "Open", 2: "High", 3: "Low", 4: "Close", 5: "Volume",
                                       6: "Close_time", 7: "Quote_asset_volume", 8: "Number_of_trades",
                                       9: "Taker_buy_base_asset_volume", 10: "Taker_buy_quote_asset_volume"})
    candles_df['Open'] = pd.to_numeric(candles_df['Open'])
    candles_df['High'] = pd.to_numeric(candles_df['High'])
    candles_df['Low'] = pd.to_numeric(candles_df['Low'])
    candles_df['Close'] = pd.to_numeric(candles_df['Close'])
    candles_df['Volume'] = pd.to_numeric(candles_df['Volume'])
    
    
    return candles_df


# In[6]:


def MACD(df,maxIndex):
    try:
#         maxIndex = df.shape[0]
        macd_bool = 0
        check_bool = 0
        temp = df['trend_macd'][maxIndex-2] - df['trend_macd_signal'][maxIndex-2]
        distance = df['trend_macd'][maxIndex-1] - df['trend_macd_signal'][maxIndex-1]
        if distance > 0:
            if distance >= temp:
                check_bool = 1
            else:
                check_bool = -1
        elif distance < 0:
            if (-1*distance) >= (-1*temp):
                check_bool = -1
            else:
                check_bool = 1
        j = maxIndex
        while (j > maxIndex - 7):
            j-=1
            if df['trend_macd'][j] == df['trend_macd_signal'][j]:
                i = j-1
                while df['trend_macd'][i] == df['trend_macd_signal'][i] and i > maxIndex-50:
                    i-=1
                if df['trend_macd'][i] > df['trend_macd_signal'][i]:
                    macd_bool = -1
                else:
                    macd_bool = 1
                break
    except:
        return('Exception : MACD Buy or Sell')
    if macd_bool == 1:
        return('MACD Says buy.')
    elif macd_bool == -1:
        return('MACD Says sell.')
    elif macd_bool == 0 and check_bool == -1:
        return('MACD Says be carefull because this chart shows that maybe it will be worse in future!')
    elif  macd_bool==0 and check_bool == 1:
        return('MACD Says be carefull because this chart shows that maybe it will be fine in future!')
    elif  macd_bool==0 and check_bool == 0:
        return('MACD Says it is not good')
    else:
        return('****')


# In[7]:


def ichimoku(df,maxIndex):
    try:
        check = 0
#         maxIndex = df.shape[0]
        conv = df['trend_ichimoku_conv'][maxIndex-1]
        base = df['trend_ichimoku_base'][maxIndex-1]
        lowCloud = df['trend_ichimoku_b'][maxIndex-1]
        highCloud = df['trend_ichimoku_a'][maxIndex-1]
        price = df['Close'][maxIndex-1]
        check = 0
        if price > lowCloud:
            if price < base:
                check = 1
            elif price > conv and conv > base:
                check = 1
        elif price < highCloud:
            if price > base:
                check = -1
            elif price < conv:
                check = -1
        if price == conv:
            if df['Close'][maxIndex-2] > df['trend_ichimoku_conv'][maxIndex-2]:
                check = 0.5
            elif df['Close'][maxIndex-2] < df['trend_ichimoku_conv'][maxIndex-2]:
                check = -0.5
        if price == lowCloud:
            if df['Close'][maxIndex-2] > df['trend_ichimoku_b'][maxIndex-2]:
                check = 0.5
            elif df['Close'][maxIndex-2] < df['trend_ichimoku_b'][maxIndex-2]:
                check = -0.5
        if price == highCloud:
            if df['Close'][maxIndex-2] > df['trend_ichimoku_a'][maxIndex-2]:
                check = 0.5
            elif df['Close'][maxIndex-2] < df['trend_ichimoku_a'][maxIndex-2]:
                check = -0.5
        if price == base:
            if df['Close'][maxIndex-2] > df['trend_ichimoku_base'][maxIndex-2]:
                check = 0.5
            elif df['Close'][maxIndex-2] < df['trend_ichimoku_base'][maxIndex-2]:
                check = -0.5
    except:
        return 'ex'
    if check == 1:
        return('Ichimoku says buy it.')
    elif check == -1:
        return('Ichimoku says sell it.')
    elif check == -0.5:
        return('Ichimoku says we have resist.')
    elif check == 0.5:
        return('Ichimoku says we have support.')
    else:
        return('Ichimoku has no word to say!')


# In[8]:


def RSI(df,maxIndex):
    try:
#         maxIndex = df.shape[0]
        if df['momentum_rsi'][maxIndex-1] > 60:
            return 'RSI Says overbought has been occur, so it is time to sell.'
        elif df['momentum_rsi'][maxIndex-1] < 40:
            return 'RSI Says oversold has been occur, so it is time to buy.'
        else:
            return 'RSI : nothing to say.'
    except:
        return 'RSI Exception'


# In[9]:


def BB(df,maxIndex):
#     maxIndex = df.shape[0]
    check = 0
    try:
        i = maxIndex
        while i > maxIndex - 7:
            i-=1
            if(df['Close'][i] == df['volatility_bbm'][i]):
                if df['Close'][i-1] > df['volatility_bbm'][i-1]:
                    check = -1
                else:
                    check = 1
                break
        i = maxIndex
        if(df['Close'][maxIndex-1] > df['volatility_bbh'][maxIndex-1]):
            check = -2
        elif(df['Close'][maxIndex-1] < df['volatility_bbl'][maxIndex-1]):
            check = 2
        if check == 1:
            return 'BB Says buy it.'
        elif check == -1:
            return 'BB Says sell it.'
        elif check == -2:
            return 'BB Says we will have pull back from up.'
        elif check == 2:
            return 'BB Says we will have pull back from down.'
        else:
            return 'BB no word to say!'
    except:
        return('exception BB')


# In[10]:


def movingAverage20_50_200(df,maxIndex):
#     maxIndex = df.shape[0]
    check = 0
    check2 = 0
    try:
        if df['EMA20'][maxIndex-1] > df['EMA50'][maxIndex-1]:
            if df['EMA50'][maxIndex-1] > df['EMA200'][maxIndex-1]:
                check = 1
        elif df['EMA20'][maxIndex-1] < df['EMA50'][maxIndex-1]:
            if df['EMA50'][maxIndex-1] > df['EMA200'][maxIndex-1]:
                 check = 0
            elif df['EMA50'][maxIndex-1] < df['EMA200'][maxIndex-1]:
                check = -1
        i = maxIndex - 20
        while i < maxIndex :
            if df['EMA20'][i] == df['EMA50'][i]:
                if df['EMA20'][i-1] < df['EMA50'][i-1]:
                    check2 = 1
                elif df['EMA20'][i-1] > df['EMA50'][i-1]:
                    check2 = -1
            if check2 == 1:
                if df['EMA50'][i] == df['EMA200'][i]:
                    if df['EMA50'][i-1] < df['EMA200'][i-1]:
                        check = 1
            elif check2 == -1:
                if df['EMA50'][i] == df['EMA200'][i]:
                    if df['EMA50'][i-1] > df['EMA200'][i-1]:
                        check = -1
            i+=1
            if check == 1:
                return 'EMA20_50_200 Says buy it.'
            elif check == -1:
                return 'EMA20_50_200 Says sell it.'
            else:
                return 'EMA20_50_200 no word to say!'
    except:
        return 'EMA:Exception'


# In[11]:


def STO(df,maxIndex):
    try:
        if df['momentum_stoch'][maxIndex-1] > 80 or df['momentum_stoch'][maxIndex-1] < df['momentum_stoch_signal'][maxIndex-1]:
            return 'STO Says overbought has been occur, so it is time to sell.'
        elif  df['momentum_stoch'][maxIndex-1] > 80 or df['momentum_stoch'][maxIndex-1] > df['momentum_stoch_signal'][maxIndex-1]:
            return 'STO Says buy it.'
        elif df['momentum_stoch'][maxIndex-1] < 20 or df['momentum_stoch'][maxIndex-1] > df['momentum_stoch_signal'][maxIndex-1]:
            return 'STO Says oversold has been occur, so it is time to buy.'
        elif df['momentum_stoch'][maxIndex-1] < 20 or df['momentum_stoch'][maxIndex-1] < df['momentum_stoch_signal'][maxIndex-1]:
            return 'STO Says sell it.'
        else:
            if df['momentum_stoch'][maxIndex-1] > df['momentum_stoch_signal'][maxIndex-1]:
                return 'STO Says buy it.'
            elif df['momentum_stoch'][maxIndex-1] < df['momentum_stoch_signal'][maxIndex-1]:
                return 'STO Says sell it.'
            else:
                if df['momentum_stoch'][maxIndex-2] > df['momentum_stoch_signal'][maxIndex-2]:
                    return 'STO Says sell it.'
                else:
                    return 'STO Says buy it.'
    except:
        return 'STO Exception'


# In[12]:


def sinaCheck(df,maxIndex):
    duration = 9
    mean = statistics.mean(df['Close'][maxIndex-1-duration :maxIndex - 1])
    price = df['Close'][maxIndex-1]
    if price > mean:
        return -1
    elif price < mean:
        return 1
    else:
        return 0


# In[13]:


def checkBitcoinSituation(df,maxIndex):
    duration = 1
    mean = statistics.mean(df['Close'][maxIndex-1-duration :maxIndex - 1])
    price = df['Close'][maxIndex-1]
    if price > mean:
        return 1
    elif price < mean:
        return -1
    else:
        return 0


# In[14]:


def OBV(df,maxIndex):
    maxIndex -= 1
    mean = statistics.mean(df['volume_obv'][maxIndex-20:maxIndex])
    if df['volume_obv'][maxIndex] > (mean*1.05):
        return(1)
    elif df['volume_obv'][maxIndex] < (mean*1.05):
        return(-1)
    else:
        return(0)


# In[20]:


def CMF(df,maxIndex):
    maxIndex = maxIndex - 1
    n = 5
    res = 0
    if df['volume_cmf'][maxIndex] > 0.05:
        res+=0.5
    elif df['volume_cmf'][maxIndex] < -0.05:
        res-=0.5
    else:
        pass

    for i in range(n):
        if df['volume_cmf'][maxIndex-n+i-1] == 0.05 and df['volume_cmf'][maxIndex-n+i-2] > 0.05:
            res-=0.5
        elif df['volume_cmf'][maxIndex-n+i-1] == 0.05 and df['volume_cmf'][maxIndex-n+i-2] < 0.05:
            res+=1
        elif df['volume_cmf'][maxIndex-n+i-1] == -0.05 and df['volume_cmf'][maxIndex-n+i-2] > -0.05:
            res-=1
        elif df['volume_cmf'][maxIndex-n+i-1] == -0.05 and df['volume_cmf'][maxIndex-n+i-2] < -0.05:
            res+=0.5
    return res


# In[16]:


def FI(df,maxIndex):
    res = 0
    df['FI_39'] = ta.volume.force_index(df['Close'], df['Volume'], 39,False)
    df['FI_100'] = ta.volume.force_index(df['Close'], df['Volume'], 100,False)
    n = 30
    max1 = 0
    min1 = 9999999999
    for i in range(n):
        if df['FI_100'][maxIndex-n-1+i] > max1:
            max1 = df['FI_100'][maxIndex-n-1+i]
    for i in range(n):
        if df['FI_100'][maxIndex-n-1+i] < min1:
            min1 = df['FI_100'][maxIndex-n-1+i]
    if df['FI_100'][maxIndex-1] > max1:
        res +=1
    elif df['FI_100'][maxIndex-1] == max1 and df['FI_100'][maxIndex-2] < max1:
        res +=0.5
    if df['FI_100'][maxIndex-1] < min1:
        res -= 1
    elif df['FI_100'][maxIndex-1] == min1 and df['FI_100'][maxIndex-2] > min1:
        res -= 0.5
    mean13 = statistics.mean(df['FI_39'][maxIndex-21:maxIndex-1])
    if df['FI_39'][maxIndex-1] > mean13:
        res +=1
    elif df['FI_39'][maxIndex-1] < mean13:
        res -=1
    return res


# In[17]:


all_tickers = client.get_all_tickers()
all_tickers = pd.DataFrame(all_tickers)
all_symbols = all_tickers['symbol']
usdt_symbols = []
for el in all_symbols:
    if el[-4:] == 'USDT':
        usdt_symbols.append(el)


# In[18]:


len(usdt_symbols)


# In[22]:


# new_sampels = ['EOS','LINK','KAVA','LTC','ETC','REP','TRX','XLM','XTZ','XRP','BTC','ADA','XMR','ALGO','ETH','BNB','BCH','IOTA','ATOM','VET','NEO','DASH']


# In[23]:


tempList = []

# df2 = getMetaData('BTCUSDT')
for el in usdt_symbols:
    print(el)
    i = 499
    df = getMetaData(el)
    df = dropna(df)
    try:
        df = add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume")
        df['EMA20'] = df['Close'].ewm(span=20,min_periods=0,adjust=True,ignore_na=False).mean()
        df['EMA50'] = df['Close'].ewm(span=50,min_periods=0,adjust=True,ignore_na=False).mean()
        df['EMA200'] = df['Close'].ewm(span=200,min_periods=0,adjust=True,ignore_na=False).mean()
    except:
        print('No')
        continue
    while i > 0:
        try:
            price = df['Close'][df.shape[0]-i-1]
            ########################################################
            resFi = FI(df,df.shape[0]-i)
            ########################################################
            resSina = sinaCheck(df,df.shape[0]-i)
            ########################################################
            resCmf = CMF(df,df.shape[0]-i)
            ########################################################
            resObv = OBV(df,df.shape[0]-i)
            ########################################################
            resSto = STO(df,df.shape[0]-i)
            if resSto == 'STO Says sell it.':
                resSto = -1
            elif resSto == 'STO Says buy it.':
                resSto = 1
            elif resSto == 'STO Says oversold has been occur, so it is time to buy.':
                resSto = 1
            elif resSto == 'STO Says overbought has been occur, so it is time to sell.':
                resSto = -1
            elif resSto == 'STO Exception':
                i-=1
                continue
            else:
                resSto = 0
            ########################################################
            resMacd = MACD(df,df.shape[0]-i)
            if resMacd == 'MACD Says be carefull because this chart shows that maybe it will be fine in future!':
                resMacd = 1
            elif resMacd == 'MACD Says be carefull because this chart shows that maybe it will be worse in future!':
                resMacd = -1
            elif resMacd == 'MACD Says sell.':
                resMacd = -1
            elif resMacd == 'MACD Says buy.':
                resMacd = 1
            elif resMacd == 'Exception : MACD Buy or Sell':
                i-=1
                continue
            else:
                resMacd = 0
            ######################################################## 
            resIch = ichimoku(df,df.shape[0]-i)
            if resIch == 'Ichimoku says sell it.':
                resIch = -1
            elif resIch == 'Ichimoku says buy it.':
                resIch = 1
            elif resIch == 'Ichimoku says we have support.':
                resIch = 0.5
            elif resIch == 'Ichimoku says we have resist.':
                resIch = -0.5
            elif resIch == 'ex':
                i-=1
                continue
            else:
                resIch = 0
            ########################################################
            resRsi = RSI(df,df.shape[0]-i)
            if resRsi == 'RSI Says oversold has been occur, so it is time to buy.':
                resRsi = 1
            elif resRsi == 'RSI Says overbought has been occur, so it is time to sell.':
                resRsi = -1
            elif resRsi == 'RSI Exception':
                i-=1
                continue
            else:
                resRsi = 0
            ########################################################
            resBb = BB(df,df.shape[0]-i)
            
            if resBb == 'BB Says buy it.':
                resBb = 1
            elif resBb == 'BB Says sell it.':
                resBb = -1
            elif resBb == 'BB Says we will have pull back from up.':
                resBb = 0.5
            elif resBb == 'BB Says we will have pull back from down.':
                resBb = 0
            elif resBb == 'exception BB':
                i-=1
                continue
            else:
                resBb = 0
            ########################################################
            resEMA = movingAverage20_50_200(df,df.shape[0]-i)
            if resEMA == 'EMA20_50_200 Says buy it.':
                resEMA = 1
            elif resEMA == 'EMA20_50_200 Says sell it.':
                resEMA = -1
            elif resEMA == 'EMA:Exception':
                i-=1
                continue
            else:
                resEMA = 0
            ########################################################
            resBTC = checkBitcoinSituation(df2,df.shape[0]-i)
            ########################################################
            nextPrice = max(df['High'][df.shape[0]-i:df.shape[0]-i+20]) 
            nextPrice2 = min(df['Low'][df.shape[0]-i:df.shape[0]-i+20]) 
            percentage = ((nextPrice - price)/price)*100
            percentage2 = ((nextPrice2 - price)/price)*100
            if percentage >= 5:
                classs = 1
            elif percentage2 <= -5:
                classs = -1
            else:
                classs = 0
            tempList.append([price,resMacd,resIch,resRsi,resBb,resEMA,resSto,resSina,resBTC,resObv,resCmf,resFi,classs])
        except:
            pass
        i-=1


# In[ ]:





# In[24]:


tempdf = pd.DataFrame(tempList,columns=['price','MACD','Ichimoku','RSI','BB','EMA20_50_200','RTO','Sina','resBTC','resObv','resCmf','resFi','class'])
print(tempdf.shape)
# tempdf[tempdf['class'] == -1]
tempdf.head()


# In[ ]:





# In[25]:


from sklearn.model_selection import train_test_split

inputs = tempdf[['MACD','Ichimoku','RSI','BB','EMA20_50_200','RTO','Sina','resBTC','resObv','resCmf','resFi']]
target = tempdf['class']
X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size = 0.2)


# In[26]:


from sklearn.neighbors import KNeighborsClassifier
model2 = KNeighborsClassifier(n_neighbors=101)
from sklearn import metrics
model2.fit(X_train, y_train)
resultK = model2.predict(X_test)
accuracyK = metrics.accuracy_score(y_test, resultK) 
print(accuracyK)


# In[27]:


from sklearn import tree

clf = tree.DecisionTreeClassifier()
model3 = clf.fit(X_train, y_train)
resultDT = model3.predict(X_test)  # ....DT => Decision Tree
from sklearn import metrics
accuracyDT = metrics.accuracy_score(y_test, resultDT)  # ....DT => Decision Tree
print(accuracyDT)


# In[23]:


templist = []
templist2 = ['MACD','Ichimoku','RSI','BB','EMA20_50_200','RTO','Sina','resBTC','resObv','resCmf','resFi']
from sklearn import tree
from sklearn import metrics
for i in range(len(templist2)):
    for subset in itertools.combinations(templist2, i+1):
        X = (list(subset))
        inputs = tempdf[X]
        target = tempdf['class']


        X_train, X_test, y_train, y_test = train_test_split(inputs,target,test_size = 0.2)

        clf = tree.DecisionTreeClassifier()
        model3 = clf.fit(X_train, y_train)
        resultDT = model3.predict(X_test)
        accuracyDT = metrics.accuracy_score(y_test, resultDT) 

        templist.append([list(subset),accuracyDT])
def sortSecond(val): 
    return val[1] 
templist.sort(key = sortSecond,reverse=True)
templist[0]


# In[24]:


inputs = tempdf[templist[0][0]]
target = tempdf['class']
X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size = 0.2)


# In[ ]:





# In[25]:


from sklearn import tree

clf = tree.DecisionTreeClassifier()
model3 = clf.fit(X_train, y_train)
resultDT = model3.predict(X_test)  # ....DT => Decision Tree
from sklearn import metrics
accuracyDT = metrics.accuracy_score(y_test, resultDT)  # ....DT => Decision Tree
print(accuracyDT)


# In[22]:


res9 = []
for el in usdt_symbols:
    try:
        df = getMetaData(el)
        df = dropna(df)
        try:
            df = add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume")
            df['EMA20'] = df['Close'].ewm(span=20,min_periods=0,adjust=True,ignore_na=False).mean()
            df['EMA50'] = df['Close'].ewm(span=50,min_periods=0,adjust=True,ignore_na=False).mean()
            df['EMA200'] = df['Close'].ewm(span=200,min_periods=0,adjust=True,ignore_na=False).mean()
        except:
            print('No')
            continue
        ########################################################
        resSto = STO(df,df.shape[0])
        if resSto == 'STO Says sell it.':
            resSto = -1
        elif resSto == 'STO Says buy it.':
            resSto = 1
        elif resSto == 'STO Says oversold has been occur, so it is time to buy.':
            resSto = 1
        elif resSto == 'STO Says overbought has been occur, so it is time to sell.':
            resSto = -1
        elif resSto == 'STO Exception':
            continue
        else:
            resSto = 0
        ########################################################
        resMacd = MACD(df,df.shape[0])
        if resMacd == 'MACD Says be carefull because this chart shows that maybe it will be fine in future!':
            resMacd = 1
        elif resMacd == 'MACD Says be carefull because this chart shows that maybe it will be worse in future!':
            resMacd = -1
        elif resMacd == 'MACD Says sell.':
            resMacd = -1
        elif resMacd == 'MACD Says buy.':
            resMacd = 1
        elif resMacd == 'Exception : MACD Buy or Sell':
            continue
        else:
            resMacd = 0
        ########################################################
        resIch = ichimoku(df,df.shape[0])
        if resIch == 'Ichimoku says sell it.':
            resIch = -1
        elif resIch == 'Ichimoku says buy it.':
            resIch = 1
        elif resIch == 'ex':
            continue
        else:
            resIch = 0
        ########################################################
        resRsi = RSI(df,df.shape[0])
        if resRsi == 'RSI Says oversold has been occur, so it is time to buy.':
            resRsi = 1
        elif resRsi == 'RSI Says overbought has been occur, so it is time to sell.':
            resRsi = -1
        elif resRsi == 'RSI Exception':
            continue
        else:
            resRsi = 0
        ########################################################
        resBb = BB(df,df.shape[0])

        if resBb == 'BB Says buy it.':
            resBb = 1
        elif resBb == 'BB Says sell it.':
            resBb = -1
        elif resBb == 'BB Says we will have pull back from up.':
            resBb = 0.5
        elif resBb == 'BB Says we will have pull back from down.':
            resBb = 0
        elif resBb == 'exception BB':
            continue
        else:
            resBb = 0
        ########################################################
        resEMA = movingAverage20_50_200(df,df.shape[0])
        if resEMA == 'EMA20_50_200 Says buy it.':
            resEMA = 1
        elif resEMA == 'EMA20_50_200 Says sell it.':
            resEMA = -1
        elif resEMA == 'EMA:Exception':
            continue
        else:
            resEMA = 0
        ########################################################
        resFi = FI(df,df.shape[0])
        ########################################################
        ########################################################
        resCmf = CMF(df,df.shape[0])
        ########################################################
        resSina = sinaCheck(df,df.shape[0])
        resObv = OBV(df,df.shape[0])
#         resBTC = checkBitcoinSituation(df,df.shape[0])
        print(el , ' : ' , [resMacd,resIch,resRsi,resBb,resEMA,resSto,resSina,resObv,resCmf,resFi])
#         ressss = [resIch,resBb,resSto]
#         ressss = [ressss]
#         df9 = pd.DataFrame(ressss,columns=X_test.columns)
#         resssssssss = model3.predict(df9)
#         print([el,list(resssssssss),df['Close'][df.shape[0]-1]])
#         res9.append([el,list(resssssssss),df['Close'][df.shape[0]-1]])
    except:
        print(el)


# In[29]:


res9


# In[30]:


df10 = pd.DataFrame(res9)
df10.to_csv('res_8_27_2020.csv')


# In[27]:


from sklearn.model_selection import cross_val_score

scoresDT3F = cross_val_score(model3, inputs, target, cv=3)
scoresDT5F = cross_val_score(model3, inputs, target, cv=5)
print("\nscores of Decistion Tree with 3-fold validation : ")
print(scoresDT3F)
print("\nscores of Decistion Tree with 5-fold validation : ")
print(scoresDT5F)


# In[ ]:


from sklearn.linear_model import LogisticRegressionCV

modelLR = LogisticRegressionCV(n_jobs=-1, Cs=3, cv=10, refit=True, class_weight='balanced', random_state=42)
modelLR.fit(X_train, y_train.ravel())
resultLR = modelLR.predict(X_test)


from sklearn import metrics


accuracyLR = metrics.accuracy_score(y_test, resultLR)  # ....LR => Logistic Regression
print(accuracyLR)


# In[ ]:





# In[ ]:


from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
resultNB = gnb.fit(inputs, target).predict(X_test)

from sklearn import metrics

accuracyNB = metrics.accuracy_score(y_test, resultNB)  # ....NB => Navie bayes
print(accuracyNB)


# In[ ]:





# In[46]:


df = getMetaData('LTCUSDT')
df = dropna(df)
try:
    df = add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume")
    df['EMA20'] = df['Close'].ewm(span=20,min_periods=0,adjust=True,ignore_na=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50,min_periods=0,adjust=True,ignore_na=False).mean()
    df['EMA200'] = df['Close'].ewm(span=200,min_periods=0,adjust=True,ignore_na=False).mean()
except:
    print('No')


# In[47]:


df.columns


# In[19]:


import matplotlib.pyplot as plt
plt.plot(df['Date'], df['volume_cmf'])
plt.show()


# In[31]:


import matplotlib.pyplot as plt
plt.plot(df['Date'], df['Close'])
plt.show()


# In[21]:


df['FI_39'] = ta.volume.force_index(df['Close'], df['Volume'], 39,False)


# In[22]:


df['FI_39']


# In[23]:


df['volume_fi']


# In[50]:


FI(df,499)


# In[ ]:


for el in 

