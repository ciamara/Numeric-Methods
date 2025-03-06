import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

DATA = 'shiba_Data.csv'
CRYPTO = 'Shiba_Coin'
CURR = 'USD'
TITLE = 'Shiba Coin Pricing'

#loading data from shiba_Data
df = pd.read_csv(DATA)
df = df[["Date", "Price"]]

#date to datetime
df["Date"] = pd.to_datetime(df["Date"])
#price to numeric
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

#for insurance sort by date
df = df.sort_values(by="Date")

plt.figure().set_figwidth(15)
plt.plot(df['Date'], df['Price'], label=CRYPTO, color='pink')
plt.xlabel('Date')
plt.ylabel(f'Price [{CURR}]')
plt.title(f'{TITLE}')
plt.legend()
plt.show()

def EMA(prices, N):
    
    alpha = 2 / (N + 1)
    ema_values = [prices[0]]  #first = first price
    
    # EMA for each price
    for price in prices[1:]:
        ema_values.append(alpha * price + (1 - alpha) * ema_values[-1])
    
    return ema_values

#EMA12 and EMA26
df['EMA12'] = EMA(df['Price'], 12)
df['EMA26'] = EMA(df['Price'], 26)


# MACD = EMA12 - EMA26
df['MACD'] = df['EMA12'] - df['EMA26']

# SIGNAL (EMA z MACD)
df['Signal'] = EMA(df['MACD'], 9)

plt.figure().set_figwidth(15)
plt.plot(df['Date'], df['MACD'], label='MACD', color='pink')
plt.plot(df['Date'], df['Signal'], label='Signal', color='purple')

# buy/sell
buy_markers = np.argwhere(np.diff(np.sign(df['MACD'] - df['Signal'])) > 0).flatten() + 1
sell_markers = np.argwhere(np.diff(np.sign(df['MACD'] - df['Signal'])) < 0).flatten() + 1
buy = df.iloc[buy_markers]
sell = df.iloc[sell_markers]

plt.plot(buy['Date'], buy['MACD'], 'g^', label='Buy')
plt.plot(sell['Date'], sell['MACD'], 'rv', label='Sell')
plt.xlabel('Date')
plt.title('MACD & Signal')
plt.legend()
plt.show()