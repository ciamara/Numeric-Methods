import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

#loading data from shiba_Data
df = pd.read_csv('shiba_Data.csv')
df = df[["Date", "Price"]]

#date to datetime
df["Date"] = pd.to_datetime(df["Date"])
#price to numeric
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

#for insurance sort by date
df = df.sort_values(by="Date")

#plotting regular price history
plt.figure().set_figwidth(15)
plt.plot(df['Date'], df['Price'], label='Shiba Coin', color='pink')
plt.xlabel('Date')
plt.ylabel(f'Price [{'USD'}]')
plt.title(f'{'Shiba Coin Pricing'}')
plt.legend()
plt.savefig('ShibaCoinPriceHistory.png', bbox_inches='tight')
plt.show()


def EMA(prices, N):
    
    alpha = 2 / (N + 1)
    ema_values = [prices[0]]  #first = first price
    
    # EMA for each price
    for price in prices[1:]:
        ema_values.append(alpha * price + (1 - alpha) * ema_values[-1])
    
    return ema_values

#EMA12 and EMA26
df['EMA12'] = EMA(df['Price'], 12) #12 period ema
df['EMA26'] = EMA(df['Price'], 26) #26 period ema

#using pandas ewm
#df['EMA12'] = df['Price'].ewm(span=12, adjust=False).mean()
#df['EMA26'] = df['Price'].ewm(span=26, adjust=False).mean()

# MACD = EMA12 - EMA26
df['MACD'] = df['EMA12'] - df['EMA26']

# SIGNAL (EMA z MACD)
df['Signal'] = EMA(df['MACD'], 9) #9 period ema of macd

#using pandas ewm
#df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

#plotting whole period MACD & SIGNAL
plt.figure().set_figwidth(15)
plt.plot(df['Date'], df['MACD'], label='MACD', color='pink')
plt.plot(df['Date'], df['Signal'], label='Signal', color='purple')

# buy/sell markers
buy_markers = np.argwhere(np.diff(np.sign(df['MACD'] - df['Signal'])) > 0).flatten() + 1
sell_markers = np.argwhere(np.diff(np.sign(df['MACD'] - df['Signal'])) < 0).flatten() + 1
#rows where buy occured
buy = df.iloc[buy_markers]
#rows where sell occurs
sell = df.iloc[sell_markers]

plt.plot(buy['Date'], buy['MACD'], 'g^', label='Buy')
plt.plot(sell['Date'], sell['MACD'], 'rv', label='Sell')
plt.xlabel('Date')
plt.title('MACD & Signal')
plt.legend()
plt.savefig('ShibaCoinFullMACDSignal.png', bbox_inches='tight')
plt.show()

#best trades for showcase
best_trades = {
    "2021-07 to 2022-07": {"buy_date": "2021-9-30", "sell_date": "2021-11-04"},
    "2024-01 to 2025-01": {"buy_date": "2024-02-24", "sell_date": "2024-03-15"},
}

def plot_MACD_SIGNAL(start_date, end_date, title, best_trade):

    #selects rows in chosen period
    df_selected = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

    #selects buys within period
    buy_selected = buy[(buy['Date'] >= start_date) & (buy['Date'] <= end_date)]
    #selects sells within period
    sell_selected = sell[(sell['Date'] >= start_date) & (sell['Date'] <= end_date)]

    #selects the best buy/sell row based on best_trades
    best_buy = df[df['Date'] == best_trade["buy_date"]].iloc[0]
    best_sell = df[df['Date'] == best_trade["sell_date"]].iloc[0]

    #extracts values for best buy/sell dates
    #rounds value to 6 decimal places
    buy_value = round(best_buy['Price'], 6)
    sell_value = round(best_sell['Price'], 6)
    #calculates profit from investment
    profit = round(sell_value - buy_value, 6)
    
    #plotting MACD & SIGNAL with showcase investments
    plt.figure(figsize=(12,5))
    plt.plot(df_selected['Date'], df_selected['MACD'], label='MACD', color='pink')
    plt.plot(df_selected['Date'], df_selected['Signal'], label='Signal', color='purple')
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)

    plt.scatter(buy_selected['Date'], buy_selected['MACD'], color='green', marker='^', label='Buy', s=100)
    plt.scatter(sell_selected['Date'], sell_selected['MACD'], color='red', marker='v', label='Sell', s=100)

    plt.scatter(best_buy['Date'], best_buy['MACD'], color='yellow', marker='^', s=200, label='Best Buy', edgecolors='black', linewidth=1.5)
    plt.scatter(best_sell['Date'], best_sell['MACD'], color='yellow', marker='v', s=200, label='Best Sell', edgecolors='black', linewidth=1.5)
    plt.plot([best_buy['Date'], best_sell['Date']], [best_buy['MACD'], best_sell['MACD']], 'b--', linewidth=2, color='orange')

    plt.xlabel('Date')
    plt.title(f"{title}\nBuy: {buy_value:.6f}, Sell: {sell_value:.6f}, Profit: {profit:.6f}")
    plt.legend()
    plt.grid()
    if start_date == "2021-07-01":
        plt.savefig('ShibaCoinMACDSignalPeriod1.png', bbox_inches='tight')
    else:
        plt.savefig('ShibaCoinMACDSignalPeriod2.png', bbox_inches='tight')
    plt.show()

plot_MACD_SIGNAL("2021-07-01", "2022-07-01", "MACD & Signal (2021-07 to 2022-07) with Buy/Sell Markers & Best Trade", best_trades["2021-07 to 2022-07"])

plot_MACD_SIGNAL("2024-01-01", "2025-01-01", "MACD & Signal (2024-01 to 2025-01) with Buy/Sell Markers & Best Trade", best_trades["2024-01 to 2025-01"])


#simulation of investing
#full buy/sell on every buy/sell marker
def simulation(df, initial_capital=1000):

    capital = initial_capital
    shiba_coins = 0
    #tracks wallet value over time
    wallet = [capital]
    #stores all transactions
    transactions = [] #(buy/sell, date, price, wallet value)
    #tracks profit from each trade
    profits = []

    #all historical data
    for i in range(1, len(df)):
        
        # buy when MACD crosses Signal from below
        if df['MACD'].iloc[i] > df['Signal'].iloc[i] and df['MACD'].iloc[i-1] <= df['Signal'].iloc[i-1]:

            #all capital into shiba_coin at current price
            shiba_coins = capital / df['Price'].iloc[i]
            #removes all capital(invested)
            capital = 0
            #logs transaction
            transactions.append(('Buy', df['Date'].iloc[i], df['Price'].iloc[i], capital + shiba_coins * df['Price'].iloc[i]))

        # sell when MACD crosses Signal from above
        elif df['MACD'].iloc[i] < df['Signal'].iloc[i] and df['MACD'].iloc[i-1] >= df['Signal'].iloc[i-1] and shiba_coins > 0:
            
            #convert all shiba_coins into capital at current price
            capital = shiba_coins * df['Price'].iloc[i]
            #removes all shiba_coins (sold)
            shiba_coins = 0
            #logs profit
            profit = capital - initial_capital
            profits.append(profit)
            #logs transaction
            transactions.append(('Sell', df['Date'].iloc[i], df['Price'].iloc[i], capital + shiba_coins * df['Price'].iloc[i]))
        
        #calculate wallet value
        wallet_value = capital + shiba_coins * df['Price'].iloc[i]
        #log wallet value
        wallet.append(wallet_value)


    #plotting wallet value over time
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], wallet[:len(df)], label="Wallet Value (USD)", color='green')
    plt.xlabel('Date')
    plt.ylabel('Wallet Value (USD)')
    plt.title('Wallet Value Over Time')
    plt.grid(True)


    # extracts dates and wallet values for buy and sell transactions
    #t[0] -> buy/sell
    #t[1] -> date
    #t[2] -> price
    #t[3] -> wallet value
    buy_dates = [t[1] for t in transactions if t[0] == 'Buy']
    buy_wallet_values = [t[3] for t in transactions if t[0] == 'Buy']
    sell_dates = [t[1] for t in transactions if t[0] == 'Sell']
    sell_wallet_values = [t[3] for t in transactions if t[0] == 'Sell']

    #plotting transactions(buy and sell dates)
    plt.scatter(buy_dates, buy_wallet_values, marker='^', color='green', label='Buy', s=100)
    plt.scatter(sell_dates, sell_wallet_values, marker='v', color='red', label='Sell', s=100)

    plt.legend()
    plt.savefig('WalletValueByDate.png', bbox_inches='tight')
    plt.show()

    # prints stats of simulation
    print("\n")
    print(f"Initial Capital: {initial_capital}")
    print(f"Final Wallet Value: {capital + shiba_coins * df['Price'].iloc[-1]}")
    print(f"Total Profit/Loss: {capital + shiba_coins * df['Price'].iloc[-1] - initial_capital}")
    print(f"Number of Profitable Transactions: {sum(p > 0 for p in profits)}")
    print(f"Number of Loss Transactions: {sum(p < 0 for p in profits)}")
    print("\n")

simulation(df)