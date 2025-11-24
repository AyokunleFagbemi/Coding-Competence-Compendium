# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 05:43:08 2024

@author: David F
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

# Read Excel files into pandas dataframes
file1 = 'stock1.xlsx'
file2 = 'stock2.xlsx'
file3 = 'stock3.xlsx'
file4 = 'stock4.xlsx'
file5 = 'stock5.xlsx'
file6 = 'stock6.xlsx'
df1 = pd.read_excel(file1).dropna()
df2 = pd.read_excel(file2).dropna()
df3 = pd.read_excel(file3).dropna()
df4 = pd.read_excel(file4).dropna()
df5 = pd.read_excel(file5).dropna()
df6 = pd.read_excel(file6).dropna()

# Plotting for the first Excel file
plt.figure(figsize=(10, 6))
plt.plot(df1['Date'], df1['Closing Price'], label='Stock 1')  
plt.ylabel('Stock Price')
plt.title('Stock Prices Over Time - Stock 1')
plt.legend()
plt.grid(True)
plt.show()

# Plotting for the second Excel file
plt.figure(figsize=(10, 6))
plt.plot(df2['Date'], df2['Closing Price'], label='Stock 2') 
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Stock Prices Over Time - Stock 2')
plt.legend()
plt.grid(True)
plt.show()

# Plotting for the third Excel file
plt.figure(figsize=(10, 6))
plt.plot(df3['Date'], df3['Closing Price'], label='Stock 3')  
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Stock Prices Over Time - Stock 3')
plt.legend()
plt.grid(True)
plt.show()

# Plotting for the fourth Excel file
plt.figure(figsize=(10, 6))
plt.plot(df4['Date'], df4['Closing Price'], label='Stock 4')  
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Stock Prices Over Time - Stock 4')
plt.legend()
plt.grid(True)
plt.show()

# Plotting for the fifth Excel file
plt.figure(figsize=(10, 6))
plt.plot(df5['Date'], df5['Closing Price'], label='Stock 1')  
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Stock Prices Over Time - Stock 5')
plt.legend()
plt.grid(True)
plt.show()

# Plotting for the second Excel file
plt.figure(figsize=(10, 6))
plt.plot(df6['Date'], df6['Closing Price'], label='Stock 6')  
plt.ylabel('Stock Price')
plt.title('Stock Prices Over Time - Stock 6')
plt.legend()
plt.grid(True)
plt.show()

# Subtract the mean from the closing prices
df1['Closing Price'] -= df1['Closing Price'].mean()
df2['Closing Price'] -= df2['Closing Price'].mean()
df3['Closing Price'] -= df3['Closing Price'].mean()
df4['Closing Price'] -= df4['Closing Price'].mean()
df5['Closing Price'] -= df5['Closing Price'].mean()
df6['Closing Price'] -= df6['Closing Price'].mean()


# Ensure the date ranges are aligned
common_dates_1_2 = df1['Date'].isin(df2['Date'])
df1 = df1[common_dates_1_2]
df2 = df2[common_dates_1_2]

common_dates_3_4 = df3['Date'].isin(df4['Date'])
df3 = df3[common_dates_3_4]
df4 = df4[common_dates_3_4]

common_dates_5_6 = df5['Date'].isin(df6['Date'])
df5 = df5[common_dates_5_6]
df6 = df6[common_dates_5_6]

# Autocorrelation Function - Stock 1
plt.figure(figsize=(10, 6))
acf = signal.correlate(df1['Closing Price'], df1['Closing Price'], mode='full')
acf /= np.max(acf)  # Normalize ACF so that the peak at lag 0 is 1
lags_acf = signal.correlation_lags(len(df1['Closing Price']), len(df1['Closing Price']), mode='full')
plt.plot(lags_acf, acf)
plt.xlabel('Lag')
plt.ylabel('Autocorrelation Value')
plt.title('Autocorrelation Function - Stock 1')
plt.grid(True)
plt.savefig("Figure 7")
plt.show()

# Function to calculate and plot cross-correlation
def plot_ccf(df_a, df_b, label_a, label_b, fig_num):
    plt.figure(figsize=(10, 6))
    ccf = signal.correlate(df_a['Closing Price'], df_b['Closing Price'], mode='full')
    normalization_factor = np.sqrt(np.sum(df_a['Closing Price'] ** 2) * np.sum(df_b['Closing Price'] ** 2))
    ccf /= normalization_factor
    lags_ccf = signal.correlation_lags(len(df_a['Closing Price']), len(df_b['Closing Price']), mode='full')
    plt.plot(lags_ccf, ccf)
    plt.xlabel('Lag')
    plt.ylabel('Cross-correlation Value')
    plt.title(f'Cross Correlation Function - {label_a} and {label_b}')
    plt.grid(True)
    plt.savefig(f"Figure {fig_num}")
    plt.show()
    optimal_lag = lags_ccf[np.argmax(np.abs(ccf))]
    print(f'The optimal lag between {label_a} and {label_b} is {optimal_lag}')

# Cross Correlation Function - Stock 1 and Stock 2
plot_ccf(df1, df2, 'Stock 1', 'Stock 2', 8)

# Cross Correlation Function - Stock 3 and Stock 4
plot_ccf(df3, df4, 'Stock 3', 'Stock 4', 9)

# Cross Correlation Function - Stock 5 and Stock 6
plot_ccf(df5, df6, 'Stock 5', 'Stock 6', 10)
