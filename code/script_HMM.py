# -*- coding: utf-8 -*-# -*- coding: utf-8 -*-
import yfinance as yf
from hmmlearn.hmm import GMMHMM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from  sklearn.preprocessing  import StandardScaler
# simbolo dell'S&P 500 su Yahoo Finance
ticker = "SPY"

# scarica i dati settimanali dal 2000 a oggi
spy_ = yf.download(ticker, interval="1d", start="2004-11-25", end="2025-09-12")


gld_ = yf.download('GLD', interval = '1d', start = "2004-11-25", end = '2025-09-12')


for df in [spy_,gld_]:
    df.columns = df.columns.get_level_values(0)
    df.columns= [col.capitalize() for col in df.columns]

spy = spy_['Close'].resample('W-FRI').last()
gld = gld_['Close'].resample('W-FRI').last()

spy_returns= np.log(spy.dropna() / spy.shift(1).dropna()).to_frame(name="SPY_Return")
gld_returns= np.log(gld.dropna() / gld.shift(1).dropna()).to_frame(name="GLD_Return")


combined= spy_returns.join(gld_returns, how = 'inner')

combined= combined.dropna()
print(combined)
#STRATEGY PARAMETERS
window_size= 104 #dimensione della finestra rolling 
n_components = 2 #numero di stati

strategy_returns = []
n_mix=3
# ===WALK FORWARD LOOP ===

for i in range(window_size, len(combined)-1):
    train_data = combined.iloc[i - window_size:i]
    test_index= combined.index[i]
    

    model = GMMHMM(n_components=2,n_mix = n_mix, covariance_type='tied', n_iter=100, random_state=0)
    model.fit(train_data[['SPY_Return']].values)

    posteriors = model.predict_proba(train_data[['SPY_Return']].values)
    pi_t = posteriors[-1]
    pi_next = pi_t @ model.transmat_ #è LA DISTRIBUZIONE SUGI STATI AL TEMPO T+1

    #PREDICT STATES
    states= model.predict(train_data[['SPY_Return']].values)

    train_data=train_data.copy()
    train_data['State'] = states

    states_returns = train_data.groupby('State')['SPY_Return'].mean()
    high_risk_state= states_returns.idxmin()


    #ALLOCATION
    weight_gld = pi_next[high_risk_state]
    weight_spy = 1-weight_gld

    r_spy = combined.loc[test_index, 'SPY_Return']
    r_gld = combined.loc[test_index, 'GLD_Return']
    strategy_return= (weight_spy*r_spy)+ (weight_gld*r_gld)

    strategy_returns.append((test_index,strategy_return,r_spy,r_gld, weight_gld))


strategy_df = pd.DataFrame(strategy_returns, columns= ['Date', 'Strategy_return', 'SPY_Return', 'GLD_Return', 'GLD_weight'])

strategy_df.set_index('Date', inplace = True)

strategy_df['Cumulative_Strategy'] = (1+strategy_df['Strategy_return']).cumprod()
strategy_df['Cumulative_SPY'] = (1+strategy_df['SPY_Return']).cumprod()
strategy_df['Cumulative_GLD'] = (1+strategy_df['GLD_Return']).cumprod()
strategy_df['Mixed_Return'] = 0.5 * strategy_df['SPY_Return'] + 0.5 * strategy_df['GLD_Return']
strategy_df['Cumulative_mixed_return'] = (1+strategy_df['Mixed_Return']).cumprod()

strategy_df[['Cumulative_Strategy', 'Cumulative_SPY']].plot(figsize= (12,6))
plt.grid(True)
##STATS
def compute_stats(returns, freq = 'W'):
    periods_per_year= {'D': 252, 'W': 52, 'M':12}[freq]
    cumulative_return = (1+returns).prod()
    n_period = len(returns)
    years= n_period/periods_per_year
    cagr = cumulative_return ** (1/years) -1
    
    rolling = (1+returns).cumprod()
    peak = rolling.cummax()
    drawdown = (rolling - peak ) / peak
    max_dd = drawdown.min()
    avg_dd = drawdown[drawdown<0].mean()
    
    volatility = returns.std() * np.sqrt(periods_per_year)
    sharpe = returns.mean() / returns.std() * np.sqrt(periods_per_year)
    return {'CAGR' : cagr, 'MAX_DRAWDOWN': max_dd, 'AVD_DRAWDOWN': avg_dd, 'SHARPE_RATIO': sharpe}
 

stats_strategy = compute_stats(strategy_df['Strategy_return'], freq='W')
stats_spy = compute_stats(strategy_df['SPY_Return'], freq='W')
stats_gld = compute_stats(strategy_df['GLD_Return'])
stats_mixed = compute_stats(strategy_df['Mixed_Return'])

# Combine into report
report = pd.DataFrame(
    [stats_strategy, stats_spy, stats_gld, stats_mixed],
    index=['Strategy', 'SPY', 'GLD', '50/50']
)

# Display results
print(report.round(4))

"""
Spyder Editor

This is a temporary script file.
"""

