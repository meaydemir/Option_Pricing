import numpy as np
import pandas as pd
import Utils.greeks as gr
import Utils.option_pricing_utils as opu


def simulate_gbm(S0, T_array, sigma, r, q, M):
    W = np.random.normal(loc=0.0, scale=1.0, size=M)
    dt = T_array[0] - T_array[1]
    S = np.zeros(M)
    S[0] = S0
    for i in range(M-1):
        S[i+1] = S[i] + (r-q)*S[i]*dt + sigma*S[i]*W[i]*np.sqrt(dt)
    return S

def simulate_delta_hedge(S0, K, T, sigma, r, q, type, option_exposure, M):
    T_array = np.linspace(T, 0, M)
    S = simulate_gbm(S0, T_array, sigma, r, q, M)
    dt = T_array[0] - T_array[1]

    # Calculate option Greeks
    delta_array = gr.get_delta(S, K, T_array, sigma, r, q, type)

    # Simulate hedging
    summary_df = pd.DataFrame(list(zip(S, T_array, delta_array)), columns=['Stock_Price', 't', 'Delta'])
    summary_df['Shares_Held'] = -option_exposure*delta_array
    summary_df['Shares_Purchased'] = -summary_df['Shares_Held'].sub(summary_df['Shares_Held'].shift(-1).fillna(0)).shift().fillna(-summary_df['Shares_Held'])
    summary_df['Cost_of_Shares_Purchased'] = summary_df['Shares_Purchased'].mul(summary_df['Stock_Price'])
    summary_df['Shares_Cumulative_Cost'] = summary_df['Cost_of_Shares_Purchased'].cumsum()
    summary_df['Interest_Cost'] = (summary_df['Shares_Cumulative_Cost']*(r*dt)).shift().fillna(0)
    summary_df['Cumulative_Cost_Inc_Interest'] = np.nan
    summary_df['Cumulative_Cost_Inc_Interest'].iloc[0] = summary_df['Shares_Cumulative_Cost'].iloc[0]
    for i in range(1, M):
        summary_df.loc[i, 'Cumulative_Cost_Inc_Interest'] = summary_df.loc[i-1, 'Cumulative_Cost_Inc_Interest'] + summary_df.loc[i, 'Cost_of_Shares_Purchased']+ summary_df.loc[i, 'Interest_Cost']

    hedging_cost = summary_df['Cumulative_Cost_Inc_Interest'].iloc[-1] - summary_df['Shares_Held'].iloc[-1]*K
    discounted_hedging_cost = np.exp(-r*T)*hedging_cost

    return discounted_hedging_cost
