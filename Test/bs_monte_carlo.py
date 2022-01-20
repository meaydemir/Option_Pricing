import numpy as np
import matplotlib.pyplot as plt
import option_pricing_utils as opu


# Black-Scholes Model Parameters
sigma = 0.1
r = 0.02
k = 50
s0 = 50
T = 1

# Generate paths based on stock price following a GBM with constant volatility and interest rate
M_array = np.arange(1, 100000, step=100)
option_value_list = [opu.get_bs_option_price_mc(r, sigma, s0, k, T, M, display_result=False) for M in M_array]

# Plot convergence curve
plt.plot(M_array, option_value_list)
plt.xlabel('Number of Simulations')
plt.ylabel('Average Option Value')
plt.title(f'BS Model - Sigma: {sigma}, r: {r}, mu: {mu}, T: {T}, K: {k}, S0: {s0}')
plt.show()

print(f'Option Value ~= {option_value_list[-1]}')