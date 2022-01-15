import numpy as np
import matplotlib.pyplot as plt
import option_pricing_utils as opu

# Black-Scholes Model Parameters
sigma = 0.1
r = 0.02
k = 50
s0_array = np.arange(20, 70, 0.1)
T = 5
M = 100000  # Number of simulations

# Get option values
option_value_list = [opu.get_bs_option_price_mc(r, sigma, s0, k, T, M, display_result=False) for s0 in s0_array]

plt.plot(s0_array, option_value_list)
plt.xlabel('Initial Stock Price (S0)')
plt.ylabel('Option Value')
plt.title(f'BS Model - Sigma: {sigma}, r: {r}, T: {T}, K: {k}')
plt.show()
