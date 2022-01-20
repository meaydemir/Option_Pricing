import numpy as np
from scipy.stats import norm

# Black-Scholes Model Parameters
sigma = 0.1
r = 0.02
k = 50
s0 = 50   # Initial stock price
T = 1
div_array = np.array([0.02, 0.03])  # List of upcoming dividends before maturity T
s0_adj = s0*np.cumprod(1-div_array)[-1]

# Compute analytical solution of Black-Scholes model for a European call option on a stock paying lump-sum dividends
d1 = 1/(sigma*np.sqrt(T))*(np.log(s0_adj/k) + (r + 1/2*sigma**2)*T)
d2 = d1 - sigma*np.sqrt(T)
option_price = s0*norm.cdf(d1) - np.exp(-r*T)*k*norm.cdf(d2)
print(option_price)
