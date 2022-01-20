import numpy as np
from scipy.stats import norm

# Black-Scholes Model Parameters
sigma = 0.1
r = 0.02
k = 50
s0 = 50
T = 1

# Compute analytical solution of Black-Scholes model for a European call option
d1 = 1/(sigma*np.sqrt(T))*(np.log(s0/k) + (r + 1/2*sigma**2)*T)
d2 = d1 - sigma*np.sqrt(T)
option_price = s0*norm.cdf(d1) - np.exp(-r*T)*k*norm.cdf(d2)
print(option_price)
