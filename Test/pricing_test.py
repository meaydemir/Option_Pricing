import Utils.option_pricing_utils as opu
import numpy as np

# Black-Scholes Model Parameters
sigma = 0.1
r = 0.02
k = 50
s0 = 50
T = 1
q = 0.02
mu = 0.06  # For binomial tree function that uses the real-world probability measure
type = 'p'

print('Monte-Carlo Simulation:')
opu.get_bs_option_price_mc(s0, k, T, sigma, r, q, type, M=1000000, display_result=True)
print('\n')

print('Binomial Tree Under Risk-Neutral Probability Measure')
opu.get_bs_option_price_binomial(s0, k, T, sigma, r, q, type, N=1000, display_result=True)
print('\n')

print('Binomial Tree Under Real-World Probability Measure')
opu.get_bs_option_price_binomial_real_world_probabilities(mu, r, sigma, s0, k, T, dt=0.01, display_result=True)
print('\n')

print('Black-Scholes Analytical Solution')
opu.get_bs_price_analytical(s0, k, T, sigma, r, q, type, display_result=True)
print('\n')

print('Black-Scholes Analytical Solution - With Discrete Dividends')
opu.get_bs_price_dividends(r, sigma, s0, k, T, dividend_array = np.array([0.02, 0.03]), display_result=True)