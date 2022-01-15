import option_pricing_utils as opu

# Black-Scholes Model Parameters
sigma = 0.1
r = 0.02
k = 50
s0 = 50
T = 1
mu = 0.06  # For binomial tree function that uses the real-world probability measure

print('Monte-Carlo Simulation:')
opu.get_bs_option_price_mc(r, sigma, s0, k, T, M=10000000, display_result=True)
print('\n')

print('Binomial Tree Under Risk-Neutral Probability Measure')
opu.get_bs_option_price_binomial(r, sigma, s0, k, T, dt=0.001, display_result=True)
print('\n')

print('Binomial Tree Under Real-World Probability Measure')
opu.get_bs_option_price_binomial_real_world_probabilities(mu, r, sigma, s0, k, T, dt=0.001, display_result=True)
print('\n')
