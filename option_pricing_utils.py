import numpy as np
from scipy.stats import norm
import time


def get_bs_option_price_mc(r, sigma, s0, k, T, M, display_result=True):
    
    # Calculate Black-Scholes option price via Monte Carlo

    start = time.time()
    W = np.random.normal(loc=0.0, scale=T, size=M)
    s_array = s0*np.exp((r-0.5*sigma**2)*T + sigma*W)
    option_payoff_array = np.maximum(s_array - k, 0)  # European call option
    discounted_option_payoff_array = np.exp(-r*T)*option_payoff_array
    bs_option_price = np.mean(discounted_option_payoff_array)
    end = time.time()

    if display_result == True:
        print(f'The time of execution of is: {end-start}')
        print(f'Black-Scholes option price is {bs_option_price}')
    
    return bs_option_price


def get_bs_option_price_binomial(r, sigma, s0, k, T, dt, display_result=True):

    # Calculate the price of an option using a binomial tree (parameters allow convergence to Black-Scholes solution with small step size)
    start = time.time()

    # Define parameters
    t_array = np.arange(start=0, step=dt, stop=T + dt)
    n = len(t_array) - 1
    u = np.exp(sigma*np.sqrt(T/n))
    d = 1/u

    # Risk-neutral probabilities
    p = (np.exp(r*dt)-d)/(u-d)
    q = 1 - p

    # Build boundary conditions
    S_T = np.array([s0*u**(n-i)*d**i for i in range(n+1)])
    V_T = np.maximum(S_T-k, 0)

    # Propagate backwards in tree and solve for option value at each node
    V_curr = V_T
    for i in range(n): # Loop backwards through each step in the tree
        m = len(V_curr) - 1
        V_prev = [np.exp(-r*dt)*(p*V_curr[j] + q*V_curr[j+1]) for j in range(m)] # Calculate the value at each node
        V_curr = V_prev
    end = time.time()

    if display_result == True:
        print(f'The time of execution of is: {end-start}')
        print(f'Option price using step size {dt} is {V_prev[0]}')

    return V_prev[0]


def get_bs_option_price_binomial_real_world_probabilities(mu, r, sigma, s0, k, T, dt, display_result=True):
    
    # Calculate the price of an option using a binomial tree (parameters allow convergence to Black-Scholes solution with small step size)
    # This function does not explicitly utilize the risk-neutral measure, and instead uses the actual drift of the stock 
    # to compute the option price via an adjusted discount factor

    # This is mainly a demonstration of how remaining in the "real-world" probability measure complicates the underlying simulation, although
    # they end up producing the same result (as they should)
    
    start = time.time()

    # Define parameters
    t_array = np.arange(start=0, step=dt, stop=T + dt)
    n = len(t_array) - 1
    u = np.exp(sigma*np.sqrt(T/n))
    d = 1/u

    # Real-world probabilities (assuming we can build a hedging portfolio)
    p = (np.exp(mu*dt)-d)/(u-d)
    q = 1 - p

    # Build boundary conditions
    S_T = np.array([s0*u**(n-i)*d**i for i in range(n+1)])
    V_T = np.maximum(S_T-k, 0)

    # Propagate backwards in tree and solve for option value at each node
    V_curr = V_T
    for i in range(n): # Loop backwards through each step in the tree
        V_prev = np.array([]) # Initialize
        m = len(V_curr) - 1
        for j in range(m):

            # Compute discount factor (no longer r since we are using the real-world probability measure)
            d1 = (np.exp(r*dt) - d)*V_curr[j] + (u-np.exp(r*dt))*V_curr[j+1]
            d2 = (np.exp(mu*dt)-d)*V_curr[j] + (u-np.exp(mu*dt))*V_curr[j+1]
            if d2 == 0:
                discount_factor = 0  # Both options in node have zero value, discounted value is zero regardless of value of discount factor
            else:
                discount_factor = r - 1/dt*np.log(d1/d2)
            
            # Calculate the value of the option at the node
            V_prev = np.append(V_prev, np.exp(-discount_factor*dt)*(p*V_curr[j] + q*V_curr[j+1]))
        V_curr = V_prev
    end = time.time()

    if display_result == True:
        print(f'The time of execution of is: {end-start}')
        print(f'Option price using step size {dt} is {V_prev[0]}')
    
    return V_prev[0]

def get_bs_price_analytical(r, sigma, s0, k, T, display_result=True):

    # Compute analytical solution of Black-Scholes model for a European call option

    start = time.time()

    # Black-Scholes Model Parameters
    d1 = 1/(sigma*np.sqrt(T))*(np.log(s0/k) + (r + 1/2*sigma**2)*T)
    d2 = d1 - sigma*np.sqrt(T)
    option_price = s0*norm.cdf(d1) - np.exp(-r*T)*k*norm.cdf(d2)
    end = time.time()

    if display_result == True:
        print(f'The time of execution of is: {end-start}')
        print(f'Option price is {option_price}')
    return option_price


def get_bs_price_dividends(r, sigma, s0, k, T, dividend_array, display_result=True):

    # Compute analytical solution of Black-Scholes model for a European call option on a stock paying lump-sum dividends

    start = time.time()

    # Black-Scholes Model Parameters
    s0_adj = s0*np.cumprod(1-dividend_array)[-1]

    d1 = 1/(sigma*np.sqrt(T))*(np.log(s0_adj/k) + (r + 1/2*sigma**2)*T)
    d2 = d1 - sigma*np.sqrt(T)
    option_price = s0*norm.cdf(d1) - np.exp(-r*T)*k*norm.cdf(d2)

    end = time.time()

    if display_result == True:
        print(f'The time of execution of is: {end-start}')
        print(f'Option price is {option_price}')
    return option_price