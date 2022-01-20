import numpy as np
import time

start = time.time()

# Define parameters
T = 1
dt = 0.0001
t_array = np.arange(start=0, step=dt, stop=T + dt)
n = len(t_array) - 1
sigma = 0.1
r = 0.02
k = 50
s0 = 50
mu = 0.04  # Note that we don't need the stock drift for risk-neutral pricing!
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
print(f'The time of execution of is: {end-start}')
print(f'Option price using step size {dt} with maturity {T} is {V_prev[0]}')
 