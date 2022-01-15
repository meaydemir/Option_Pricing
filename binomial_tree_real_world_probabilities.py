import numpy as np
import time

start = time.time()

# Define parameters
T = 1
dt = 0.001
t_array = np.arange(start=0, step=dt, stop=T + dt)
n = len(t_array) - 1
sigma = 0.1
r = 0.02
k = 50
s0 = 50
mu = 0.04  # Under no-arbitrage, mu = r
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
        d1 = (np.exp(r*dt) - d)*V_curr[j] + (u-np.exp(r*dt))*V_curr[j+1]
        d2 = (np.exp(mu*dt)-d)*V_curr[j] + (u-np.exp(mu*dt))*V_curr[j+1]
        if d2 == 0:
            discount_factor = 0  # Both options in node have zero value, discounted value is zero regardless of value of discount factor
        else:
            discount_factor = r - 1/dt*np.log(d1/d2)
        V_prev = np.append(V_prev, np.exp(-discount_factor*dt)*(p*V_curr[j] + q*V_curr[j+1]))
    V_curr = V_prev
end = time.time()
print(f'The time of execution of is: {end-start}')
print(f'Option price using step size {dt} with maturity {T} is {V_prev[0]}')
 