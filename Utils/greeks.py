import numpy as np
import Utils.option_pricing_utils as opu
from scipy.stats import norm


def get_delta(S, K, T, sigma, r, q, type):
    d1, d2 = opu.get_d1_d2_values(S, K, T, sigma, r, q)
    if type.lower() == 'c':
        return norm.cdf(d1)
    elif type.lower() == 'p':
        return norm.cdf(d1) - 1
    else:
        return np.nan


def get_gamma(S, K, T, sigma, r, q, type):
    d1, d2 = opu.get_d1_d2_values(S, K, T, sigma, r, q)
    if type.lower() in ['c', 'p']:
        return norm.pdf(d1)/(S*sigma*np.sqrt(T))
    else:
        return np.nan


def get_theta(S, K, T, sigma, r, q, type):
    d1, d2 = opu.get_d1_d2_values(S, K, T, sigma, r, q)
    theta_call = -r*K*np.exp(-r*T)*norm.cdf(d2) - S*norm.pdf(d1)*sigma/(2*np.sqrt(T))
    if type.lower() == 'c':
        return theta_call
    elif type.lower() == 'p':
        return theta_call + r*K*np.exp(-r*T)
    else:
        return np.nan