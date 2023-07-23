#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import enum

# This class defines puts and calls
class OptionType(enum.Enum):
    CALL = 1.0
    PUT = -1.0

def generate_paths_gbm(no_of_paths, no_of_steps, T, r, sigma, S_0):
    Z = np.random.normal(0.0, 1.0, [no_of_paths, no_of_steps])
    X = np.zeros([no_of_paths, no_of_steps + 1])
    W = np.zeros([no_of_paths, no_of_steps + 1])
    time = np.zeros([no_of_steps + 1])

    X[:, 0] = np.log(S_0)

    dt = T / float(no_of_steps)
    for i in range(0, no_of_steps):
        # Making sure that samples from the normal distribution have mean 0 and variance 1
        if no_of_paths > 1:
            Z[:, i] = (Z[:, i] - np.mean(Z[:, i])) / np.std(Z[:, i])
        W[:, i + 1] = W[:, i] + np.power(dt, 0.5) * Z[:, i]
        X[:, i + 1] = X[:, i] + (r - 0.5 * sigma * sigma) * dt + sigma * (W[:, i + 1] - W[:, i])
        time[i + 1] = time[i] + dt

    # Compute exponent of ABM
    S = np.exp(X)
    paths = {"time": time, "S": S}
    return paths

# Black-Scholes call option price
def bs_call_put_option_price(cp, S_0, K, sigma, t, T, r):
    K = np.array(K).reshape([len(K), 1])
    d1 = (np.log(S_0 / K) + (r + 0.5 * np.power(sigma, 2.0)) * (T - t)) / (sigma * np.sqrt(T - t))
    d2 = d1 - sigma * np.sqrt(T - t)
    
    if cp == OptionType.CALL:
        value = st.norm.cdf(d1) * S_0 - st.norm.cdf(d2) * K * np.exp(-r * (T - t))
    elif cp == OptionType.PUT:
        value = st.norm.cdf(-d2) * K * np.exp(-r * (T - t)) - st.norm.cdf(-d1) * S_0
    
    return value

def bs_delta(cp, S_0, K, sigma, t, T, r):
    # When defining a time grid it may happen that the last grid point lies slightly a bit behind the maturity time
    if t - T > 10e-20 and T - t < 10e-7:
        t = T
    K = np.array(K).reshape([len(K), 1])
    d1 = (np.log(S_0 / K) + (r + 0.5 * np.power(sigma, 2.0)) * (T - t)) / (sigma * np.sqrt(T - t))
    
    if cp == OptionType.CALL:
        value = st.norm.cdf(d1)
    elif cp == OptionType.PUT:
        value = st.norm.cdf(d1) - 1.0
    
    return value

def main_calculation():
    no_of_paths = 1000
    no_of_steps = 200
    T = 1.0
    r = 0.1
    sigma = 0.2
    s0 = 1.0
    K = [0.95]
    cp = OptionType.CALL

    np.random.seed(1)
    paths = generate_paths_gbm(no_of_paths, no_of_steps, T, r, sigma, s0)
    time = paths["time"]
    S = paths["S"]

    # Setting up some handy lambda values
    C = lambda t, K, S0: bs_call_put_option_price(cp, S0, K, sigma, t, T, r)
    Delta = lambda t, K, S0: bs_delta(cp, S0, K, sigma, t, T, r)

    # Setting up the initial portfolio
    PnL = np.zeros([no_of_paths, no_of_steps + 1])
    delta_init = Delta(0.0, K, s0)
    PnL[:, 0] = C(0.0, K, s0) - delta_init * s0

    CallM = np.zeros([no_of_paths, no_of_steps + 1])
    CallM[:, 0] = C(0.0, K, s0)
    DeltaM = np.zeros([no_of_paths, no_of_steps + 1])
    DeltaM[:, 0] = Delta(0, K, s0)

    for i in range(1, no_of_steps + 1):
        dt = time[i] - time[i - 1]
        delta_old = Delta(time[i - 1], K, S[:, i - 1])
        delta_curr = Delta(time[i], K, S[:, i])

        PnL[:, i] = PnL[:, i - 1] * np.exp(r * dt) - (delta_curr - delta_old) * S[:, i]
        CallM[:, i] = C(time[i], K, S[:, i])
        DeltaM[:, i] = delta_curr

    # Final transaction, payment of the option (if in the money) and selling the hedge
    PnL[:, -1] = PnL[:, -1] - np.maximum(S[:, -1] - K, 0) + DeltaM[:, -1] * S[:, -1]

    # We plot only one path at a time
    path_id = 13
    plt.figure(1)
    plt.plot(time, S[path_id, :])
    plt.plot(time, CallM[path_id, :])
    plt.plot(time, DeltaM[path_id, :])
    plt.plot(time, PnL[path_id, :])
    plt.legend(['Stock', 'CallPrice', 'Delta', 'PnL'])
    plt.grid()

    # Plot the histogram of the PnL
    plt.figure(2)
    plt.hist(PnL[:, -1], 50)
    plt.grid()
    plt.xlim([-0.1, 0.1])
    plt.title('Histogram of P&L')

    # Analysis for each path
    for i in range(0, no_of_paths):
        print('path_id = {0:2d}, PnL(t_0)={1:0.4f}, PnL(Tm-1)={2:0.4f}, S(t_m)={3:0.4f}, max(S(tm)-K,0)={4:0.4f}, PnL(t_m)={5:0.4f}'
              .format(i, PnL[0, 0], PnL[i, -2], S[i, -1], np.max(S[i, -1] - K, 0), PnL[i, -1]))

main_calculation()

