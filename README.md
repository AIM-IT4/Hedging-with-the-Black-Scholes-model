# Hedging-with-the-Black-Scholes-model

The provided code is a Python implementation of a financial model to value and analyze European call options under the Black-Scholes framework. The Black-Scholes model is widely used to estimate the theoretical price of options based on certain inputs, such as the underlying asset's price, time to expiration, risk-free interest rate, and volatility.

The code consists of several functions and calculations, each serving a specific purpose:

# GeneratePathsGBM:

This function simulates multiple paths of a geometric Brownian motion (GBM) to model the behavior of the underlying asset's price over time. It uses random normal variates to generate price paths, taking into account the provided parameters for the number of paths, number of steps, time to expiration, risk-free rate, volatility, and initial asset price.

# BS_Call_Put_Option_Price:

This function calculates the option price (premium) using the Black-Scholes formula for both call and put options. It takes the current underlying asset price, strike price, volatility, time to expiration, and risk-free rate as inputs and returns the option price accordingly.

# BS_Delta:

This function calculates the delta, which represents the sensitivity of the option price to changes in the underlying asset's price. The delta is calculated using the Black-Scholes delta formula, considering the option type (call or put), the current asset price, strike price, volatility, time to expiration, and risk-free rate.

# MainCalculation:

This function serves as the main driver of the program. It sets up the initial parameters, generates price paths using the GBM model, and then calculates and analyzes the option prices, deltas, and portfolio profit and loss (P&L) over time. The code also includes plotting functionalities to visualize the asset price paths, option prices, deltas, and P&L.
