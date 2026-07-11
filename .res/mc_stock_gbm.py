import numpy as np
import matplotlib.pyplot as plt

# import matplotlib
# matplotlib.use("WebAgg")

import matplotlib
matplotlib.use("Agg")
import subprocess

initial_price = 100   # Initial stock price
mu = 0.0005           # Daily expected return (mean of log-returns)
sigma = 0.02          # Daily volatility (std of log-returns)
days = 252            # Number of trading days in a year
simulations = 1000    # Number of Monte Carlo simulations
np.random.seed(42)    

def simulate_stock_price(initial_price, mu, sigma, days, simulations):
    price_paths = np.zeros((simulations, days))
    price_paths[:, 0] = initial_price
    drift = (mu - 0.5 * sigma**2)

    for i in range(1, days):
        z = np.random.normal(loc=0.0, scale=1.0, size=simulations)
        price_paths[:, i] = price_paths[:, i - 1] * np.exp(drift + sigma * z)

    return price_paths

simulated_prices = simulate_stock_price(initial_price, mu, sigma, days, simulations)

plt.figure(figsize=(10, 6))
plt.plot(simulated_prices.T, color='blue', alpha=0.2) 
plt.title('Monte Carlo Simulation of Stock Price Movement')
plt.xlabel('Days')
plt.ylabel('Stock Price')

plt.savefig("/tmp/plot.png")
subprocess.run(["code", "/tmp/plot.png"])