import numpy as np

def fix_equal_weight_calculation(price_array, ticker_list):
    """
    Safely calculate equal weight portfolio values
    
    Args:
        price_array: numpy array of shape (time_steps, num_assets)
        ticker_list: list of ticker symbols
    
    Returns:
        equal_weight_values: list of portfolio values over time
    """
    print(f"Price array shape: {price_array.shape}")
    print(f"Number of tickers: {len(ticker_list)}")
    
    # Make sure we only use as many tickers as we have price data for
    num_assets = min(price_array.shape[1], len(ticker_list))
    effective_ticker_list = ticker_list[:num_assets]
    
    print(f"Using {num_assets} assets: {effective_ticker_list}")
    
    # Get initial prices
    initial_prices = price_array[0, :num_assets]
    
    # Calculate equal weight (100k initial investment)
    initial_investment = 1e5
    # Equal weight means we invest initial_investment / num_assets in each asset
    investment_per_asset = initial_investment / num_assets
    # Calculate how many units of each asset we buy
    units = np.array([investment_per_asset / initial_prices[i] for i in range(num_assets)])
    
    # Calculate portfolio value over time
    equal_weight_values = []
    for i in range(price_array.shape[0]):
        # Get prices at time i
        prices = price_array[i, :num_assets]
        # Calculate portfolio value
        portfolio_value = sum(units * prices)
        equal_weight_values.append(portfolio_value)
    
    return equal_weight_values

# Example usage:
"""
# Load price array
price_array = np.load('./price_array.npy')

# Define ticker list
TICKER_LIST = ["BTC", "ETH", "ADA", "BNB", "XRP", "SOL", "DOT", "DOGE", "AVAX", "UNI"]

# Calculate equal weight portfolio values
equal_weight_values = fix_equal_weight_calculation(price_array, TICKER_LIST)

# Plot results
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(equal_weight_values)
plt.title('Equal Weight Portfolio Value')
plt.xlabel('Time')
plt.ylabel('Portfolio Value')
plt.grid(True)
plt.show()
"""
