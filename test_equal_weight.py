import numpy as np
from fix_equal_weight import fix_equal_weight_calculation

# Create a sample price array (time_steps=5, num_assets=9)
price_array = np.array([
    [100, 200, 300, 400, 500, 600, 700, 800, 900],  # Initial prices
    [110, 220, 330, 440, 550, 660, 770, 880, 990],  # 10% increase
    [105, 210, 315, 420, 525, 630, 735, 840, 945],  # 5% increase from initial
    [95, 190, 285, 380, 475, 570, 665, 760, 855],   # 5% decrease from initial
    [120, 240, 360, 480, 600, 720, 840, 960, 1080]  # 20% increase from initial
])

# Define a ticker list with 10 elements (more than the price array has)
TICKER_LIST = ["BTC", "ETH", "ADA", "BNB", "XRP", "SOL", "DOT", "DOGE", "AVAX", "UNI"]

# Test the fix_equal_weight_calculation function
print("Testing fix_equal_weight_calculation function...")
equal_weight_values = fix_equal_weight_calculation(price_array, TICKER_LIST)

# Print the results
print("\nEqual weight portfolio values:")
for i, value in enumerate(equal_weight_values):
    print(f"Day {i}: ${value:.2f}")

# Calculate the expected values manually for verification
initial_investment = 1e5
# We expect to use only the first 9 tickers
num_assets = 9
# Equal weight means we invest initial_investment / num_assets in each asset
investment_per_asset = initial_investment / num_assets
# Calculate how many units of each asset we buy
units = np.array([investment_per_asset / price_array[0, i] for i in range(num_assets)])
# Calculate the expected portfolio values
expected_values = [sum(units * price_array[i, :]) for i in range(len(price_array))]

# Compare the results
print("\nVerification:")
print("Expected values:")
for i, value in enumerate(expected_values):
    print(f"Day {i}: ${value:.2f}")

# Check if the results match
if np.allclose(equal_weight_values, expected_values):
    print("\nTest passed! The fix_equal_weight_calculation function works correctly.")
else:
    print("\nTest failed! The fix_equal_weight_calculation function does not work correctly.")
    print("Differences:")
    for i in range(len(equal_weight_values)):
        diff = equal_weight_values[i] - expected_values[i]
        print(f"Day {i}: {diff:.2f}")
