# alpaca_trading_agent/src/backtesting/backtest_agent.py

import pandas as pd
import numpy as np
import os
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import pyfolio
from finrl.plot import backtest_stats, backtest_plot, get_daily_return
import matplotlib.pyplot as plt # Import matplotlib

# --- Configuration and Environment Loading ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_DIR = os.path.join(PROJECT_ROOT, 'config')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')

# Add relevant directories to sys.path
import sys
sys.path.insert(0, CONFIG_DIR)
sys.path.insert(0, os.path.join(SRC_DIR, 'environment')) # To import trading_env

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    import settings
    from trading_env import create_env # Import our environment creation function
except ImportError as e:
    logging.error(f"Error importing configuration or environment: {e}.")
    logging.error("Ensure config/settings.py and src/environment/trading_env.py exist.")
    sys.exit(1)

# Create results/backtesting directory if it doesn't exist
BACKTEST_RESULTS_DIR = os.path.join(RESULTS_DIR, 'backtesting')
os.makedirs(BACKTEST_RESULTS_DIR, exist_ok=True)

# --- Backtesting Parameters ---
# Use the same model details as training for consistency
MODEL_ALGO = "PPO"
TOTAL_TIMESTEPS = 20000 # From training run
MODEL_FILENAME_BASE = f"{MODEL_ALGO}_{settings.TIME_INTERVAL}_{TOTAL_TIMESTEPS}"
MODEL_PATH = os.path.join(RESULTS_DIR, 'models', f"{MODEL_FILENAME_BASE}.zip")

# --- Load Processed Test Data ---
processed_test_filename = f"test_processed_{settings.TEST_START_DATE}_{settings.TEST_END_DATE}.csv"
processed_test_filepath = os.path.join(DATA_DIR, processed_test_filename)

try:
    test_df = pd.read_csv(processed_test_filepath)
    test_df['date'] = pd.to_datetime(test_df['date'])
    # Sort and reset index for consistency with how env expects data
    test_df = test_df.sort_values(by=['date', 'tic']).reset_index(drop=True)
    logging.info(f"Loaded processed testing data: {processed_test_filepath}")
except FileNotFoundError:
    logging.error(f"Processed testing data file not found: {processed_test_filepath}")
    logging.error("Please run the preprocessing script first.")
    sys.exit(1)
except Exception as e:
    logging.error(f"Error loading processed testing data: {e}")
    sys.exit(1)

# --- Create Test Environment ---
logging.info("Creating testing environment...")
# Important: Use the test_df here
try:
    # Create a single instance for backtesting prediction loop
    env_test_instance = create_env(test_df)
    logging.info("Testing environment instance created successfully.")
except Exception as e:
    logging.error(f"Failed to create testing environment: {e}", exc_info=True)
    sys.exit(1)

# --- Load Trained Model ---
logging.info(f"Loading trained model from: {MODEL_PATH}")
try:
    model = PPO.load(MODEL_PATH, env=None) # Load structure, weights will be loaded
    logging.info("Model loaded successfully.")
except FileNotFoundError:
    logging.error(f"Trained model file not found: {MODEL_PATH}")
    logging.error("Please run the training script first.")
    sys.exit(1)
except Exception as e:
    logging.error(f"Error loading trained model: {e}", exc_info=True)
    sys.exit(1)

# --- Backtesting Loop ---
logging.info("Starting backtesting loop...")
account_value_list = []
rewards_list = []
actions_list = []

obs, info = env_test_instance.reset()
account_value_list.append(env_test_instance.initial_amount) # Start with initial amount

terminated = False
truncated = False # Gymnasium uses truncated

while not (terminated or truncated):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env_test_instance.step(action)

    # Store results from the step
    # info might contain portfolio value, but let's use the env's asset_memory if available
    # Note: Our replicated env stores asset value in self.asset_memory
    if hasattr(env_test_instance, 'asset_memory') and env_test_instance.asset_memory:
         account_value_list.append(env_test_instance.asset_memory[-1])
    else:
         # Fallback or alternative calculation if needed
         current_portfolio_value = env_test_instance.state[0] + sum(
             np.array(env_test_instance.state[1 : (env_test_instance.stock_dim + 1)])
             * env_test_instance.data["close"].values
         )
         account_value_list.append(current_portfolio_value)

    rewards_list.append(reward)
    actions_list.append(action) # Store actions if needed for analysis

    if terminated or truncated:
        logging.info("Backtesting loop finished.")
        # Final values are often in the 'info' dict upon termination
        if info:
             logging.info(f"Final Info: {info}")
        break

# --- Add Plotting Function ---
def plot_account_value(df, save_path):
    """Plots account value over time and saves the figure."""
    try:
        plt.figure(figsize=(15, 6))
        plt.plot(df.index, df['account_value'])
        plt.title('Backtest Account Value Over Time')
        plt.xlabel('Date')
        plt.ylabel('Account Value')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close() # Close the figure to free memory
        logging.info(f"Account value plot saved to: {save_path}")
    except Exception as e:
        logging.error(f"Error generating account value plot: {e}", exc_info=True)

# --- Performance Analysis ---
logging.info("Calculating performance metrics...")

# Create DataFrame from results
account_value_df = pd.DataFrame({'account_value': account_value_list})
# Get unique dates from the test period actually used by the environment
# The environment starts at the first valid day, which might not be test_df['date'].min()
start_day_index = env_test_instance.day - len(account_value_list) + 1 # Infer start day index used
test_dates = env_test_instance.date_index[start_day_index : start_day_index + len(account_value_list)]
account_value_df['date'] = pd.to_datetime(test_dates)
account_value_df = account_value_df.set_index('date')

logging.info(f"Account Value DF shape: {account_value_df.shape}")
logging.info(account_value_df.head())
logging.info(account_value_df.tail())

# Calculate daily returns using FinRL utility
# Ensure the index is DatetimeIndex
if not isinstance(account_value_df.index, pd.DatetimeIndex):
     account_value_df.index = pd.to_datetime(account_value_df.index)

# Reset index so 'date' becomes a column, as expected by get_daily_return
account_value_df_for_finrl = account_value_df.reset_index()

# get_daily_return returns a Series with DatetimeIndex
daily_returns_series = get_daily_return(account_value_df_for_finrl)
daily_returns_series.name = "Daily Return" # Rename for Pyfolio

# --- Generate Pyfolio Report ---
logging.info("Generating Pyfolio tearsheet...")
# Note: Pyfolio might require specific data frequency or additional inputs (e.g., benchmark)
# For simplicity, we generate a basic tearsheet first.
try:
    # Ensure index is timezone-naive or UTC for Pyfolio
    if daily_returns_series.index.tz is not None:
        daily_returns_series.index = daily_returns_series.index.tz_localize(None)

    # Define output path for the tearsheet
    tearsheet_path = os.path.join(BACKTEST_RESULTS_DIR, f"{MODEL_FILENAME_BASE}_pyfolio_tearsheet.html")

    # Use pyfolio.create_full_tear_sheet and save to file
    # create_full_tear_sheet doesn't directly save to HTML, we need a workaround
    # Option 1: Use plotting functions and combine manually (complex)
    # Option 2: Use a wrapper or capture output (might be fragile)
    # Option 3: Use FinRL's plotting utilities which wrap pyfolio somewhat

    # Using FinRL's backtest_plot
    # plot_path = os.path.join(BACKTEST_RESULTS_DIR, f"{MODEL_FILENAME_BASE}_backtest_plot.png")
    # logging.info("Generating backtest plot...")
    # # Call backtest_plot using the df with 'date' as a column
    # # --- Temporarily commented out due to pyfolio/pandas/numpy version conflicts ---
    # # backtest_plot(account_value_df_for_finrl, # Use df with date column
    # #               baseline_ticker='^GSPC',
    # #               baseline_start=account_value_df_for_finrl['date'].min().strftime('%Y-%m-%d'),
    # #               baseline_end=account_value_df_for_finrl['date'].max().strftime('%Y-%m-%d'))
    # # # Save the displayed plot to a file
    # # plt.savefig(plot_path)
    # # plt.close() # Close the plot figure
    # # logging.info(f"Backtest plot saved to {plot_path}")
    logging.warning("Skipping backtest_plot due to dependency version conflicts (pyfolio/pandas/numpy).")

    # Using FinRL's backtest_stats to print stats (expects df with date column)
    stats_str = backtest_stats(account_value=account_value_df_for_finrl, value_col_name='account_value')
    print("\n--- Backtesting Performance Stats ---")
    print(stats_str)
    stats_path = os.path.join(BACKTEST_RESULTS_DIR, f"{MODEL_FILENAME_BASE}_backtest_stats.txt")
    with open(stats_path, 'w') as f:
        f.write(stats_str.to_string()) # Convert Series to string before writing
    logging.info(f"Backtest stats saved to {stats_path}")

    # Optional: Try generating full pyfolio report if possible (might need adjustments)
    # import matplotlib.pyplot as plt
    # fig = pyfolio.create_full_tear_sheet(
    #     returns=daily_returns_series, # Pass the Series directly
    #     # benchmark_rets=benchmark_rets, # Add benchmark returns if available
    #     set_context=False, # Avoid issues in non-notebook environments
    #     round_trips=False # May need transaction data for round trips
    # )
    # # Save the figure generated by pyfolio
    # fig_path = os.path.join(BACKTEST_RESULTS_DIR, f"{MODEL_FILENAME_BASE}_pyfolio_figure.png")
    # fig.savefig(fig_path)
    # plt.close(fig) # Close the plot figure
    # logging.info(f"Pyfolio figure saved to {fig_path}")


except Exception as e:
    logging.error(f"Error during Pyfolio report generation: {e}", exc_info=True)

# --- Generate and Save Matplotlib Plot ---
plot_save_path = os.path.join(BACKTEST_RESULTS_DIR, f"{MODEL_FILENAME_BASE}_account_value_plot.png")
plot_account_value(account_value_df, plot_save_path) # Call the new function

# --- Save Results ---
results_df_path = os.path.join(BACKTEST_RESULTS_DIR, f"{MODEL_FILENAME_BASE}_account_values.csv")
account_value_df.to_csv(results_df_path)
logging.info(f"Account values saved to {results_df_path}")

logging.info("--- Backtesting Script Finished ---")
