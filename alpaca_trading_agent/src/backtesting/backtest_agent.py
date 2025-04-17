# alpaca_trading_agent/src/backtesting/backtest_agent.py
import os
import sys

# --- Add project root to sys.path BEFORE other imports ---
# This ensures that 'alpaca_trading_agent' can be found
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, os.path.dirname(PROJECT_ROOT)) # Add parent directory (/app/workspaces/FinRL-Tutorials)
sys.path.insert(0, PROJECT_ROOT) # Add the project root itself (/app/workspaces/FinRL-Tutorials/alpaca_trading_agent)

# --- Now proceed with other imports ---
import pandas as pd
import numpy as np
import argparse  # Added for command-line arguments
import logging
import random # Added for seeding
import torch # Added for seeding
from stable_baselines3 import PPO
import pyfolio
from alpaca_trading_agent.config import settings as config # This should work now
from finrl.plot import backtest_stats, backtest_plot, get_daily_return
import matplotlib.pyplot as plt

# --- Configuration and Path Setup (Simplified as PROJECT_ROOT is defined above) ---
# PROJECT_ROOT is already defined
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

# Note: We added PROJECT_ROOT to sys.path earlier, which covers submodules in 'src' etc.
# Explicitly adding CONFIG_DIR, SRC_DIR etc. might be redundant now but can be kept for clarity
# or removed if preferred. Let's keep them for now.
sys.path.insert(0, CONFIG_DIR) # Keep for direct 'import settings' later if needed
sys.path.insert(0, os.path.join(SRC_DIR, "environment")) # Keep for 'from trading_env import ...'
sys.path.insert(0, SRC_DIR) # Keep for 'from utils.logging_setup import ...'

# --- Argument Parsing ---
parser = argparse.ArgumentParser(
    description="Run backtesting for the Alpaca Trading Agent."
)
parser.add_argument(
    "--log-level",
    type=str,
    default="INFO",
    help="Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)",
)
# args = parser.parse_args() # This is redefined below, removing duplicate

# Logging setup will be done explicitly using the shared utility
# Moved the import here as 'utils' should be findable via SRC_DIR in sys.path
from utils.logging_setup import configure_file_logging

# --- Load Configuration Settings ---
# The 'from alpaca_trading_agent.config import settings as config' at the top should handle this.
# We rely on that import now.


parser = argparse.ArgumentParser(description="Alpaca Trading Agent Backtester")
parser.add_argument(
    "--log-level",
    default="INFO",
    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    help="Set the logging level for the backtesting script.",
)
args = parser.parse_args()
# --- Configure Logging for this script ---
configure_file_logging(args.log_level) # Ensure file logging is configured
# Note: Console logging is not added here by default.

# --- Seed for Reproducibility ---
SEED = 42  # Or load from settings.py if preferred
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # Optional: Enforce determinism (might impact performance)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
logging.info(f"Global random seeds set to: {SEED}")
# --- End Seeding ---

# --- Load Settings and Environment ---
try:
    import settings
    from trading_env import create_env  # Import our environment creation function
except ImportError as e:
    logging.error(f"Error importing configuration or environment: {e}.")
    logging.error("Ensure config/settings.py and src/environment/trading_env.py exist.")
    sys.exit(1)

# Create results/backtesting directory if it doesn't exist
BACKTEST_RESULTS_DIR = os.path.join(RESULTS_DIR, "backtesting")
os.makedirs(BACKTEST_RESULTS_DIR, exist_ok=True)

# --- Backtesting Parameters ---
# Use the same model details as training for consistency
MODEL_ALGO = "PPO"
#TOTAL_TIMESTEPS = config.TOTAL_TIMESTEPS  # From training run

# Check for Walk-Forward model suffix
model_suffix = os.environ.get("WF_MODEL_SUFFIX", "")
if model_suffix:
    logging.info(f"Using Walk-Forward model suffix: {model_suffix}")

MODEL_FILENAME_BASE = (
    f"{MODEL_ALGO}_{settings.TIME_INTERVAL}_{settings.TOTAL_TIMESTEPS}{model_suffix}"
)
MODEL_PATH = os.path.join(
    RESULTS_DIR, "models", f"{MODEL_FILENAME_BASE}.zip"
)  # Use suffixed base

# --- Load Processed Test Data (and filter for Walk-Forward) ---
full_processed_filename = "processed_data.csv" # Use the standard processed data filename
full_processed_filepath = os.path.join(DATA_DIR, full_processed_filename)

# Check for Walk-Forward override environment variables
test_start_override = os.environ.get("WF_OVERRIDE_TEST_START")
test_end_override = os.environ.get("WF_OVERRIDE_TEST_END")

try:
    full_df = pd.read_csv(full_processed_filepath)
    full_df["date"] = pd.to_datetime(full_df["date"])
    full_df = full_df.sort_values(by=["date", "tic"]).reset_index(drop=True)
    logging.info(f"Loaded full processed data: {full_processed_filepath}")

    # Filter data for the current testing window
    if test_start_override and test_end_override:
        logging.info(
            f"Applying Walk-Forward date overrides: Test Start={test_start_override}, Test End={test_end_override}"
        )
        test_start_dt = pd.to_datetime(test_start_override)
        test_end_dt = pd.to_datetime(test_end_override)
        test_df = full_df[
            (full_df["date"] >= test_start_dt) & (full_df["date"] <= test_end_dt)
        ].reset_index(drop=True)
    else:
        # Default behavior: Use TEST_START_DATE and TEST_END_DATE from settings
        logging.info(
            "No Walk-Forward overrides found. Using default TEST dates from settings."
        )
        test_start_dt = pd.to_datetime(settings.TEST_START_DATE)
        test_end_dt = pd.to_datetime(settings.TEST_END_DATE)
        test_df = full_df[
            (full_df["date"] >= test_start_dt) & (full_df["date"] <= test_end_dt)
        ].reset_index(drop=True)

    if test_df.empty:
        logging.error(
            f"Testing data is empty after filtering for dates {test_start_dt} to {test_end_dt}. Check date ranges."
        )
        sys.exit(1)

    logging.info(
        f"Using testing data for period: {test_df['date'].min()} to {test_df['date'].max()}. Shape: {test_df.shape}"
    )

except FileNotFoundError:
    logging.error(f"Full processed data file not found: {full_processed_filepath}")
    logging.error("Please run the preprocessing script first.")
    sys.exit(1)
except Exception as e:
    logging.error(
        f"Error loading or filtering processed testing data: {e}"
    )  # Updated error message
    sys.exit(1)

# --- Create Test Environment ---
logging.info("Creating testing environment...")
try:
    # Pass the filtered test_df to the environment creator
    env_test_instance = create_env(
        test_df.copy()
    )  # Pass a copy to avoid modifying original df
    logging.info("Testing environment created successfully.")
except Exception as e:
    logging.error(f"Failed to create testing environment: {e}", exc_info=True)
    sys.exit(1)

# --- Load Trained Model (using potentially suffixed name) ---
logging.info(
    f"Loading trained model from: {MODEL_PATH}"
)  # Path already includes suffix
try:
    # We don't need to pass env here if we're only predicting
    model = PPO.load(MODEL_PATH, env=None)
    logging.info("Model loaded successfully.")
    # Get the list of tickers from the initial data in the env
    # Ensure env_test_instance.data is populated after reset() or init
    tickers = env_test_instance.data["tic"].unique().tolist()
    logging.info(f"Tickers being backtested: {tickers}")
    if len(tickers) != settings.STOCK_DIM:
        logging.warning(
            f"Mismatch between loaded tickers ({len(tickers)}) and settings.STOCK_DIM ({settings.STOCK_DIM})."
        )

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
actions_list = []  # Store actions
state_memory = []  # Store state before action for debugging

obs, info = env_test_instance.reset()
initial_day_index = env_test_instance.day  # Capture the starting day index AFTER reset
initial_value = env_test_instance.initial_amount
account_value_list.append(initial_value)  # Start with initial amount
logging.debug(f"--- Backtest Start ---")
logging.debug(f"Initial Observation Shape: {obs.shape}")
# Avoid logging potentially huge observation arrays unless necessary
# logging.debug(f"Initial Observation: {obs}")
logging.debug(f"Initial Info: {info}")
logging.debug(f"Starting Account Value: {initial_value:.2f}")

terminated = False
truncated = False  # Gymnasium uses truncated
step_count = 0

while not (terminated or truncated):
    step_count += 1
    current_day_index = env_test_instance.day  # Day index within the test data
    current_date = env_test_instance.date_index[current_day_index]
    logging.debug(
        f"\n--- Step {step_count} (Date: {current_date.strftime('%Y-%m-%d')}) ---"
    )
    logging.debug(f"Current Observation Shape: {obs.shape}")
    # Log state components if needed (careful with size)
    cash = obs[0]
    holdings = obs[1 : 1 + env_test_instance.stock_dim]
    # tech_indicators = obs[1+env_test_instance.stock_dim:]
    logging.debug(f"  State - Cash: {cash:.2f}")
    logging.debug(f"  State - Holdings: {holdings}")
    # logging.debug(f"  State - Tech Indicators Shape: {tech_indicators.shape}")

    action, _states = model.predict(obs, deterministic=True)
    logging.debug(
        f"Raw Action Vector: {action}"
    )  # Log the raw action output by the model

    # --- Log Buy/Sell/Hold Decisions ---
    decision_threshold = 0.0  # Threshold for buy/sell (on raw action [-1, 1])
    scaled_action = action * env_test_instance.hmax  # Action scaled to shares
    logging.debug(f"Scaled Action (Target Shares): {scaled_action.astype(int)}")
    for i, ticker in enumerate(tickers):
        raw_act = action[i]
        scaled_act = scaled_action[i]
        decision = "HOLD"
        if raw_act > decision_threshold:
            decision = f"BUY ({scaled_act:.0f} shares target)"
        elif raw_act < -decision_threshold:
            decision = f"SELL ({abs(scaled_act):.0f} shares target)"
        logging.debug(f"  Decision for {ticker}: {decision} (Raw: {raw_act:.3f})")
    # --- End Log Decisions ---

    # Store state before taking action for better debugging context
    state_memory.append(obs)

    # Take the step
    obs, reward, terminated, truncated, info = env_test_instance.step(action)

    # Log results after the step
    logging.debug(
        f"Action Taken: {action}"
    )  # Log the action actually processed by env (might be clipped/modified)
    logging.debug(f"Reward Received: {reward:.4f}")
    logging.debug(f"Terminated: {terminated}, Truncated: {truncated}")
    logging.debug(f"Info Dict: {info}")

    # Store results from the step
    # info might contain portfolio value, but let's use the env's asset_memory if available
    # Note: Our replicated env stores asset value in self.asset_memory
    if hasattr(env_test_instance, "asset_memory") and env_test_instance.asset_memory:
        account_value_list.append(env_test_instance.asset_memory[-1])
    else:
        # Fallback or alternative calculation if needed
        current_portfolio_value = env_test_instance.state[0] + sum(
            np.array(env_test_instance.state[1 : (env_test_instance.stock_dim + 1)])
            * env_test_instance.data["close"].values
        )
        account_value_list.append(current_portfolio_value)

    rewards_list.append(reward)
    actions_list.append(action)  # Store actions if needed for analysis

    if terminated or truncated:
        logging.info("Backtesting loop finished.")
        logging.debug("--- Backtest End ---")
        # Final values are often in the 'info' dict upon termination
        if info:
            logging.info(f"Final Info: {info}")
            logging.debug(f"Final Info (Debug): {info}")  # Also log final info at debug
        break


# --- Add Plotting Function ---
def plot_account_value(df, save_path):
    """Plots account value over time and saves the figure."""
    try:
        plt.figure(figsize=(15, 6))
        plt.plot(df.index, df["account_value"])
        plt.title("Backtest Account Value Over Time")
        plt.xlabel("Date")
        plt.ylabel("Account Value")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()  # Close the figure to free memory
        logging.info(f"Account value plot saved to: {save_path}")
    except Exception as e:
        logging.error(f"Error generating account value plot: {e}", exc_info=True)


def plot_stock_prices(df, tickers, save_path):
    """Plots closing prices for multiple stocks over time and saves the figure."""
    try:
        plt.figure(figsize=(15, 7))  # Adjusted size slightly
        for ticker in tickers:
            ticker_df = df[df["tic"] == ticker]
            # Ensure dates are sorted for plotting
            ticker_df = ticker_df.sort_values(by="date")
            plt.plot(ticker_df["date"], ticker_df["close"], label=ticker)

        plt.title("Stock Closing Prices During Backtest Period")
        plt.xlabel("Date")
        plt.ylabel("Closing Price")
        plt.legend(loc="upper left")  # Add legend
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()  # Close the figure to free memory
        logging.info(f"Stock prices plot saved to: {save_path}")
    except Exception as e:
        logging.error(f"Error generating stock prices plot: {e}", exc_info=True)


# --- Performance Analysis ---
logging.info("Calculating performance metrics...")

# Create DataFrame from results
account_value_df = pd.DataFrame({"account_value": account_value_list})
# Align dates with account values. account_value_list has N+1 items (initial + N steps).
num_steps = len(account_value_list) - 1
test_dates = []  # Initialize

if num_steps < 0:
    logging.error("Account value list has insufficient data for date alignment.")
    # Cannot proceed with date-based analysis if no steps were taken or list is malformed.
    # Set account_value_df index to simple range to avoid downstream errors expecting an index.
    account_value_df.index = pd.RangeIndex(start=0, stop=len(account_value_df))

elif initial_day_index + num_steps > len(env_test_instance.date_index):
    logging.error(
        f"Date alignment error: Required end index {initial_day_index + num_steps} exceeds date_index length {len(env_test_instance.date_index)}."
    )
    # Cannot proceed reliably. Set simple index.
    account_value_df.index = pd.RangeIndex(start=0, stop=len(account_value_df))

else:
    # Get dates corresponding to the *end* of each step
    step_dates_slice = slice(initial_day_index, initial_day_index + num_steps)
    step_dates = env_test_instance.date_index[step_dates_slice]

    if len(step_dates) != num_steps:
        logging.warning(
            f"Date slicing mismatch during backtest processing: Expected {num_steps} step dates, got {len(step_dates)}. Slice was {step_dates_slice}."
        )
        # If slicing failed, we cannot reliably create a date index.
        account_value_df.index = pd.RangeIndex(start=0, stop=len(account_value_df))
    else:
        # Use the first step's date for the initial account value timestamp
        initial_date_for_df = step_dates[0]
        test_dates = [
            initial_date_for_df
        ] + step_dates.tolist()  # Combine initial date + step dates

        if len(test_dates) == len(account_value_df):
            account_value_df["date"] = pd.to_datetime(test_dates)
            account_value_df = account_value_df.set_index("date")
            logging.info("Successfully aligned dates and set date index.")
        else:
            # This case should ideally not happen with the new logic, but check just in case.
            logging.error(
                f"CRITICAL: Final date alignment mismatch. Values: {len(account_value_list)}, Dates: {len(test_dates)}. Setting simple range index."
            )
            account_value_df.index = pd.RangeIndex(start=0, stop=len(account_value_df))

# Ensure account_value_df has an index before proceeding
if account_value_df.index is None:
    logging.warning(
        "Account value DataFrame index is None after date alignment attempt. Setting simple range index."
    )
    account_value_df.index = pd.RangeIndex(start=0, stop=len(account_value_df))

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
daily_returns_series.name = "Daily Return"  # Rename for Pyfolio

# --- Generate Pyfolio Report ---
logging.info("Generating Pyfolio tearsheet...")
# Note: Pyfolio might require specific data frequency or additional inputs (e.g., benchmark)
# For simplicity, we generate a basic tearsheet first.
try:
    # Ensure index is timezone-naive or UTC for Pyfolio
    if daily_returns_series.index.tz is not None:
        daily_returns_series.index = daily_returns_series.index.tz_localize(None)

    # Define output path for the tearsheet
    tearsheet_path = os.path.join(
        BACKTEST_RESULTS_DIR, f"{MODEL_FILENAME_BASE}_pyfolio_tearsheet.html"
    )

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
    logging.warning(
        "Skipping backtest_plot due to dependency version conflicts (pyfolio/pandas/numpy)."
    )

    # --- Calculate and Save Stats (with detailed logging) ---
    stats_str = None # Initialize
    try:
        logging.info("Attempting to calculate backtest statistics using finrl.plot.backtest_stats...")
        stats_str = backtest_stats(
            account_value=account_value_df_for_finrl, value_col_name="account_value"
        )
        logging.info("Successfully calculated backtest statistics.")
        print("\n--- Backtesting Performance Stats ---")
        print(stats_str)
    except Exception as e_stats:
        logging.error(f"ERROR calculating backtest_stats: {e_stats}", exc_info=True)
        # Do not proceed to save if calculation failed

    if stats_str is not None:
        try:
            logging.info("Attempting to save backtest statistics to file...")
            stats_filename = (
                f"{MODEL_FILENAME_BASE}_backtest_stats.txt"  # Base already includes suffix
            )
            stats_path = os.path.join(BACKTEST_RESULTS_DIR, stats_filename)
            with open(stats_path, "w") as f:
                f.write(stats_str.to_string())  # Convert Series to string before writing
            logging.info(f"Successfully saved backtest stats to {stats_path}")
        except Exception as e_save:
            logging.error(f"ERROR saving backtest stats to file {stats_path}: {e_save}", exc_info=True)
    else:
        logging.warning("Skipping saving stats file because stats calculation failed.")
    # --- End Calculate and Save Stats ---


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
    logging.error(
        f"Error during Pyfolio report generation or saving stats: {e}", exc_info=True
    )

# --- Generate and Save Matplotlib Plots (using potentially suffixed names) ---
plot_filename = (
    f"{MODEL_FILENAME_BASE}_account_value_plot.png"  # Base already includes suffix
)
plot_save_path = os.path.join(BACKTEST_RESULTS_DIR, plot_filename)
plot_account_value(
    account_value_df, plot_save_path
)  # Call the account value plot function

prices_plot_filename = (
    f"{MODEL_FILENAME_BASE}_stock_prices_plot.png"  # Base already includes suffix
)
prices_plot_save_path = os.path.join(BACKTEST_RESULTS_DIR, prices_plot_filename)
# We need the original test_df and the list of tickers
plot_stock_prices(
    test_df, tickers, prices_plot_save_path
)  # Call the new stock prices plot function

# --- Save Results (using potentially suffixed name) ---
results_filename = (
    f"{MODEL_FILENAME_BASE}_account_values.csv"  # Base already includes suffix
)
results_df_path = os.path.join(BACKTEST_RESULTS_DIR, results_filename)
account_value_df.to_csv(results_df_path)
logging.info(f"Account values saved to {results_df_path}")

logging.info("--- Backtesting Script Finished ---")
