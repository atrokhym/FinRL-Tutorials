# alpaca_trading_agent/src/backtesting/walkforward_backtest.py

import os
import sys
import logging
import argparse
import pandas as pd
from dateutil.relativedelta import relativedelta
import subprocess
import json

# --- Project Setup & Configuration Loading ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_DIR = os.path.join(PROJECT_ROOT, 'config')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')

# Add config dir to sys.path to import settings
sys.path.insert(0, CONFIG_DIR)
try:
    import settings
except ImportError as e:
    logging.error(f"Error importing configuration: {e}. Ensure config/settings.py exists.")
    sys.exit(1)

# Define paths to other scripts called by this one
MAIN_SCRIPT = os.path.join(PROJECT_ROOT, 'main.py') # To call tune, train, backtest modes

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Run Walk-Forward Backtesting for the Alpaca Trading Agent.")
parser.add_argument(
    '--log-level',
    type=str,
    default='INFO',
    help='Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)'
)
# Add other relevant arguments, e.g., skip tuning?
parser.add_argument(
    '--skip-tuning',
    action='store_true', # If present, evaluates to True
    help='Skip the hyperparameter tuning step for each window.'
)

args = parser.parse_args()

# --- Logging Setup ---
def setup_logging(level_str="INFO"):
    level = getattr(logging, level_str.upper(), logging.INFO)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=level, format='%(asctime)s - %(name)s (WF) - %(levelname)s - %(message)s', force=True)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logging.info(f"Walk-Forward Backtester logging configured to level: {level_str.upper()}")

setup_logging(args.log_level)

# --- Helper Function to Run Main Script ---
# Replicates the logic from main.py's run_script, but calls main.py itself
def run_main_mode(mode, train_start=None, train_end=None, test_start=None, test_end=None, log_level_str="INFO", model_suffix=""):
    """
    Runs a specific mode of main.py using subprocess, potentially overriding date settings.
    Uses environment variables for overrides as it's simpler than modifying all scripts
    to accept date args directly.
    """
    logging.info(f"--- Executing main.py in mode: {mode} {model_suffix} ---")
    temp_env = os.environ.copy() # Copy current environment

    # Set environment variables for date overrides if provided
    if train_start: temp_env['WF_OVERRIDE_TRAIN_START'] = train_start
    if train_end: temp_env['WF_OVERRIDE_TRAIN_END'] = train_end
    if test_start: temp_env['WF_OVERRIDE_TEST_START'] = test_start
    if test_end: temp_env['WF_OVERRIDE_TEST_END'] = test_end
    # Add model suffix override if needed for training/backtesting specific window models
    if model_suffix: temp_env['WF_MODEL_SUFFIX'] = model_suffix

    try:
        command = [sys.executable, MAIN_SCRIPT, mode, '--log-level', log_level_str]
        logging.debug(f"Running command: {' '.join(command)} with overrides")
        result = subprocess.run(command, check=True, capture_output=True, text=True, env=temp_env)

        # Log output (consider logging DEBUG output only at DEBUG level)
        if args.log_level == 'DEBUG':
             logging.debug(f"Script Output:\n{result.stdout}")
             if result.stderr:
                 logging.debug(f"Script Error Output:\n{result.stderr}")
        else:
            # For INFO level, maybe log just the last few lines or specific markers
            pass

        logging.info(f"--- Successfully finished main.py mode: {mode} {model_suffix} ---")
        return True
    except FileNotFoundError:
        logging.error(f"Main script not found: {MAIN_SCRIPT}")
        return False
    except subprocess.CalledProcessError as e:
        logging.error(f"Error executing main.py mode {mode}:")
        logging.error(f"Return Code: {e.returncode}")
        logging.error(f"Output:\n{e.stdout}")
        logging.error(f"Error Output:\n{e.stderr}")
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred while running main.py mode {mode}: {e}", exc_info=True)
        return False
    finally:
         # Clean up env vars (optional, as they only exist for the subprocess)
         pass


# --- Date Calculation Helper ---
def add_years(d, years):
    try:
        return d + relativedelta(years=years)
    except Exception as e:
        logging.error(f"Error adding {years} years to date {d}: {e}")
        # Fallback or raise error? For now, return None
        return None

# --- Main Walk-Forward Logic ---
def perform_walk_forward():
    logging.info("Starting Walk-Forward Backtesting Process...")

    # Load Walk-Forward parameters from settings
    try:
        wf_start_date_str = settings.WF_START_DATE
        wf_end_date_str = settings.WF_END_DATE
        train_window_years = settings.WF_TRAIN_WINDOW_YEARS
        step_years = settings.WF_STEP_YEARS

        wf_start_date = pd.to_datetime(wf_start_date_str)
        wf_end_date = pd.to_datetime(wf_end_date_str)
    except AttributeError as e:
        logging.error(f"Missing Walk-Forward parameter in settings.py: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error parsing Walk-Forward parameters: {e}")
        sys.exit(1)

    logging.info(f"Walk-Forward Period: {wf_start_date_str} to {wf_end_date_str}")
    logging.info(f"Initial Training Window: {train_window_years} years")
    logging.info(f"Step Size / Test Window: {step_years} years")

    all_period_results = []
    current_train_start = wf_start_date
    window_num = 0

    while True:
        window_num += 1
        logging.info(f"\n===== Processing Walk-Forward Window {window_num} =====")

        # Calculate dates for the current window
        current_train_end = add_years(current_train_start, train_window_years)
        current_test_start = current_train_end
        current_test_end = add_years(current_test_start, step_years)

        # Adjust end dates slightly to avoid overlap issues (e.g., end of day) if needed
        # For simplicity, assume start/end dates are inclusive/exclusive as needed by scripts
        current_train_end_str = (current_train_end - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        current_test_end_str = (current_test_end - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        current_train_start_str = current_train_start.strftime('%Y-%m-%d')
        current_test_start_str = current_test_start.strftime('%Y-%m-%d')


        # Check if the test period goes beyond the overall end date
        if current_test_start >= wf_end_date:
            logging.info("Test period start date exceeds overall end date. Stopping walk-forward.")
            break
        # Adjust the final test end date if it overshoots
        if current_test_end > wf_end_date:
            current_test_end = wf_end_date
            current_test_end_str = current_test_end.strftime('%Y-%m-%d')
            logging.info(f"Adjusting final test end date to {current_test_end_str}")

        logging.info(f"Window {window_num}: Train {current_train_start_str} - {current_train_end_str}")
        logging.info(f"Window {window_num}: Test  {current_test_start_str} - {current_test_end_str}")

        # Define a unique suffix for models/results of this window
        model_suffix = f"_wf{window_num}_{current_test_start_str}_{current_test_end_str}"

        # --- Run Steps for the Current Window ---

        # 1. (Optional) Tuning
        if not args.skip_tuning:
            logging.info(f"--- Running Tuning for Window {window_num} ---")
            success = run_main_mode(
                "tune",
                train_start=current_train_start_str,
                train_end=current_train_end_str,
                log_level_str=args.log_level,
                model_suffix=model_suffix # Pass suffix to potentially save tuned params specific to window
            )
            if not success:
                logging.error(f"Tuning failed for window {window_num}. Skipping window.")
                # Decide how to proceed: skip window, stop all? For now, skip.
                current_train_start = add_years(current_train_start, step_years)
                continue
        else:
            logging.info(f"--- Skipping Tuning for Window {window_num} ---")
            # Ensure best_params.json exists or handle default params in train script

        # 2. Training
        logging.info(f"--- Running Training for Window {window_num} ---")
        success = run_main_mode(
            "train",
            train_start=current_train_start_str,
            train_end=current_train_end_str,
            log_level_str=args.log_level,
            model_suffix=model_suffix # Ensure train script saves model with suffix
        )
        if not success:
            logging.error(f"Training failed for window {window_num}. Skipping window.")
            current_train_start = add_years(current_train_start, step_years)
            continue

        # 3. Backtesting
        logging.info(f"--- Running Backtesting for Window {window_num} ---")
        success = run_main_mode(
            "backtest",
            test_start=current_test_start_str,
            test_end=current_test_end_str,
            log_level_str='DEBUG', # Run backtest with DEBUG for detailed logs if needed
            model_suffix=model_suffix # Ensure backtest script loads model with suffix
        )
        if not success:
            logging.error(f"Backtesting failed for window {window_num}. Skipping window.")
            current_train_start = add_years(current_train_start, step_years)
            continue

        # 4. Collect Results
        # TODO: Modify backtest script to output results in JSON for more robust parsing.
        # Current manual parsing is fragile.
        try:
            # Construct filename (Now uses settings.TOTAL_TIMESTEPS)
            stats_filename = f"PPO_{settings.TIME_INTERVAL}_{settings.TOTAL_TIMESTEPS}{model_suffix}_backtest_stats.txt"
            stats_filepath = os.path.join(RESULTS_DIR, 'backtesting', stats_filename)
            logging.info(f"Attempting to read results from: {stats_filepath}")

            # Basic parsing
            with open(stats_filepath, 'r') as f:
                lines = f.readlines()
            period_results = {
                "window": window_num,
                "train_start": current_train_start_str,
                "train_end": current_train_end_str,
                "test_start": current_test_start_str,
                "test_end": current_test_end_str,
            }
            for line in lines:
                parts = line.split()
                if len(parts) >= 2:
                    metric_name = " ".join(parts[:-1]).strip()
                    try:
                        metric_value = float(parts[-1])
                    except ValueError:
                        metric_value = parts[-1] # Keep as string if not float (e.g., NaN)
                    period_results[metric_name] = metric_value
            all_period_results.append(period_results)
            logging.info(f"Successfully collected results for window {window_num}")
            logging.debug(f"Window {window_num} Results: {period_results}")

        except FileNotFoundError:
            logging.error(f"Result file not found for window {window_num}: {stats_filepath}")
        except AttributeError as e:
             logging.error(f"AttributeError during results collection for window {window_num} (likely missing setting): {e}", exc_info=True)
        except Exception as e:
            logging.error(f"Error parsing results for window {window_num}: {e}", exc_info=True)

        # --- Advance to the next training window start date ---
        current_train_start = add_years(current_train_start, step_years)


    # --- Aggregate and Save Overall Results ---
    logging.info("\n===== Walk-Forward Analysis Complete =====")
    if not all_period_results:
        logging.warning("No results collected from any window.")
        return

    results_df = pd.DataFrame(all_period_results)
    results_df = results_df.set_index('window') # Use window number as index

    # Save detailed results per window
    detailed_results_path = os.path.join(RESULTS_DIR, 'backtesting', 'walkforward_detailed_results.csv')
    try:
        results_df.to_csv(detailed_results_path)
        logging.info(f"Detailed walk-forward results saved to: {detailed_results_path}")
    except Exception as e:
        logging.error(f"Failed to save detailed results: {e}")

    # Calculate and print summary statistics
    logging.info("\n--- Overall Walk-Forward Summary ---")
    try:
         logging.debug(f"Results DataFrame before summary calculation:\n{results_df.to_string()}")

         # Explicitly define numeric columns to average (excluding Skew/Kurtosis which are NaN)
         numeric_cols_to_average = [
             'Annual return', 'Cumulative returns', 'Annual volatility',
             'Sharpe ratio', 'Calmar ratio', 'Stability', 'Max drawdown',
             'Omega ratio', 'Sortino ratio', 'Tail ratio', 'Daily value at risk'
         ]
         # Check which of these columns actually exist in the DataFrame
         existing_numeric_cols = [col for col in numeric_cols_to_average if col in results_df.columns]
         logging.info(f"Attempting to average columns: {existing_numeric_cols}")

         if existing_numeric_cols:
             # Calculate mean only for existing numeric columns
             summary = results_df[existing_numeric_cols].mean()
             logging.info("Successfully calculated mean summary.")
             logging.info(f"Calculated Summary Series:\n{summary.to_string()}") # Log the summary Series

             # TODO: Calculate compounded return across all test periods

             summary_path = os.path.join(RESULTS_DIR, 'backtesting', 'walkforward_summary_results.csv')
             summary_df = summary.to_frame(name='Average Value') # Convert Series to DataFrame

             if not summary_df.empty:
                 logging.info(f"Attempting to save summary DataFrame (Shape: {summary_df.shape}) to: {summary_path}")
                 summary_df.to_csv(summary_path, header=True) # Save DataFrame
                 logging.info(f"Successfully saved summary walk-forward results to: {summary_path}")
             else:
                 logging.warning("Summary DataFrame was empty after calculation, skipping save.")
         else:
             logging.warning("No numeric columns found in results DataFrame to calculate summary.")

    except Exception as e:
         # Log the specific error encountered during summary calculation or saving
         logging.error(f"Error calculating or saving summary statistics: {e}", exc_info=True)

    # TODO: Generate combined plots (e.g., equity curve across all test periods)

if __name__ == "__main__":
    # NOTE: This script assumes that the other scripts (tune, train, backtest)
    # can be driven by environment variables WF_OVERRIDE_*_DATE and WF_MODEL_SUFFIX.
    # Modifications might be needed in those scripts to read these env vars
    # and apply them appropriately (e.g., override settings.py values, modify file save paths).
    # Also, preprocess script must have been run to generate data covering the full WF period.

    logging.warning("Ensure preprocessing covers the full WF period.")
    logging.warning("Ensure tune/train/backtest scripts handle WF_* overrides and suffix.")

    perform_walk_forward()

    logging.info("--- Walk-Forward Script Finished ---")
