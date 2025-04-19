# alpaca_trading_agent/src/backtesting/walkforward_backtest.py

import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np # Added for seeding
import random # Added for seeding
import torch # Added for seeding
from dateutil.relativedelta import relativedelta
import subprocess
import json

# --- Project Setup & Configuration Loading ---
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

# Add config dir to sys.path to import settings
sys.path.insert(0, CONFIG_DIR)
try:
    import settings
except ImportError as e:
    logging.error(
        f"Error importing configuration: {e}. Ensure config/settings.py exists."
    )
    sys.exit(1)

# Define paths to other scripts called by this one
MAIN_SCRIPT = os.path.join(
    PROJECT_ROOT, "main.py"
)  # To call tune, train, backtest modes

# --- Argument Parsing ---
parser = argparse.ArgumentParser(
    description="Run Walk-Forward Backtesting for the Alpaca Trading Agent."
)
parser.add_argument(
    "--log-level",
    type=str,
    default="INFO",
    help="Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)",
)
# Add other relevant arguments, e.g., skip tuning?
parser.add_argument(
    "--skip-tuning",
    action="store_true",  # If present, evaluates to True
    help="Skip the hyperparameter tuning step for each window.",
)

args = parser.parse_args()

# Logging setup will be done explicitly using the shared utility
# Add src to sys.path to find utils
sys.path.insert(0, SRC_DIR)
try:
    # Import both functions now
    from utils.logging_setup import configure_file_logging, add_console_logging
except ImportError:
     # Fallback if the utility somehow isn't found
    logging.basicConfig(
        level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.error(
        "Failed to import logging setup functions from src.utils. Cannot configure logging."
    )
    # Define dummy functions to prevent NameErrors later if import fails
    def configure_file_logging(level): pass
    def add_console_logging(level): pass

# --- Configure Logging for this script ---
try:
    configure_file_logging(args.log_level)  # Configure logging based on argument
    add_console_logging(args.log_level) # ADDED THIS CALL
    logging.info(f"--- Walk-Forward Script Logging Initialized (Level: {args.log_level.upper()}) ---")
except Exception as e:
    # Use basic print if logging setup itself fails
    print(f"ERROR setting up logging: {e}", file=sys.stderr)
    # Fallback basic config to console if setup fails
    logging.basicConfig(
        level=args.log_level.upper(), format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.error(f"Failed to configure logging via utility: {e}", exc_info=True)
    # sys.exit(1) # Optionally exit

# Matplotlib log level is also set by the caller.


# --- Helper Function to Run Main Script ---
# Replicates the logic from main.py's run_script, but calls main.py itself
def run_main_mode(
    mode,
    train_start=None,
    train_end=None,
    test_start=None,
    test_end=None,
    log_level_str="INFO",
    model_suffix="",
):
    """
    Runs a specific mode of main.py using subprocess, potentially overriding date settings.
    Uses environment variables for overrides as it's simpler than modifying all scripts
    to accept date args directly.
    """
    logging.info(f"--- Executing main.py in mode: {mode} {model_suffix} ---")
    temp_env = os.environ.copy()  # Copy current environment

    # Set environment variables for date overrides if provided
    if train_start:
        temp_env["WF_OVERRIDE_TRAIN_START"] = train_start
    if train_end:
        temp_env["WF_OVERRIDE_TRAIN_END"] = train_end
    if test_start:
        temp_env["WF_OVERRIDE_TEST_START"] = test_start
    if test_end:
        temp_env["WF_OVERRIDE_TEST_END"] = test_end
    # Add model suffix override if needed for training/backtesting specific window models
    if model_suffix:
        temp_env["WF_MODEL_SUFFIX"] = model_suffix

    try:
        command = [sys.executable, MAIN_SCRIPT, mode, "--log-level", log_level_str]
        # Pass --skip-tuning flag to walkforward mode if set
        if mode == 'walkforward' and args.skip_tuning:
             command.append('--skip-tuning')

        logging.debug(f"Running command: {' '.join(command)} with overrides")
        # Changed: Allow output to stream directly, don't capture
        result = subprocess.run(
            command, check=True, text=True, env=temp_env # Removed capture_output=True
        )

        logging.info(
            f"--- Successfully finished main.py mode: {mode} {model_suffix} ---"
        )
        return True
    except FileNotFoundError:
        logging.error(f"Main script not found: {MAIN_SCRIPT}")
        return False
    except subprocess.CalledProcessError as e:
        logging.error(f"Error executing main.py mode {mode}:")
        logging.error(f"Return Code: {e.returncode}")
        # Output/Error is streamed, so not captured in e.stdout/e.stderr
        # logging.error(f"Output:\n{e.stdout}") # No longer available
        # logging.error(f"Error Output:\n{e.stderr}") # No longer available
        logging.error(f"Check console/log file for detailed output from the failed script.")
        return False
    except Exception as e:
        logging.error(
            f"An unexpected error occurred while running main.py mode {mode}: {e}",
            exc_info=True,
        )
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
        current_train_end_str = (current_train_end - pd.Timedelta(days=1)).strftime(
            "%Y-%m-%d"
        )
        current_test_end_str = (current_test_end - pd.Timedelta(days=1)).strftime(
            "%Y-%m-%d"
        )
        current_train_start_str = current_train_start.strftime("%Y-%m-%d")
        current_test_start_str = current_test_start.strftime("%Y-%m-%d")

        # Check if the test period goes beyond the overall end date
        if current_test_start >= wf_end_date:
            logging.info(
                "Test period start date exceeds overall end date. Stopping walk-forward."
            )
            break
        # Adjust the final test end date if it overshoots
        if current_test_end > wf_end_date:
            current_test_end = wf_end_date
            current_test_end_str = current_test_end.strftime("%Y-%m-%d")
            logging.info(f"Adjusting final test end date to {current_test_end_str}")

        logging.info(
            f"Window {window_num}: Train {current_train_start_str} - {current_train_end_str}"
        )
        logging.info(
            f"Window {window_num}: Test  {current_test_start_str} - {current_test_end_str}"
        )

        # Define a unique suffix for models/results of this window
        model_suffix = (
            f"_wf{window_num}_{current_test_start_str}_{current_test_end_str}"
        )

        # --- Run Steps for the Current Window ---

        # 1. (Optional) Tuning
        if not args.skip_tuning:
            logging.info(f"--- Running Tuning for Window {window_num} ---")
            success = run_main_mode(
                "tune",
                train_start=current_train_start_str,
                train_end=current_train_end_str,
                log_level_str=args.log_level,
                model_suffix=model_suffix,  # Pass suffix to potentially save tuned params specific to window
            )
            if not success:
                logging.error(
                    f"Tuning failed for window {window_num}. Skipping window."
                )
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
            model_suffix=model_suffix,  # Ensure train script saves model with suffix
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
            log_level_str="DEBUG",  # Run backtest with DEBUG for detailed logs if needed
            model_suffix=model_suffix,  # Ensure backtest script loads model with suffix
        )
        if not success:
            logging.error(
                f"Backtesting failed for window {window_num}. Skipping window."
            )
            current_train_start = add_years(current_train_start, step_years)
            continue

        # 4. Collect Results
        # TODO: Modify backtest script to output results in JSON for more robust parsing.
        # Current manual parsing is fragile.
        try:
            # Construct filename (Now uses settings.TOTAL_TIMESTEPS)
            stats_filename = f"PPO_{settings.TIME_INTERVAL}_{settings.TOTAL_TIMESTEPS}{model_suffix}_backtest_stats.txt"
            stats_filepath = os.path.join(RESULTS_DIR, "backtesting", stats_filename)
            logging.info(f"Attempting to read results from: {stats_filepath}")

            # Basic parsing
            with open(stats_filepath, "r") as f:
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
                        # Handle percentage values
                        if parts[-1].endswith('%'):
                            metric_value = float(parts[-1].strip('%')) / 100.0
                        else:
                            metric_value = float(parts[-1])
                    except ValueError:
                        metric_value = parts[
                            -1
                        ]  # Keep as string if not float (e.g., NaN)
                    period_results[metric_name] = metric_value
            all_period_results.append(period_results)
            logging.info(f"Successfully collected results for window {window_num}")
            logging.debug(f"Window {window_num} Results: {period_results}")

        except FileNotFoundError:
            logging.error(
                f"Result file not found for window {window_num}: {stats_filepath}"
            )
        except AttributeError as e:
            logging.error(
                f"AttributeError during results collection for window {window_num} (likely missing setting): {e}",
                exc_info=True,
            )
        except Exception as e:
            logging.error(
                f"Error parsing results for window {window_num}: {e}", exc_info=True
            )

        # --- Advance to the next training window start date ---
        current_train_start = add_years(current_train_start, step_years)

    # --- Aggregate and Save Overall Results ---
    logging.info("\n===== Walk-Forward Analysis Complete =====")
    if not all_period_results:
        logging.warning("No results collected from any window.")
        return

    results_df = pd.DataFrame(all_period_results)
    results_df = results_df.set_index("window")  # Use window number as index

    # Save detailed results per window
    detailed_results_path = os.path.join(
        RESULTS_DIR, "backtesting", "walkforward_detailed_results.csv"
    )
    try:
        results_df.to_csv(detailed_results_path)
        logging.info(f"Detailed walk-forward results saved to: {detailed_results_path}")
    except Exception as e:
        logging.error(f"Failed to save detailed results: {e}")

    # Calculate and print summary statistics
    logging.info("\n--- Overall Walk-Forward Summary ---")
    try:
        logging.debug(
            f"Results DataFrame before summary calculation:\n{results_df.to_string()}"
        )

        # Select only numeric columns for averaging (excluding dates and potentially 'Skew', 'Kurtosis' if non-numeric)
        numeric_cols = results_df.select_dtypes(include=np.number).columns
        logging.info(f"Attempting to average numeric columns: {numeric_cols.tolist()}")


        if not numeric_cols.empty:
            summary = results_df[numeric_cols].mean()
            logging.info("Successfully calculated mean summary.")
            logging.info(
                f"Calculated Summary Series:\n{summary.to_string()}"
            )  # Log the summary Series

            # TODO: Calculate compounded return across all test periods

            summary_path = os.path.join(
                RESULTS_DIR, "backtesting", "walkforward_summary_results.csv"
            )
            summary_df = summary.to_frame(
                name="Average Value"
            )  # Convert Series to DataFrame

            if not summary_df.empty:
                logging.info(
                    f"Attempting to save summary DataFrame (Shape: {summary_df.shape}) to: {summary_path}"
                )
                summary_df.to_csv(summary_path, header=True)  # Save DataFrame
                logging.info(
                    f"Successfully saved summary walk-forward results to: {summary_path}"
                )
            else:
                logging.warning(
                    "Summary DataFrame was empty after calculation, skipping save."
                )
        else:
            logging.warning(
                "No numeric columns found in results DataFrame to calculate summary."
            )

    except Exception as e:
        # Log the specific error encountered during summary calculation or saving
        logging.error(
            f"Error calculating or saving summary statistics: {e}", exc_info=True
        )

    # TODO: Generate combined plots (e.g., equity curve across all test periods)


if __name__ == "__main__":
    # --- Seed for Reproducibility ---
    # Note: This seeds the main orchestration script. Subprocesses (tune, train, backtest)
    # should have their own seeding logic for full reproducibility of their internal operations.
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
    logging.info(f"Global random seeds set for walkforward script: {SEED}")
    # --- End Seeding ---

    # NOTE: This script assumes that the other scripts (tune, train, backtest)
    # can be driven by environment variables WF_OVERRIDE_*_DATE and WF_MODEL_SUFFIX,
    # and that those scripts implement their own seeding.
    # Also, preprocess script must have been run to generate data covering the full WF period.

    logging.warning("Ensure preprocessing covers the full WF period.")
    logging.warning(
        "Ensure tune/train/backtest scripts handle WF_* overrides, suffix, and internal seeding."
    )

    perform_walk_forward()

    logging.info("--- Walk-Forward Script Finished ---")
