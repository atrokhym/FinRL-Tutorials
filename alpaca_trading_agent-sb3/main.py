# alpaca_trading_agent/main.py

import argparse
import argparse
import logging
# import logging.handlers # No longer needed here
import os
import sys
import subprocess # To call other scripts

# --- Project Setup ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
# LOG_DIR and LOG_FILE are now defined in the utility
# Ensure src is in path to find utils module
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
try:
    # Import the setup functions from the new utility module
    from utils.logging_setup import configure_file_logging, add_console_logging
except ImportError:
    logging.basicConfig(level=logging.ERROR) # Basic config for error message
    logging.error("Failed to import logging_setup from src.utils. Ensure src/utils/logging_setup.py exists.")
    sys.exit(1)
DATA_FETCHER_SCRIPT = os.path.join(SRC_DIR, 'data_fetcher', 'fetch_data.py')
PREPROCESS_SCRIPT = os.path.join(SRC_DIR, 'preprocessing', 'preprocess_data.py')
TRAIN_SCRIPT = os.path.join(SRC_DIR, 'training', 'train_agent.py')
TUNE_SCRIPT = os.path.join(SRC_DIR, 'training', 'tune_agent.py') # Added tuning script path
BACKTEST_SCRIPT = os.path.join(SRC_DIR, 'backtesting', 'backtest_agent.py')
WALKFORWARD_SCRIPT = os.path.join(SRC_DIR, 'backtesting', 'walkforward_backtest.py') # Added walkforward script path
PAPERTRADE_SCRIPT = os.path.join(SRC_DIR, 'trading', 'alpaca_papertrader.py')

# Path setup moved above import

# --- Helper Function to Run Scripts ---
def run_script(script_path, log_level_str="INFO", extra_args=None):
    """
    Runs a Python script using subprocess, passing the log level and optional extra args.
    """
    logging.info(f"Executing script: {script_path} with log level {log_level_str}")
    try:
        # Use sys.executable to ensure the correct Python interpreter is used
        command = [sys.executable, script_path, '--log-level', log_level_str]
        if extra_args:
            command.extend(extra_args) # Add extra arguments if provided
        logging.debug(f"Running command: {' '.join(command)}")
        # Note: Subprocess environment is inherited by default
        # Removed capture_output=True to allow subprocess output to stream to console
        result = subprocess.run(command, check=True, text=True)
        # # Removed logging of captured stdout/stderr as it's no longer captured
        # logging.debug(f"Script Output:\n{result.stdout}")
        # if result.stderr:
        #     logging.debug(f"Script Error Output:\n{result.stderr}")
        logging.info(f"Script {script_path} finished successfully (Return Code: {result.returncode}).")
        return True
    except FileNotFoundError:
        logging.error(f"Script not found: {script_path}")
        return False
    except subprocess.CalledProcessError as e:
        logging.error(f"Error executing script {script_path}:")
        logging.error(f"Return Code: {e.returncode}")
        # Stderr/Stdout might be None if capture_output was False, or might be printed directly
        if e.stdout:
            logging.error(f"Output:\n{e.stdout}")
        if e.stderr:
            logging.error(f"Error Output:\n{e.stderr}")
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred while running {script_path}: {e}", exc_info=True)
        return False

# --- Argument Parsing ---
def setup_logging(level_str="INFO", enable_console=True):
    """Configures logging using the shared utility."""
    # Configure file logging (mandatory)
    configure_file_logging(level_str)

    # Optionally add console logging
    if enable_console:
        add_console_logging(level_str)

    # The utility already handles noisy libraries and initial log messages.
    logging.info(f"--- Main Orchestrator Logging Initialized (Level: {level_str.upper()}) ---")
    logging.info(f"Console logging enabled: {enable_console}")
def main():
    parser = argparse.ArgumentParser(description="Alpaca Trading Agent Orchestrator")
    # Main mode argument
    parser.add_argument(
        'mode',
        choices=['fetch', 'preprocess', 'tune', 'train', 'backtest', 'walkforward', 'papertrade', 'all'],
        help="The primary operation mode."
    )
    # General log level
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help="Set the logging level for the orchestrator and default for scripts."
    )
    # Mode-specific arguments (example for walkforward)
    parser.add_argument(
        '--skip-tuning',
        action='store_true',
        help="[Walkforward Mode Only] Skip hyperparameter tuning for each window."
    )
    # Argument for dynamic training end date
    parser.add_argument(
        '--train-end-date',
        type=str,
        default=None, # Default to None, meaning settings.py dates will be used unless overridden
        help="Specify the end date for training data in YYYY-MM-DD format. Overrides settings.py TRAIN_END_DATE for fetch/preprocess."
    )

    args = parser.parse_args()

    # --- Validate arguments ---
    # Ensure --skip-tuning is only used with walkforward mode
    if args.skip_tuning and args.mode != 'walkforward':
        parser.error("--skip-tuning argument is only valid for the 'walkforward' mode.")


    # Configure logging based on the argument
    setup_logging(args.log_level)

    logging.info(f"--- Running in mode: {args.mode} ---")
    if args.mode == 'walkforward' and args.skip_tuning:
        logging.info("Walkforward: --skip-tuning flag is set.")


    # Determine the log level to pass to scripts
    script_log_level = args.log_level
    if args.mode == 'backtest' or (args.mode == 'all' and 'backtest' in sys.argv): # Check if backtest is part of 'all'
         script_log_level_backtest = 'DEBUG'
         logging.info("Setting log level for backtest script to DEBUG")
    # Removed duplicated code block that was causing indentation/syntax errors

    else:
         script_log_level_backtest = script_log_level # Use default for non-backtest steps in 'all'

    # --- Prepare Extra Arguments ---
    # Arguments to pass specifically to fetch/preprocess if --train-end-date is given
    fetch_preprocess_extra_args = []
    if args.train_end_date:
        logging.info(f"Overriding training end date with command-line argument: {args.train_end_date}")
        fetch_preprocess_extra_args.extend(['--train-end-date', args.train_end_date])

    if args.mode == 'fetch' or args.mode == 'all':
        logging.info("--- Step: Fetching Data ---")
        if not run_script(DATA_FETCHER_SCRIPT, script_log_level, extra_args=fetch_preprocess_extra_args):
            logging.error("Data fetching failed. Aborting.")
            if args.mode == 'all': sys.exit(1)

    if args.mode == 'preprocess' or args.mode == 'all':
        logging.info("--- Step: Preprocessing Data ---")
        if not run_script(PREPROCESS_SCRIPT, script_log_level, extra_args=fetch_preprocess_extra_args):
            logging.error("Data preprocessing failed. Aborting.")
            if args.mode == 'all': sys.exit(1)

    if args.mode == 'tune' or args.mode == 'all':
        logging.info("--- Step: Tuning Agent Hyperparameters ---")
        if not run_script(TUNE_SCRIPT, script_log_level):
            logging.error("Agent tuning failed. Aborting.")
            if args.mode == 'all': sys.exit(1)

    if args.mode == 'train' or args.mode == 'all':
        logging.info("--- Step: Training Agent ---")
        if not run_script(TRAIN_SCRIPT, script_log_level):
            logging.error("Agent training failed. Aborting.")
            if args.mode == 'all': sys.exit(1)


    if args.mode == 'backtest' or args.mode == 'all':
        logging.info("--- Step: Backtesting Agent ---")
        # Use the potentially overridden DEBUG level for backtesting
        if not run_script(BACKTEST_SCRIPT, script_log_level_backtest):
            logging.error("Agent backtesting failed.")
            # Don't necessarily exit if 'all' mode, just report failure

    if args.mode == 'walkforward':
        logging.info("--- Step: Running Walk-Forward Backtesting ---")
        wf_extra_args = []
        if args.skip_tuning:
            wf_extra_args.append('--skip-tuning')
        # Call walkforward script, passing the log level and the extra arg if present
        if not run_script(WALKFORWARD_SCRIPT, script_log_level, extra_args=wf_extra_args):
             logging.error("Walk-forward backtesting failed.")

    if args.mode == 'papertrade': # Removed 'all' condition for papertrade
        logging.info("--- Step: Starting Paper Trading Agent ---")
        # Note: The papertrade script runs indefinitely due to the schedule loop
        if not run_script(PAPERTRADE_SCRIPT, script_log_level): # Use default log level
            logging.error("Failed to start paper trading agent.")

    logging.info(f"--- Mode '{args.mode}' finished. ---")


if __name__ == "__main__":
    main()
