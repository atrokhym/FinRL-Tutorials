# alpaca_trading_agent/main.py

import argparse
import argparse
import logging
import os
import sys
import subprocess # To call other scripts

# --- Project Setup ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
DATA_FETCHER_SCRIPT = os.path.join(SRC_DIR, 'data_fetcher', 'fetch_data.py')
PREPROCESS_SCRIPT = os.path.join(SRC_DIR, 'preprocessing', 'preprocess_data.py')
TRAIN_SCRIPT = os.path.join(SRC_DIR, 'training', 'train_agent.py')
BACKTEST_SCRIPT = os.path.join(SRC_DIR, 'backtesting', 'backtest_agent.py')
PAPERTRADE_SCRIPT = os.path.join(SRC_DIR, 'trading', 'alpaca_papertrader.py')

# Ensure src directory is in path if needed (though scripts should handle their own imports)
# sys.path.insert(0, SRC_DIR)

# --- Helper Function to Run Scripts ---
def run_script(script_path, log_level_str="INFO"):
    """Runs a Python script using subprocess, passing the log level."""
    logging.info(f"Executing script: {script_path} with log level {log_level_str}")
    try:
        # Use sys.executable to ensure the correct Python interpreter is used
        command = [sys.executable, script_path, '--log-level', log_level_str]
        logging.debug(f"Running command: {' '.join(command)}")
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        # Log stdout/stderr at debug level to avoid cluttering INFO logs unless needed
        logging.debug(f"Script Output:\n{result.stdout}")
        if result.stderr:
            logging.debug(f"Script Error Output:\n{result.stderr}") # Use debug for stderr too
        logging.info(f"Script {script_path} finished successfully.")
        return True
    except FileNotFoundError:
        logging.error(f"Script not found: {script_path}")
        return False
    except subprocess.CalledProcessError as e:
        logging.error(f"Error executing script {script_path}:")
        logging.error(f"Return Code: {e.returncode}")
        logging.error(f"Output:\n{e.stdout}")
        logging.error(f"Error Output:\n{e.stderr}")
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred while running {script_path}: {e}", exc_info=True)
        return False

# --- Argument Parsing ---
def setup_logging(level_str="INFO"):
    """Configures the root logger."""
    level = getattr(logging, level_str.upper(), logging.INFO)
    # Remove existing handlers to avoid duplicate messages if re-configured
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    # Configure root logger first
    logging.basicConfig(level=level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', force=True)
    # Set higher level for noisy libraries like matplotlib
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING) # Be specific if needed
    logging.info(f"Root logging configured to level: {level_str.upper()}")
    logging.info(f"Matplotlib logging level set to WARNING to reduce verbosity.")


def main():
    parser = argparse.ArgumentParser(description="Alpaca Trading Agent Orchestrator")
    parser.add_argument(
        'mode',
        choices=['fetch', 'preprocess', 'train', 'backtest', 'papertrade', 'all'],
        help=(
            "Mode to run: "
            "'fetch' - Fetch raw data. "
            "'preprocess' - Preprocess raw data. "
            "'train' - Train the RL agent. "
            "'backtest' - Backtest the trained agent. "
            "'papertrade' - Start the paper trading agent (runs continuously). "
            "'all' - Run fetch, preprocess, train, and backtest sequentially."
        )
    )
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help="Set the logging level for the orchestrator and default for scripts."
    )
    # Add other arguments if needed, e.g., specific config file, model name

    args = parser.parse_args()

    # Configure logging based on the argument
    setup_logging(args.log_level)

    logging.info(f"--- Running in mode: {args.mode} ---")

    # Determine the log level to pass to scripts
    script_log_level = args.log_level
    if args.mode == 'backtest' or (args.mode == 'all' and 'backtest' in sys.argv): # Check if backtest is part of 'all'
         script_log_level_backtest = 'DEBUG'
         logging.info("Setting log level for backtest script to DEBUG")
    else:
         script_log_level_backtest = script_log_level # Use default for non-backtest steps in 'all'

    if args.mode == 'fetch' or args.mode == 'all':
        logging.info("--- Step: Fetching Data ---")
        if not run_script(DATA_FETCHER_SCRIPT, script_log_level):
            logging.error("Data fetching failed. Aborting.")
            if args.mode == 'all': sys.exit(1)

    if args.mode == 'preprocess' or args.mode == 'all':
        logging.info("--- Step: Preprocessing Data ---")
        if not run_script(PREPROCESS_SCRIPT, script_log_level):
            logging.error("Data preprocessing failed. Aborting.")
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

    if args.mode == 'papertrade':
        logging.info("--- Step: Starting Paper Trading Agent ---")
        # Note: The papertrade script runs indefinitely due to the schedule loop
        if not run_script(PAPERTRADE_SCRIPT, script_log_level): # Use default log level
            logging.error("Failed to start paper trading agent.")

    logging.info(f"--- Mode '{args.mode}' finished. ---")

if __name__ == "__main__":
    main()
