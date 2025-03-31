# alpaca_trading_agent/main.py

import argparse
import logging
import os
import sys
import subprocess # To call other scripts

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
def run_script(script_path):
    """Runs a Python script using subprocess."""
    logging.info(f"Executing script: {script_path}")
    try:
        # Use sys.executable to ensure the correct Python interpreter is used
        result = subprocess.run([sys.executable, script_path], check=True, capture_output=True, text=True)
        logging.info(f"Script Output:\n{result.stdout}")
        if result.stderr:
            logging.warning(f"Script Error Output:\n{result.stderr}")
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
    # Add other arguments if needed, e.g., specific config file, model name

    args = parser.parse_args()

    logging.info(f"--- Running in mode: {args.mode} ---")

    if args.mode == 'fetch' or args.mode == 'all':
        logging.info("--- Step: Fetching Data ---")
        if not run_script(DATA_FETCHER_SCRIPT):
            logging.error("Data fetching failed. Aborting.")
            if args.mode == 'all': sys.exit(1)

    if args.mode == 'preprocess' or args.mode == 'all':
        logging.info("--- Step: Preprocessing Data ---")
        if not run_script(PREPROCESS_SCRIPT):
            logging.error("Data preprocessing failed. Aborting.")
            if args.mode == 'all': sys.exit(1)

    if args.mode == 'train' or args.mode == 'all':
        logging.info("--- Step: Training Agent ---")
        if not run_script(TRAIN_SCRIPT):
            logging.error("Agent training failed. Aborting.")
            if args.mode == 'all': sys.exit(1)

    if args.mode == 'backtest' or args.mode == 'all':
        logging.info("--- Step: Backtesting Agent ---")
        if not run_script(BACKTEST_SCRIPT):
            logging.error("Agent backtesting failed.")
            # Don't necessarily exit if 'all' mode, just report failure

    if args.mode == 'papertrade':
        logging.info("--- Step: Starting Paper Trading Agent ---")
        # Note: The papertrade script runs indefinitely due to the schedule loop
        if not run_script(PAPERTRADE_SCRIPT):
            logging.error("Failed to start paper trading agent.")

    logging.info(f"--- Mode '{args.mode}' finished. ---")

if __name__ == "__main__":
    main()
