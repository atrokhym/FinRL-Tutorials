# alpaca_trading_agent/src/data_fetcher/fetch_data.py

import alpaca_trade_api as tradeapi
import pandas as pd
import os
from datetime import datetime
import logging
import argparse # Added argparse

# Logging setup will be done explicitly using the shared utility

# --- Configuration Loading ---
# Correctly navigate up two levels from src/data_fetcher to the project root
# then access config and data directories.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_DIR = os.path.join(PROJECT_ROOT, 'config')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# Add project root to sys.path to allow absolute imports from config
import sys
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT) # Add project root if not already there

# Import the logging setup utility
try:
    from src.utils.logging_setup import configure_file_logging
except ImportError:
    # Fallback if the utility somehow isn't found
    logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.error("Failed to import configure_file_logging from src.utils. Cannot configure file logging.")
    # Optionally exit if logging is critical
    # sys.exit(1)

try:
    from config import settings
    from config import api_keys  # Use absolute import from config package
except ImportError as e:
    logging.error(f"Error importing configuration: {e}. Ensure config/settings.py and config/api_keys.py exist and PROJECT_ROOT is added to sys.path.")
    sys.exit(1)

# Create data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

# --- Alpaca API Initialization ---
try:
    api = tradeapi.REST(
        key_id=api_keys.ALPACA_API_KEY,
        secret_key=api_keys.ALPACA_API_SECRET,
        base_url=api_keys.ALPACA_API_BASE_URL,
        api_version='v2'
    )
    account = api.get_account()
    logging.info(f"Connected to Alpaca. Account status: {account.status}")
except Exception as e:
    logging.error(f"Failed to connect to Alpaca: {e}")
    logging.error("Please ensure your API keys in config/api_keys.py are correct and the Alpaca service is running.")
    sys.exit(1)

# --- Data Fetching Function ---
def fetch_historical_data(tickers, start_date, end_date, timeframe='1Day', limit=10000):
    """
    Fetches historical OHLCV data for given tickers from Alpaca.

    Args:
        tickers (list): List of stock ticker symbols.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        timeframe (str): Alpaca API timeframe ('1Min', '5Min', '15Min', '1H', '1Day').
        limit (int): The maximum number of bars to fetch per request. Alpaca's limit is 10000 for v2.

    Returns:
        pd.DataFrame: DataFrame containing combined OHLCV data for all tickers,
                      with MultiIndex (date, tic) or None if fetching fails.
    """
    all_data = []
    start_dt = pd.Timestamp(start_date, tz='America/New_York').isoformat()
    end_dt = pd.Timestamp(end_date, tz='America/New_York').isoformat()

    logging.info(f"Fetching data for {tickers} from {start_date} to {end_date} ({timeframe})")

    for ticker in tickers:
        logging.info(f"Fetching data for {ticker}...")
        try:
            # Use get_bars for fetching historical data
            barset = api.get_bars(
                ticker,
                timeframe,
                start=start_dt,
                end=end_dt,
                adjustment='raw', # or 'split', 'dividend', 'all'
                limit=limit # Fetch maximum allowed bars per request
                # Note: For longer periods, pagination might be needed if limit is exceeded.
                # The alpaca-py library handles pagination automatically for get_bars.
            ).df
            barset['tic'] = ticker
            all_data.append(barset)
            logging.info(f"Successfully fetched {len(barset)} bars for {ticker}")
        except Exception as e:
            logging.error(f"Failed to fetch data for {ticker}: {e}")
            # Optionally continue to next ticker or raise error

    if not all_data:
        logging.warning("No data fetched for any ticker.")
        return None

    # Combine dataframes
    combined_df = pd.concat(all_data)

    # Ensure datetime index is timezone-aware (Alpaca returns UTC)
    if not combined_df.index.tz:
         combined_df = combined_df.tz_localize('UTC')
    # Convert to market time (usually Eastern Time) for easier analysis
    combined_df.index = combined_df.index.tz_convert('America/New_York')
    combined_df.index = combined_df.index.date # Keep only the date part if using daily data
    combined_df.index.name = 'date' # Explicitly name the index

    # Reset index to have 'date' and 'tic' as columns if needed by FinRL preprocessing
    # combined_df = combined_df.reset_index() # No rename needed if index is named 'date'
    # combined_df = combined_df.set_index(['date', 'tic']) # Or keep date as index, tic as column

    # Standardize column names (FinRL often expects lowercase)
    combined_df.columns = [col.lower() for col in combined_df.columns]

    logging.info(f"Finished fetching data. Shape: {combined_df.shape}")
    return combined_df

# Logging setup is now handled explicitly below using the shared utility
# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Fetcher for Alpaca Trading Agent")
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help="Set the logging level for the script."
    )
    parser.add_argument(
        '--train-end-date',
        type=str,
        default=None,
        help="Specify the end date for fetching data in YYYY-MM-DD format. Overrides settings.py TRAIN_END_DATE."
    )
    args = parser.parse_args()

    # --- Configure Logging for this script ---
    # Use the log level passed from main.py via command-line arguments
    configure_file_logging(args.log_level)
    # Note: Console logging is not added here by default. main.py handles its own console log.
    # If you needed console logging *from this script specifically*, you could call add_console_logging here.

    logging.info(f"--- Starting Data Fetching Script (PID: {os.getpid()}) ---")

    # Determine the end date to use
    end_date_to_use = args.train_end_date if args.train_end_date else settings.TRAIN_END_DATE
    if args.train_end_date:
        logging.info(f"Using command-line end date: {end_date_to_use}")
    else:
        logging.info(f"Using end date from settings: {end_date_to_use}")

    # Fetch data up to the determined end date
    raw_data = fetch_historical_data(
        tickers=settings.TICKERS,
        start_date=settings.TRAIN_START_DATE, # Start date is still from settings
        end_date=end_date_to_use,
        timeframe=settings.TIME_INTERVAL
    )

    if raw_data is not None and not raw_data.empty:
        # Save data to a standardized filename
        raw_filename = "raw_data.csv"
        raw_filepath = os.path.join(DATA_DIR, raw_filename)
        try:
            # Explicitly set separator to PIPE, ensure index and header are written
            raw_data.to_csv(raw_filepath, sep='|', index=True, header=True)
            logging.info(f"Raw data ({settings.TRAIN_START_DATE} to {end_date_to_use}) saved successfully to {raw_filepath} with '|' separator")
        except Exception as e:
            logging.error(f"Failed to save raw data: {e}")
    else:
        logging.warning("No raw data was fetched or it was empty.")

    # Removed the separate fetching and saving of test data

    logging.info("--- Data Fetching Script Finished ---")
