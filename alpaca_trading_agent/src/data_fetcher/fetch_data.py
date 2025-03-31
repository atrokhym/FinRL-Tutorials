# alpaca_trading_agent/src/data_fetcher/fetch_data.py

import alpaca_trade_api as tradeapi
import pandas as pd
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration Loading ---
# Correctly navigate up two levels from src/data_fetcher to the project root
# then access config and data directories.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_DIR = os.path.join(PROJECT_ROOT, 'config')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# Add project root to sys.path to allow absolute imports from config
import sys
sys.path.insert(0, PROJECT_ROOT) # Add project root instead of just config

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
        secret_key=api_keys.ALPACA_SECRET_KEY,
        base_url=api_keys.ALPACA_BASE_URL,
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

    # Reset index to have 'date' and 'tic' as columns if needed by FinRL preprocessing
    # combined_df = combined_df.reset_index().rename(columns={'index': 'date'})
    # combined_df = combined_df.set_index(['date', 'tic']) # Or keep date as index, tic as column

    # Standardize column names (FinRL often expects lowercase)
    combined_df.columns = [col.lower() for col in combined_df.columns]

    logging.info(f"Finished fetching data. Shape: {combined_df.shape}")
    return combined_df

# --- Main Execution ---
if __name__ == "__main__":
    logging.info("--- Starting Data Fetching Script ---")

    # Fetch training data
    train_data = fetch_historical_data(
        tickers=settings.TICKERS,
        start_date=settings.TRAIN_START_DATE,
        end_date=settings.TRAIN_END_DATE,
        timeframe=settings.TIME_INTERVAL # Corrected variable name
    )

    if train_data is not None and not train_data.empty:
        # Save training data
        train_filename = f"train_data_{settings.TRAIN_START_DATE}_{settings.TRAIN_END_DATE}.csv"
        train_filepath = os.path.join(DATA_DIR, train_filename)
        try:
            train_data.to_csv(train_filepath)
            logging.info(f"Training data saved successfully to {train_filepath}")
        except Exception as e:
            logging.error(f"Failed to save training data: {e}")
    else:
        logging.warning("No training data was fetched or it was empty.")

    # Fetch testing data (used for backtesting the trained model)
    test_data = fetch_historical_data(
        tickers=settings.TICKERS,
        start_date=settings.TEST_START_DATE,
        end_date=settings.TEST_END_DATE,
        timeframe=settings.TIME_INTERVAL # Corrected variable name
    )

    if test_data is not None and not test_data.empty:
        # Save testing data
        test_filename = f"test_data_{settings.TEST_START_DATE}_{settings.TEST_END_DATE}.csv"
        test_filepath = os.path.join(DATA_DIR, test_filename)
        try:
            test_data.to_csv(test_filepath)
            logging.info(f"Testing data saved successfully to {test_filepath}")
        except Exception as e:
            logging.error(f"Failed to save testing data: {e}")
    else:
        logging.warning("No testing data was fetched or it was empty.")

    logging.info("--- Data Fetching Script Finished ---")
