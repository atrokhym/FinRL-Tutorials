# alpaca_trading_agent/src/data_fetcher/fetch_data.py

import alpaca_trade_api as tradeapi
import pandas as pd
import os
from datetime import datetime, timedelta  # Added timedelta
import logging
import argparse  # Added argparse
import sys  # Added sys
# Removed yfinance import here as it's now handled by vix_utils primarily
import time  # Added time for sleep

# Logging setup will be done explicitly using the shared utility

# --- Configuration Loading ---
# Correctly navigate up two levels from src/data_fetcher to the project root
# then access config and data directories.
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Add project root to sys.path to allow absolute imports from config
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)  # Add project root if not already there

# Import the logging setup utility
try:
    # Import both functions now
    from src.utils.logging_setup import configure_file_logging, add_console_logging
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


# Import settings and api_keys
try:
    from config import settings
    from config import api_keys  # Use absolute import from config package
except ImportError as e:
    logging.error(
        f"Error importing configuration: {e}. Ensure config/settings.py and config/api_keys.py exist and PROJECT_ROOT is added to sys.path."
    )
    sys.exit(1)

# Import the VIX utility function - USE ABSOLUTE IMPORT based on PROJECT_ROOT in sys.path
try:
    from src.data_fetcher.vix_utils import get_vix_data
except ImportError:
    logging.error("Failed to import get_vix_data from src.data_fetcher.vix_utils. VIX data will not be added.")
    # Define as None so checks later don't cause NameError
    get_vix_data = None

# --- Argument Parsing for Logging ---
# This script needs to parse the log level passed from the main orchestrator
parser = argparse.ArgumentParser(description="Data Fetcher Script")
parser.add_argument(
    "--log-level",
    default="INFO",
    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    help="Set the logging level for this script.",
)
# Add argument for train_end_date, mirroring main.py logic if needed here
parser.add_argument(
    "--train-end-date",
    type=str,
    default=None,
    help="Optional end date for training data override (YYYY-MM-DD).",
)
# Parse only known args relevant to this script, ignore others
args, unknown = parser.parse_known_args()


# --- Logging setup is now done within the __main__ block ---


# Create data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)


# --- Data Fetching Function ---
# Note: Alpaca API initialization moved inside if __name__ == "__main__"
def fetch_historical_data(api_instance, tickers, start_date, end_date, timeframe="1Day", limit=10000):
    """
    Fetches historical OHLCV data for given tickers from Alpaca.

    Args:
        api_instance: Initialized Alpaca tradeapi.REST instance.
        tickers (list): List of stock ticker symbols.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        timeframe (str): Alpaca API timeframe ('1Min', '5Min', '15Min', '1H', '1Day').
        limit (int): The maximum number of bars to fetch per request. Alpaca's limit is 10000 for v2.

    Returns:
        pd.DataFrame: DataFrame containing combined OHLCV data for all tickers,
                      indexed by date, or None if fetching fails.
    """
    all_data = []
    # Alpaca API expects ISO format timestamps
    start_dt = pd.Timestamp(start_date, tz="America/New_York").isoformat()
    end_dt = pd.Timestamp(end_date, tz="America/New_York").isoformat()

    logging.info(
        f"Fetching data for {tickers} from {start_date} to {end_date} ({timeframe})"
    )

    for ticker in tickers:
        logging.info(f"Fetching data for {ticker}...")
        try:
            # Use get_bars for fetching historical data
            barset = api_instance.get_bars(
                ticker,
                timeframe,
                start=start_dt,
                end=end_dt,
                adjustment="all",  # Use 'all' for split and dividend adjustments
                limit=limit,
            ).df
            barset["tic"] = ticker
            all_data.append(barset)
            logging.info(f"Successfully fetched {len(barset)} bars for {ticker}")
            # Add a small delay to avoid hitting rate limits aggressively
            time.sleep(0.1)
        except Exception as e:
            logging.error(
                f"CRITICAL ERROR fetching data for {ticker}: {e}", exc_info=True
            )
            # Depending on the error, decide whether to continue or stop
            # For now, continue to try fetching other tickers
            # return None # Uncomment to stop on first ticker error

    if not all_data:
        logging.warning("No data fetched for any ticker.")
        return None

    # Combine dataframes
    combined_df = pd.concat(all_data)

    # Ensure datetime index is timezone-aware (Alpaca returns UTC)
    if not combined_df.index.tz:
        combined_df = combined_df.tz_localize("UTC")

    # Convert to market time (usually Eastern Time) for easier analysis
    combined_df.index = combined_df.index.tz_convert("America/New_York")

    # Keep only the date part if using daily data, converting to DatetimeIndex first
    if timeframe == "1Day":
         # Ensure index is DatetimeIndex before accessing .date
         if isinstance(combined_df.index, pd.DatetimeIndex):
              # Make the index timezone-naive DATE object first, then convert back to datetime
              combined_df.index = pd.to_datetime(combined_df.index.date)
         else: # If already converted to object/date somehow, ensure it's datetime and naive
              combined_df.index = pd.to_datetime(combined_df.index).tz_localize(None)

    combined_df.index.name = "date"  # Explicitly name the index

    # Standardize column names (FinRL often expects lowercase)
    combined_df.columns = [col.lower() for col in combined_df.columns]

    logging.info(f"Finished fetching stock data. Shape: {combined_df.shape}")
    return combined_df


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Data Fetcher for Alpaca Trading Agent"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level for the script.",
    )
    parser.add_argument(
        "--train-end-date",
        type=str,
        default=None,
        help="Specify the end date for fetching data in YYYY-MM-DD format. Overrides settings.py TRAIN_END_DATE/TEST_END_DATE.",
    )
    args = parser.parse_args()

    # --- Configure Logging ---
    # Call BOTH functions here to set up file and console logging
    log_level_to_use = args.log_level.upper() # Store log level
    try:
        configure_file_logging(log_level_to_use)
        add_console_logging(log_level_to_use) # ADDED THIS CALL
        logging.info(f"--- Data Fetcher Logging Initialized (Level: {log_level_to_use}) ---")
    except Exception as e:
        # Use basic print if logging setup itself fails
        print(f"ERROR setting up logging: {e}", file=sys.stderr)
        # Fallback basic config to console if setup fails
        logging.basicConfig(
            level=log_level_to_use, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        logging.error(f"Failed to configure logging via utility: {e}", exc_info=True)
        # Decide if you want to exit if logging fails
        # sys.exit(1)

    # Determine the end date to use
    # Default to TEST_END_DATE from settings to ensure all necessary data is fetched
    end_date_to_use = settings.TEST_END_DATE
    if args.train_end_date:
        # If override is provided, check if it's earlier than the default
        override_dt = pd.to_datetime(args.train_end_date)
        default_end_dt = pd.to_datetime(settings.TEST_END_DATE)
        if override_dt < default_end_dt:
            end_date_to_use = args.train_end_date
            logging.info(f"Using command-line override end date: {end_date_to_use}")
        else:
            logging.warning(f"Command-line end date ({args.train_end_date}) is not earlier than TEST_END_DATE ({settings.TEST_END_DATE}). Using TEST_END_DATE.")
            end_date_to_use = settings.TEST_END_DATE # Keep using the later date
    else:
        logging.info(
            f"Using latest required end date from settings (TEST_END_DATE): {end_date_to_use}"
        )

    # --- Alpaca API Initialization ---
    api = None
    try:
        logging.debug("Initializing Alpaca API connection...")
        api = tradeapi.REST(
            key_id=api_keys.ALPACA_API_KEY,
            secret_key=api_keys.ALPACA_API_SECRET,
            base_url=api_keys.ALPACA_API_BASE_URL,
            api_version="v2",
        )
        account = api.get_account()
        logging.info(f"Connected to Alpaca. Account status: {account.status}")
    except Exception as e:
        logging.error(f"Failed to connect to Alpaca: {e}", exc_info=True)
        logging.error(
            "Please ensure API keys in config/api_keys.py are correct and Alpaca service is running."
        )
        sys.exit(1)

    # --- Fetch Stock Data ---
    raw_data = None
    try:
        raw_data = fetch_historical_data(
            api_instance=api,
            tickers=settings.TICKERS,
            start_date=settings.TRAIN_START_DATE,
            end_date=end_date_to_use,
            timeframe=settings.TIME_INTERVAL,
        )
    except Exception as e:
        logging.error(
            f"Error occurred during fetch_historical_data call: {e}",
            exc_info=True,
        )
        sys.exit(1)

    # Proceed only if stock data was fetched successfully
    if raw_data is not None and not raw_data.empty:

        # --- Fetch and Merge VIX Data using Utility ---
        if get_vix_data: # Check if the import was successful
            logging.info(f"Fetching/Loading VIX data...")
            try:
                # Use the utility function (vix_utils ensures index is tz-naive 'date')
                vix_df = get_vix_data(settings.TRAIN_START_DATE, end_date_to_use)

                if not vix_df.empty:
                    logging.info(f"Successfully retrieved VIX data. Shape: {vix_df.shape}")

                    # --- Prepare for Column Merge ---
                    # 1. Reset raw_data index to make 'date' a column
                    raw_data_reset = raw_data.reset_index()
                    # Ensure raw_data_reset['date'] is datetime and timezone-naive
                    raw_data_reset['date'] = pd.to_datetime(raw_data_reset['date']).dt.tz_localize(None)

                    # 2. Reset vix_df index and FLATTEN MultiIndex columns
                    vix_df_reset = vix_df.reset_index()
                    # Flatten potential MultiIndex columns (e.g., [('Date', ''), ('vix', '^VIX')])
                    if isinstance(vix_df_reset.columns, pd.MultiIndex):
                        logging.debug("Flattening VIX DataFrame columns")
                        # Correctly flatten: join tuple elements (as strings), strip underscores
                        vix_df_reset.columns = ['_'.join(map(str,col)).strip('_') if isinstance(col, tuple) else col for col in vix_df_reset.columns.values]
                        logging.debug(f"Columns after flattening: {vix_df_reset.columns.tolist()}") # Log flattened columns

                    # Rename columns consistently to 'date' and 'vix' AFTER flattening
                    if 'Date' in vix_df_reset.columns: # Handle potential 'Date' from cache/download
                         vix_df_reset = vix_df_reset.rename(columns={'Date': 'date'})
                    elif 'date_' in vix_df_reset.columns: # Handle potential 'date_' after flattening
                         vix_df_reset = vix_df_reset.rename(columns={'date_': 'date'})

                    if 'Close' in vix_df_reset.columns: # Handle case where 'Close' might still be the name
                         vix_df_reset = vix_df_reset.rename(columns={'Close': 'vix'})
                    elif 'vix_^VIX' in vix_df_reset.columns: # Handle flattened name
                         vix_df_reset = vix_df_reset.rename(columns={'vix_^VIX': 'vix'})

                    # Ensure vix_df_reset['date'] is datetime and timezone-naive
                    vix_df_reset['date'] = pd.to_datetime(vix_df_reset['date']).dt.tz_localize(None)

                    # --- DEBUGGING ---
                    if log_level_to_use == 'DEBUG':
                        logging.debug("--- Pre-Merge raw_data_reset (tz-naive date) ---")
                        logging.debug(f"Columns: {raw_data_reset.columns.tolist()}")
                        logging.debug(f"Date dtype: {raw_data_reset['date'].dtype}")
                        logging.debug(f"Head:\n{raw_data_reset.head()}")
                        logging.debug("--- Pre-Merge vix_df_reset (Flattened, tz-naive date) ---")
                        logging.debug(f"Columns: {vix_df_reset.columns.tolist()}")
                        logging.debug(f"Date dtype: {vix_df_reset['date'].dtype}")
                        logging.debug(f"Head:\n{vix_df_reset.head()}")
                    # --- END DEBUGGING ---

                    # --- Perform Column Merge on Timezone-Naive Dates ---
                    # Ensure 'vix' column exists before selecting it
                    if 'vix' not in vix_df_reset.columns:
                        logging.error(f"'vix' column not found in vix_df_reset after flattening/renaming. Columns: {vix_df_reset.columns.tolist()}")
                        raise KeyError("'vix' column missing from VIX data")
                    raw_data = pd.merge(raw_data_reset, vix_df_reset[['date', 'vix']], on='date', how='left')
                    logging.info(f"Shape after merging VIX: {raw_data.shape}")

                    # Forward fill VIX after merging
                    raw_data['vix'] = raw_data['vix'].ffill()
                    logging.info("Forward-filled missing VIX values after merge.")

                    # Backfill remaining NaNs (usually at the start)
                    if raw_data['vix'].isnull().any():
                        logging.warning("NaN values remain in VIX column after ffill. Back-filling first valid value.")
                        raw_data['vix'] = raw_data['vix'].bfill()
                        if raw_data['vix'].isnull().any():
                            logging.error("NaN values STILL remain in VIX column after bfill. Filling with 0.")
                            raw_data['vix'].fillna(0, inplace=True) # Final fallback

                    # --- Set Final Multi-Index ---
                    if 'tic' in raw_data.columns and 'date' in raw_data.columns:
                         raw_data = raw_data.set_index(['date', 'tic']).sort_index()
                         logging.info("Set final index to ['date', 'tic']")
                    else:
                         logging.error("Could not set ['date', 'tic'] index. 'date' or 'tic' column missing after merge.")
                         # Attempt to set index just to date if possible
                         if 'date' in raw_data.columns:
                             raw_data = raw_data.set_index('date').sort_index()
                             logging.warning("Set index to ['date'] as fallback.")
                         else:
                             logging.error("Cannot set index, 'date' column is missing.")


                else:
                    logging.error("Failed to retrieve VIX data using get_vix_data (returned empty DataFrame).")
                    sys.exit(1) # Exit if VIX is critical

            except Exception as e:
                logging.error(
                    f"An error occurred during VIX retrieval or merging: {e}", exc_info=True
                )
                sys.exit(1) # Exit on VIX errors
        else:
             logging.warning("get_vix_data function not available. Skipping VIX data integration.")
        # --- End VIX Fetch and Merge ---

        # --- Save Combined Data ---
        raw_filename = "raw_data.csv"
        raw_filepath = os.path.join(DATA_DIR, raw_filename)
        try:
            # Reset index before saving to include 'date' and 'tic' as columns
            raw_data_to_save = raw_data.reset_index()

            logging.debug(f"Columns before saving raw_data.csv: {raw_data_to_save.columns.tolist()}")

            # Ensure essential columns exist
            required_cols = ['date', 'tic', 'vix'] # Add other essential OHLCV cols if needed
            missing_cols = [col for col in required_cols if col not in raw_data_to_save.columns]
            if missing_cols:
                logging.error(f"Missing essential columns before saving: {missing_cols}. Data saving aborted.")
                sys.exit(1)


            # Save with standard CSV format, index=False because date/tic are columns now
            raw_data_to_save.to_csv(raw_filepath, index=False)
            logging.info(
                f"Combined raw data saved successfully to {raw_filepath}"
            )
        except Exception as e:
            logging.error(f"Failed to save combined raw data: {e}", exc_info=True)
            sys.exit(1) # Exit if saving fails
    else:
        logging.warning("No raw stock data was fetched or it was empty. Cannot proceed.")
        sys.exit(1) # Exit if stock data fetching failed initially

    logging.info("--- Data Fetching Script Finished ---")
