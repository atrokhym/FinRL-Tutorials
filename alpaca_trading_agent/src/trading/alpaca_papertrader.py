# alpaca_trading_agent/src/trading/alpaca_papertrader.py

import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import os
import sys
import logging
import time
from datetime import datetime, timedelta
import pytz # Import pytz for timezone handling
import schedule # For scheduling the trading logic
# Note: argparse is imported later after potentially configuring logging
from stable_baselines3 import PPO

# --- Configuration and Environment Loading ---
# Assume environment (e.g., PYTHONPATH) is set correctly by the caller (main.py)
# Define project structure related constants based on this file's location
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_DIR = os.path.join(PROJECT_ROOT, 'config')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')

# Add Project Root and SRC directory to sys.path
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT) # Allows 'import settings'
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)     # Allows 'from utils...' and 'from preprocessing...'

# Attempt direct imports, relying on the Python path
try:
    from utils.logging_setup import configure_file_logging
    from preprocessing.preprocess_data import preprocess_data
    import settings # Assuming config is also on the path or PYTHONPATH
    import argparse # Import argparse here
except ImportError as e:
    # Basic logging configuration if imports fail immediately
    logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.error(f"Failed to import necessary modules at the start: {e}")
    logging.error("Ensure PYTHONPATH includes the project root and src directories, or run from the project root.")
    sys.exit(1) # Exit if essential modules can't be imported

# --- Constants ---
# Define constants after settings have been successfully imported
try:
    INDICATOR_LOOKBACK_DAYS = 60 # Needs to be sufficient for indicators like MACD, RSI etc.
    MODEL_ALGO = "PPO" # Must match the trained model
    #TOTAL_TIMESTEPS = 20000 # Must match the trained model
    MODEL_FILENAME_BASE = f"{MODEL_ALGO}_{settings.TIME_INTERVAL}_{settings.TOTAL_TIMESTEPS}"
    MODEL_PATH = os.path.join(RESULTS_DIR, 'models', f"{MODEL_FILENAME_BASE}.zip")
    TRADE_SCHEDULE_TIME = "15:45" # Time to run the trading logic (e.g., 15 mins before market close EST for daily)
    # Use specific log file name from settings, with a default
    LOG_FILE_NAME = getattr(settings, 'TRADE_LOG_FILE', 'trading_agent_trade.log')
except NameError:
    # Handle case where settings import failed but script continued (should not happen due to sys.exit)
    logging.error("Settings module not loaded correctly. Cannot define constants.")
    sys.exit(1)
except AttributeError as e:
    logging.error(f"Missing expected attribute in settings: {e}")
    sys.exit(1)


# --- Alpaca API Connection ---
def connect_alpaca():
    """Establishes connection to the Alpaca API."""
    logging.info("Connecting to Alpaca API...")
    try:
        api = tradeapi.REST(
            key_id=settings.ALPACA_API_KEY,
            secret_key=settings.ALPACA_API_SECRET,
            base_url=settings.ALPACA_API_BASE_URL, # Ensure this points to paper trading URL
            api_version='v2'
        )
        account = api.get_account()
        logging.info(f"Connected successfully. Account Status: {account.status}")
        if settings.ALPACA_API_BASE_URL != 'https://paper-api.alpaca.markets':
             logging.warning("Connected to LIVE Alpaca API. Ensure this is intended.")
        return api
    except Exception as e:
        logging.error(f"Failed to connect to Alpaca: {e}", exc_info=True)
        return None

# --- Data Fetching ---
def fetch_latest_data(api, tickers, timeframe, lookback_days):
    """Fetches the latest market data bars for the required lookback."""
    logging.info(f"Fetching latest data for {len(tickers)} tickers, lookback {lookback_days} days.")
    end_dt = datetime.now(pytz.timezone('America/New_York'))
    start_dt = end_dt - timedelta(days=lookback_days)
    start_iso = start_dt.isoformat()
    end_iso = end_dt.isoformat()

    try:
        barset = api.get_bars(
            tickers,
            timeframe,
            start=start_iso,
            end=end_iso,
            adjustment='raw',
            feed='iex'
        ).df
        logging.info(f"Fetched {len(barset)} bars.")

        if barset.empty:
             logging.warning("Fetched empty barset from Alpaca.")
             return pd.DataFrame() # Return empty DataFrame

        barset.index = pd.to_datetime(barset.index)
        barset.rename(columns=str.lower, inplace=True)
        if 'symbol' in barset.columns:
            barset['tic'] = barset['symbol']
        else:
            logging.error("Could not find 'symbol' column in fetched data.")
            return None

        required_cols = ['open', 'high', 'low', 'close', 'volume', 'tic']
        if not all(col in barset.columns for col in required_cols):
            logging.error(f"Fetched data missing required columns. Found: {barset.columns.tolist()}")
            return None

        barset.reset_index(inplace=True)
        if 'timestamp' in barset.columns:
             barset.rename(columns={'timestamp': 'date'}, inplace=True)
        elif 'index' in barset.columns:
             barset.rename(columns={'index': 'date'}, inplace=True)
        else:
             if 'date' not in barset.columns:
                 logging.warning("Could not find index column ('timestamp' or 'index') after reset_index to rename to 'date'.")

        return barset

    except Exception as e:
        logging.error(f"Error fetching latest data from Alpaca: {e}", exc_info=True)
        return None

# --- Trading Logic ---
def run_trading_logic(api, model):
    """Executes one cycle of the trading logic."""
    logging.info("--- Running Trading Logic Cycle ---")

    # 1. Fetch Latest Data
    raw_data = fetch_latest_data(api, settings.TICKERS, settings.TIME_INTERVAL, INDICATOR_LOOKBACK_DAYS)
    if raw_data is None or raw_data.empty:
        logging.error("Failed to fetch or received empty data. Skipping cycle.")
        return

    # 2. Preprocess Data
    processed_data_list = []
    for tic in raw_data['tic'].unique():
        tic_data = raw_data[raw_data['tic'] == tic].copy()
        # Ensure tic_data is sorted by date before passing to preprocess
        tic_data = tic_data.sort_values(by='date')
        processed_tic_data = preprocess_data(tic_data)
        if processed_tic_data is not None:
            processed_data_list.append(processed_tic_data)

    if not processed_data_list:
        logging.error("Preprocessing failed for all tickers. Skipping cycle.")
        return

    processed_data = pd.concat(processed_data_list).sort_values(by=['date', 'tic']).reset_index(drop=True)
    logging.info(f"Preprocessing complete. Processed data shape: {processed_data.shape}")

    latest_data_date = processed_data['date'].max()
    logging.info(f"Latest date in processed data: {latest_data_date}")

    # 3. Prepare Observation State
    latest_state_df = processed_data[processed_data['date'] == latest_data_date]
    if latest_state_df.empty or len(latest_state_df) != settings.STOCK_DIM:
        logging.error(f"Could not get complete state data for latest date {latest_data_date}. Found {len(latest_state_df)} tickers vs expected {settings.STOCK_DIM}.")
        return

    try:
        account = api.get_account()
        cash_balance = float(account.cash)
        positions = api.list_positions()
        holdings = {pos.symbol: int(pos.qty) for pos in positions}
        logging.info(f"Current Cash: {cash_balance:.2f}, Holdings: {holdings}")
    except Exception as e:
        logging.error(f"Failed to get account/position info: {e}", exc_info=True)
        return

    current_shares = [holdings.get(tic, 0) for tic in settings.TICKERS]
    latest_state_df = latest_state_df.set_index('tic').reindex(settings.TICKERS).reset_index()

    if latest_state_df.isnull().values.any():
        missing_tics = latest_state_df[latest_state_df.isnull().any(axis=1)]['tic'].tolist()
        logging.error(f"Missing data for tickers in the latest state: {missing_tics}")
        return

    try:
        feature_values = latest_state_df[settings.INDICATORS_WITH_TURBULENCE].values.flatten().tolist()
    except KeyError as e:
        logging.error(f"Missing expected indicator column in latest data: {e}. Available: {latest_state_df.columns.tolist()}")
        return

    state = np.array(
        [cash_balance] + current_shares + feature_values,
        dtype=np.float32
    )

    expected_len = settings.STATE_SPACE
    if len(state) != expected_len:
        logging.error(f"State length mismatch! Expected {expected_len}, got {len(state)}")
        return

    logging.info(f"State constructed for prediction. Shape: {state.shape}")

    # 4. Predict Action
    try:
        action, _states = model.predict(state, deterministic=True)
        logging.info(f"Predicted Action: {action}")
    except Exception as e:
        logging.error(f"Error during model prediction: {e}", exc_info=True)
        return

    # 5. Translate Action to Orders
    actions_scaled = action * settings.MAX_STOCK_POSITION
    actions_intended_shares = actions_scaled.astype(int)
    orders_to_submit = []

    try:
        latest_quotes = api.get_latest_quotes(settings.TICKERS)
        current_prices = {tic: q.ap for tic, q in latest_quotes.items()}
    except Exception as e:
        logging.warning(f"Could not get latest quotes: {e}")
        current_prices = {}

    logging.info("Calculating desired trades based on action...")
    for i, tic in enumerate(settings.TICKERS):
        intended_change = actions_intended_shares[i]
        current_holding = holdings.get(tic, 0)
        current_price_estimate = current_prices.get(tic, "N/A")

        if intended_change > 0: # Buy
            qty_to_buy = intended_change
            logging.info(f"ORDER PREP: Buy {qty_to_buy} {tic} @ market (Holding: {current_holding}, Est Price: {current_price_estimate})")
            orders_to_submit.append({"symbol": tic, "qty": int(qty_to_buy), "side": "buy", "type": "market", "time_in_force": "day"})
        elif intended_change < 0: # Sell
            qty_to_sell = min(abs(intended_change), current_holding)
            if qty_to_sell > 0:
                logging.info(f"ORDER PREP: Sell {qty_to_sell} {tic} @ market (Holding: {current_holding}, Est Price: {current_price_estimate})")
                orders_to_submit.append({"symbol": tic, "qty": int(qty_to_sell), "side": "sell", "type": "market", "time_in_force": "day"})

    # 6. Submit Orders
    if not orders_to_submit:
        logging.info("No orders to submit for this cycle.")
    else:
        logging.info(f"Submitting {len(orders_to_submit)} orders...")
        for order_data in orders_to_submit:
            try:
                logging.info(f"Submitting order: {order_data}")
                submitted_order = api.submit_order(**order_data)
                logging.info(f"Order submitted for {order_data['symbol']}. ID: {submitted_order.id}, Status: {submitted_order.status}")
                time.sleep(0.1)
            except tradeapi.rest.APIError as e:
                logging.error(f"Alpaca API Error submitting order for {order_data['symbol']}: {e} - Details: {order_data}")
            except Exception as e:
                logging.error(f"Unexpected error submitting order for {order_data['symbol']}: {e}", exc_info=True)

    logging.info("--- Trading Logic Cycle Finished ---")

# --- Main Execution ---
if __name__ == "__main__":
    # --- Configure Argument Parser ---
    # Need to parse args *before* configuring logging if log level depends on args
    parser = argparse.ArgumentParser(description="Alpaca Trading Agent - Paper Trader")
    parser.add_argument(
        "--log-level",
        default="INFO", # Default logging level
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level for the paper trading script.",
    )
    # Parse only known arguments specific to this script
    # Use try-except in case argparse wasn't imported successfully
    try:
        args, _ = parser.parse_known_args()
        log_level_arg = args.log_level
    except NameError:
        log_level_arg = "INFO" # Fallback if argparse failed
        logging.error("Argparse module not loaded, using default log level 'INFO'.")

    # --- Configure Logging ---
    # Configure file-based logging using the argument or default
    # Use try-except in case configure_file_logging wasn't imported
    try:
        # Pass log_level as a keyword argument
        configure_file_logging(log_level_arg)
    except NameError:
        # Fallback to basic logging if imports failed earlier
        # This level might be ERROR if basicConfig was called in the initial import try-except
        logging.error("Logging setup utility not loaded. Using basic config.")
        # Ensure level is set based on argument even with basicConfig
        logging.getLogger().setLevel(log_level_arg.upper())


    logging.info("--- Starting Alpaca Paper Trading Agent ---")

    # Check for API keys (ensure settings was imported)
    try:
        if not settings.ALPACA_API_KEY or not settings.ALPACA_API_SECRET:
            logging.error("Alpaca API Key/Secret not found in settings.")
            sys.exit(1)
    except NameError:
         logging.error("Settings module not loaded, cannot check API keys.")
         sys.exit(1)

    # Connect to Alpaca
    alpaca_api = connect_alpaca()
    if alpaca_api is None:
        logging.critical("Failed to connect to Alpaca API. Exiting.")
        sys.exit(1)

    # Load the trained model
    logging.info(f"Loading trained model from: {MODEL_PATH}")
    try:
        # Ensure MODEL_PATH is defined
        if not os.path.exists(MODEL_PATH):
             logging.error(f"Model file does not exist at path: {MODEL_PATH}")
             sys.exit(1)
        trading_model = PPO.load(MODEL_PATH, env=None)
        logging.info("Model loaded successfully.")
    except NameError:
         logging.error("MODEL_PATH not defined, likely due to settings loading error.")
         sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading trained model: {e}", exc_info=True)
        sys.exit(1)

    # --- Scheduling ---
    logging.info(f"Scheduling trading logic to run daily at {TRADE_SCHEDULE_TIME} America/New_York.")
    try:
        schedule.every().day.at(TRADE_SCHEDULE_TIME, "America/New_York").do(run_trading_logic, api=alpaca_api, model=trading_model)
    except Exception as e:
        logging.error(f"Failed to schedule trading logic: {e}", exc_info=True)
        sys.exit(1)

    # Optional: Run once immediately
    # logging.info("Running initial trading logic cycle...")
    # run_trading_logic(api=alpaca_api, model=trading_model)
    # logging.info("Initial cycle finished.")

    # Keep the script running
    logging.info("Scheduler started. Waiting for scheduled jobs...")
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)
    except KeyboardInterrupt:
        logging.info("Shutdown signal received.")
    except Exception as e:
        logging.error(f"Scheduler loop encountered an error: {e}", exc_info=True)
    finally:
        logging.info("--- Alpaca Paper Trading Agent Shutdown ---")
