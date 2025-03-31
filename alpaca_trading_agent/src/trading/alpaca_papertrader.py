# alpaca_trading_agent/src/trading/alpaca_papertrader.py

import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import os
import sys
import logging
import time
from datetime import datetime, timedelta
import schedule # For scheduling the trading logic
from stable_baselines3 import PPO

# --- Configuration and Environment Loading ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_DIR = os.path.join(PROJECT_ROOT, 'config')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')

# Add relevant directories to sys.path
sys.path.insert(0, CONFIG_DIR)
sys.path.insert(0, os.path.join(SRC_DIR, 'preprocessing')) # To import preprocess_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(RESULTS_DIR, "papertrader.log")), # Log to file
        logging.StreamHandler() # Also log to console
    ]
)

try:
    import settings
    from preprocess_data import preprocess_data # Import our preprocessing function
except ImportError as e:
    logging.error(f"Error importing configuration or preprocessing function: {e}.")
    logging.error("Ensure config/settings.py and src/preprocessing/preprocess_data.py exist.")
    sys.exit(1)

# --- Constants ---
# Lookback period needed for indicators (adjust based on longest indicator requirement, e.g., MACD needs ~30-40 periods)
# Should be slightly larger than the longest period used in settings.INDICATORS
INDICATOR_LOOKBACK_DAYS = 60 # Needs to be sufficient for indicators like MACD, RSI etc.
MODEL_ALGO = "PPO" # Must match the trained model
TOTAL_TIMESTEPS = 20000 # Must match the trained model
MODEL_FILENAME_BASE = f"{MODEL_ALGO}_{settings.TIME_INTERVAL}_{TOTAL_TIMESTEPS}"
MODEL_PATH = os.path.join(RESULTS_DIR, 'models', f"{MODEL_FILENAME_BASE}.zip")
TRADE_SCHEDULE_TIME = "15:45" # Time to run the trading logic (e.g., 15 mins before market close EST for daily)

# --- Alpaca API Connection ---
def connect_alpaca():
    """Establishes connection to the Alpaca API."""
    logging.info("Connecting to Alpaca API...")
    try:
        api = tradeapi.REST(
            key_id=settings.ALPACA_API_KEY,
            secret_key=settings.ALPACA_SECRET_KEY,
            base_url=settings.ALPACA_BASE_URL, # Ensure this points to paper trading URL
            api_version='v2'
        )
        account = api.get_account()
        logging.info(f"Connected successfully. Account Status: {account.status}")
        if settings.ALPACA_BASE_URL != 'https://paper-api.alpaca.markets':
             logging.warning("Connected to LIVE Alpaca API. Ensure this is intended.")
        return api
    except Exception as e:
        logging.error(f"Failed to connect to Alpaca: {e}", exc_info=True)
        return None

# --- Data Fetching ---
def fetch_latest_data(api, tickers, timeframe, lookback_days):
    """Fetches the latest market data bars for the required lookback."""
    logging.info(f"Fetching latest data for {len(tickers)} tickers, lookback {lookback_days} days.")
    # Calculate start and end dates for the lookback period
    # Alpaca API expects ISO format, end date is exclusive for get_bars
    end_dt = datetime.now(tradeapi.Timezone) # Use Alpaca's timezone awareness
    start_dt = end_dt - timedelta(days=lookback_days)

    # Format for Alpaca API (adjust if needed based on API version/docs)
    start_iso = start_dt.isoformat()
    end_iso = end_dt.isoformat()

    try:
        # Use get_bars for historical data retrieval
        # Note: Adjust parameters based on exact API requirements (e.g., limit, adjustment)
        barset = api.get_bars(
            tickers,
            timeframe,
            start=start_iso,
            end=end_iso,
            adjustment='raw' # Or 'split', 'dividend' as needed
        ).df
        logging.info(f"Fetched {len(barset)} bars.")

        # Format the dataframe: lowercase columns, add 'tic', ensure datetime index
        barset.index = pd.to_datetime(barset.index) # Ensure index is datetime
        barset.rename(columns=str.lower, inplace=True) # Lowercase columns
        # 'symbol' column usually contains the ticker in Alpaca's response
        if 'symbol' in barset.columns:
            barset['tic'] = barset['symbol']
        else:
            logging.error("Could not find 'symbol' column in fetched data.")
            return None

        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume', 'tic']
        if not all(col in barset.columns for col in required_cols):
            logging.error(f"Fetched data missing required columns. Found: {barset.columns.tolist()}")
            return None

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
    # Ensure data is sorted by date, tic before preprocessing if needed by indicator logic
    # fetch_latest_data should return it sorted by time, but maybe not tic within time
    raw_data = raw_data.sort_index() # Sort by time first
    # Preprocessing expects data grouped by ticker for stockstats
    processed_data_list = []
    for tic in raw_data['tic'].unique():
        tic_data = raw_data[raw_data['tic'] == tic].copy()
        processed_tic_data = preprocess_data(tic_data) # Call the imported function
        if processed_tic_data is not None:
            processed_data_list.append(processed_tic_data)

    if not processed_data_list:
        logging.error("Preprocessing failed for all tickers. Skipping cycle.")
        return

    processed_data = pd.concat(processed_data_list).sort_values(by=['date', 'tic']).reset_index(drop=True)
    logging.info(f"Preprocessing complete. Processed data shape: {processed_data.shape}")

    # Check if we have data for the most recent expected date (or close to it)
    latest_data_date = processed_data['date'].max()
    logging.info(f"Latest date in processed data: {latest_data_date}")
    # Add check if latest_data_date is too old?

    # 3. Prepare Observation State for the *latest* timestep
    # This needs to precisely match the structure expected by the environment/model
    latest_state_df = processed_data[processed_data['date'] == latest_data_date]
    if latest_state_df.empty or len(latest_state_df) != settings.STOCK_DIM:
        logging.error(f"Could not get complete state data for latest date {latest_data_date}. Found {len(latest_state_df)} tickers.")
        return

    # Get current cash balance and holdings from Alpaca
    try:
        account = api.get_account()
        cash_balance = float(account.cash) # Use non-marginable cash? Or equity? Check API docs
        positions = api.list_positions()
        holdings = {pos.symbol: int(pos.qty) for pos in positions}
        logging.info(f"Current Cash: {cash_balance:.2f}, Holdings: {holdings}")
    except Exception as e:
        logging.error(f"Failed to get account/position info: {e}", exc_info=True)
        return

    # Construct the state array [balance] + [shares] + [features]
    current_shares = [holdings.get(tic, 0) for tic in settings.TICKERS]
    # Ensure features are ordered correctly by ticker as in settings.TICKERS
    latest_state_df = latest_state_df.set_index('tic').loc[settings.TICKERS].reset_index()
    feature_values = latest_state_df[settings.INDICATORS].values.flatten().tolist()

    state = np.array(
        [cash_balance] + current_shares + feature_values,
        dtype=np.float32
    )

    # Verify state dimension
    expected_len = 1 + settings.STOCK_DIM + settings.STOCK_DIM * len(settings.INDICATORS)
    if len(state) != expected_len:
        logging.error(f"Constructed state length mismatch! Expected {expected_len}, got {len(state)}")
        return

    logging.info(f"State constructed for prediction. Shape: {state.shape}")

    # 4. Predict Action
    try:
        action, _states = model.predict(state, deterministic=True)
        logging.info(f"Predicted Action: {action}")
    except Exception as e:
        logging.error(f"Error during model prediction: {e}", exc_info=True)
        return

    # 5. Translate Action to Orders (Implement this logic)
    # Example: Convert action (proportions) to target shares/dollar values
    # Compare target to current holdings and generate buy/sell orders
    # This part requires careful implementation based on how the agent was trained
    # 5. Translate Action to Orders
    # Assuming action represents desired change in shares, scaled by MAX_STOCK_POSITION
    # The environment scales actions by hmax (default 100), let's use settings.MAX_STOCK_POSITION
    actions_scaled = action * settings.MAX_STOCK_POSITION # Scale actions (e.g., -1 to 1 -> -100 to 100 shares change)
    actions_intended_shares = actions_scaled.astype(int) # Target change in shares

    orders_to_submit = []
    try:
        # Get latest prices for order validation/logging (optional, market orders don't strictly need it)
        latest_quotes = api.get_latest_quotes(settings.TICKERS)
        current_prices = {tic: q.ap for tic, q in latest_quotes.items()} # Use ask price for estimate
    except Exception as e:
        logging.warning(f"Could not get latest quotes, proceeding without price estimates: {e}")
        current_prices = {}


    logging.info("Calculating desired trades based on action...")
    for i, tic in enumerate(settings.TICKERS):
        intended_change = actions_intended_shares[i]
        current_holding = holdings.get(tic, 0)
        current_price_estimate = current_prices.get(tic, "N/A")

        if intended_change > 0: # Buy signal
            qty_to_buy = intended_change
            logging.info(f"ORDER PREP: Buy {qty_to_buy} shares of {tic} at market (Current Holding: {current_holding}, Est Price: {current_price_estimate}).")
            orders_to_submit.append({
                "symbol": tic,
                "qty": qty_to_buy,
                "side": "buy",
                "type": "market",
                "time_in_force": "day" # Good Till Day
            })
        elif intended_change < 0: # Sell signal
            qty_to_sell = abs(intended_change)
            # Ensure we don't sell more than we hold
            qty_to_sell = min(qty_to_sell, current_holding)
            if qty_to_sell > 0:
                logging.info(f"ORDER PREP: Sell {qty_to_sell} shares of {tic} at market (Current Holding: {current_holding}, Est Price: {current_price_estimate}).")
                orders_to_submit.append({
                    "symbol": tic,
                    "qty": qty_to_sell,
                    "side": "sell",
                    "type": "market",
                    "time_in_force": "day"
                })
        # else: intended_change is 0, do nothing for this ticker

    # 6. Submit Orders
    if not orders_to_submit:
        logging.info("No orders to submit for this cycle.")
    else:
        logging.info(f"Submitting {len(orders_to_submit)} orders...")
        for order_data in orders_to_submit:
            try:
                logging.info(f"Submitting order: {order_data}")
                submitted_order = api.submit_order(**order_data)
                logging.info(f"Order submitted for {order_data['symbol']}. Order ID: {submitted_order.id}, Status: {submitted_order.status}")
                time.sleep(0.1) # Small delay between orders to avoid rate limits
            except tradeapi.rest.APIError as e:
                # Log specific Alpaca errors
                logging.error(f"Alpaca API Error submitting order for {order_data['symbol']}: {e}")
                logging.error(f"Failed order details: {order_data}")
            except Exception as e:
                logging.error(f"Unexpected error submitting order for {order_data['symbol']}: {e}", exc_info=True)

    logging.info("--- Trading Logic Cycle Finished ---")


# --- Main Execution ---
if __name__ == "__main__":
    logging.info("--- Starting Alpaca Paper Trading Agent ---")

    # Check for API keys
    if not settings.ALPACA_API_KEY or not settings.ALPACA_SECRET_KEY:
        logging.error("Alpaca API Key/Secret not found in environment variables.")
        logging.error("Please set ALPACA_API_KEY and ALPACA_SECRET_KEY.")
        sys.exit(1)

    # Connect to Alpaca
    alpaca_api = connect_alpaca()
    if alpaca_api is None:
        sys.exit(1)

    # Load the trained model
    logging.info(f"Loading trained model from: {MODEL_PATH}")
    try:
        trading_model = PPO.load(MODEL_PATH, env=None) # Load model structure and weights
        logging.info("Model loaded successfully.")
    except FileNotFoundError:
        logging.error(f"Trained model file not found: {MODEL_PATH}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading trained model: {e}", exc_info=True)
        sys.exit(1)

    # --- Scheduling ---
    logging.info(f"Scheduling trading logic to run daily at {TRADE_SCHEDULE_TIME} EST/EDT.")
    # Schedule the job
    schedule.every().day.at(TRADE_SCHEDULE_TIME, "America/New_York").do(run_trading_logic, api=alpaca_api, model=trading_model)

    # Run once immediately? Optional
    # logging.info("Running initial trading logic cycle...")
    # run_trading_logic(api=alpaca_api, model=trading_model)
    # logging.info("Initial cycle finished.")

    # Keep the script running to execute scheduled jobs
    logging.info("Scheduler started. Waiting for scheduled jobs...")
    while True:
        schedule.run_pending()
        time.sleep(60) # Check every minute

    # Note: Need to handle graceful shutdown (e.g., Ctrl+C) if running continuously.
    # logging.info("--- Alpaca Paper Trading Agent Finished ---") # This line might not be reached in loop
