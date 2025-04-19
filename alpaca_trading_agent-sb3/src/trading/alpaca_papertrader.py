# alpaca_trading_agent/src/trading/alpaca_papertrader.py

import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import os
import sys
import logging
import time
from datetime import datetime, timedelta
import pytz  # Import pytz for timezone handling
import schedule  # For scheduling the trading logic
import math  # For floor function

# Note: argparse is imported later after potentially configuring logging
from stable_baselines3 import PPO

# --- Configuration and Environment Loading ---
# Assume environment (e.g., PYTHONPATH) is set correctly by the caller (main.py)
# Define project structure related constants based on this file's location
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

# Add Project Root and SRC directory to sys.path
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)  # Allows 'import settings'
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)  # Allows 'from utils...' and 'from preprocessing...'

# Attempt direct imports, relying on the Python path
try:
    from data_fetcher.vix_utils import get_vix_data  # Import VIX fetching utility
    from utils.logging_setup import configure_file_logging, add_console_logging
    from preprocessing.preprocess_data import preprocess_data
    import settings  # Assuming config is also on the path or PYTHONPATH
    import argparse  # Import argparse here
except ImportError as e:
    # Basic logging configuration if imports fail immediately
    logging.basicConfig(
        level=logging.ERROR,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logging.error(f"Failed to import necessary modules at the start: {e}")
    logging.error(
        "Ensure PYTHONPATH includes the project root and src directories, or run from the project root."
    )

    # Define dummy functions to prevent NameErrors later if import fails
    def configure_file_logging(level):
        pass

    def add_console_logging(level):
        pass

    # Do not exit here yet, let the main block handle final checks


# --- Constants ---
# Define constants after settings have been successfully imported
try:
    INDICATOR_LOOKBACK_DAYS = (
        60  # Needs to be sufficient for indicators like MACD, RSI etc.
    )
    MODEL_ALGO = "PPO"  # Must match the trained model
    # TOTAL_TIMESTEPS = 20000 # Must match the trained model
    MODEL_FILENAME_BASE = (
        f"{MODEL_ALGO}_{settings.TIME_INTERVAL}_{settings.TOTAL_TIMESTEPS}"
    )
    MODEL_PATH = os.path.join(RESULTS_DIR, "models", f"{MODEL_FILENAME_BASE}.zip")
    TRADE_SCHEDULE_TIME = "15:45"  # Time to run the trading logic (e.g., 15 mins before market close EST for daily)
    # Use specific log file name from settings, with a default
    LOG_FILE_NAME = getattr(settings, "TRADE_LOG_FILE", "trading_agent_trade.log")
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
            base_url=settings.ALPACA_API_BASE_URL,  # Ensure this points to paper trading URL
            api_version="v2",
        )
        account = api.get_account()
        logging.info(f"Connected successfully. Account Status: {account.status}")
        logging.info(
            f"Account Number: {account.account_number}"
        )  # Added line to log account number
        if settings.ALPACA_API_BASE_URL != "https://paper-api.alpaca.markets":
            logging.warning("Connected to LIVE Alpaca API. Ensure this is intended.")
        return api
    except Exception as e:
        logging.error(f"Failed to connect to Alpaca: {e}", exc_info=True)
        return None


# --- Data Fetching ---
def fetch_latest_data(api, tickers, timeframe, lookback_days):
    """Fetches the latest market data bars for the required lookback."""
    logging.info(
        f"Fetching latest data for {len(tickers)} tickers, lookback {lookback_days} days."
    )
    end_dt = datetime.now(pytz.timezone("America/New_York"))
    start_dt = end_dt - timedelta(days=lookback_days + 5)  # Add buffer
    start_iso = start_dt.isoformat()
    end_iso = end_dt.isoformat()
    start_date_str = start_dt.strftime("%Y-%m-%d")  # For vix_utils
    end_date_str = end_dt.strftime("%Y-%m-%d")  # For vix_utils

    try:
        # Fetch data for primary tickers
        logging.info(f"Fetching data for primary tickers: {tickers}")
        ticker_bars = api.get_bars(
            tickers,
            timeframe,
            start=start_iso,
            end=end_iso,
            adjustment="all",
            feed="iex",
        ).df
        logging.info(f"Fetched {len(ticker_bars)} bars for primary tickers.")
        if ticker_bars.empty:
            logging.warning("Fetched empty barset for primary tickers.")
            return pd.DataFrame()

        # --- Fetch VIX data ---
        logging.info("Fetching VIX data using vix_utils...")
        vix_df = get_vix_data(
            start_date_str, end_date_str
        )  # Returns df with tz-naive 'date' index
        if vix_df.empty:
            logging.warning(
                "get_vix_data returned empty DataFrame. Proceeding without VIX."
            )
            vix_df_reset = pd.DataFrame(
                columns=["date", "vix"]
            )  # Ensure it has 'date' for merge
        else:
            logging.info(f"Successfully fetched VIX data. Shape: {vix_df.shape}")
            vix_df_reset = vix_df.reset_index()  # 'date' becomes a column

        # --- Process Primary Ticker Data ---
        # Ensure index is DatetimeIndex and UTC
        if not isinstance(ticker_bars.index, pd.DatetimeIndex):
            ticker_bars.index = pd.to_datetime(ticker_bars.index)
        if ticker_bars.index.tz is None:
            ticker_bars = ticker_bars.tz_localize("UTC")
        else:
            ticker_bars = ticker_bars.tz_convert("UTC")

        # Create a tz-naive 'date' column (date part only) for merging
        ticker_bars["date"] = ticker_bars.index.normalize()
        ticker_bars["date"] = ticker_bars["date"].dt.tz_localize(
            None
        )  # Make date column naive

        # Reset index AFTER creating the 'date' column
        ticker_bars = ticker_bars.reset_index(
            drop=True
        )  # Drop the original timestamp index

        ticker_bars.rename(columns=str.lower, inplace=True)
        if "symbol" in ticker_bars.columns:
            ticker_bars["tic"] = ticker_bars["symbol"]
        else:
            logging.error("Could not find 'symbol' column.")
            return None

        required_cols = ["open", "high", "low", "close", "volume", "tic", "date"]
        if not all(col in ticker_bars.columns for col in required_cols):
            logging.error(
                f"Primary ticker data missing required columns. Found: {ticker_bars.columns.tolist()}"
            )
            return None

        # --- Merge VIX ---
        logging.info("Merging VIX data using pd.merge on date...")
        # Merge on the 'date' column (both should be tz-naive date objects)
        merged_bars = pd.merge(
            ticker_bars, vix_df_reset[["date", "vix"]], on="date", how="left"
        )
        logging.info(f"Shape after merge: {merged_bars.shape}")

        # Fill VIX NaNs
        if "vix" not in merged_bars.columns:  # If VIX fetch failed completely
            merged_bars["vix"] = 0.0
            logging.warning("Added 'vix' column with zeros as VIX fetch failed.")
        else:
            # Fill NaNs resulting from merge or original NaNs in VIX data
            merged_bars["vix"] = merged_bars.groupby("tic")["vix"].ffill().bfill()
            logging.info("VIX ffilled and bfilled.")
            # Drop rows if VIX is still NaN after filling
            initial_rows = len(merged_bars)
            merged_bars.dropna(subset=["vix"], inplace=True)
            if len(merged_bars) < initial_rows:
                logging.warning(
                    f"Dropped {initial_rows - len(merged_bars)} rows with unfillable VIX."
                )
            logging.info("VIX merge and fill complete.")

        # Final Checks
        if merged_bars.empty:
            logging.error("DataFrame became empty after VIX merge/dropna.")
            return pd.DataFrame()

        final_required_cols = required_cols + ["vix"]
        if not all(col in merged_bars.columns for col in final_required_cols):
            logging.error(
                f"Final DataFrame missing required columns. Found: {merged_bars.columns.tolist()}. Expected: {final_required_cols}"
            )
            return None

        # Ensure 'date' column is datetime
        merged_bars["date"] = pd.to_datetime(merged_bars["date"])

        logging.info(f"Final data shape after VIX processing: {merged_bars.shape}")
        return merged_bars

    except Exception as e:
        logging.error(f"Error in fetch_latest_data: {e}", exc_info=True)
        return None


# --- Trading Logic ---
def run_trading_logic(api, model):
    """Executes one cycle of the trading logic."""
    logging.info("--- Running Trading Logic Cycle ---")

    # 1. Fetch Latest Data
    raw_data = fetch_latest_data(
        api, settings.TICKERS, settings.TIME_INTERVAL, INDICATOR_LOOKBACK_DAYS
    )
    if raw_data is None or raw_data.empty:
        logging.error("Failed to fetch or received empty data. Skipping cycle.")
        return

    # 2. Preprocess Data
    # Preprocess expects datetime index. Set 'date' column as index.
    if "date" not in raw_data.columns:
        logging.error("Raw data missing 'date' column before preprocessing.")
        return
    try:
        raw_data_indexed = raw_data.set_index("date")
    except KeyError:
        logging.error("'date' column not found for setting index during preprocessing.")
        return

    # Pass the indexed DataFrame to preprocessing
    # Set is_live_trading=True to get only the latest row features
    processed_data = preprocess_data(raw_data_indexed, is_live_trading=True)

    if processed_data is None or processed_data.empty:
        logging.error("Preprocessing failed or returned empty data. Skipping cycle.")
        return

    logging.info(
        f"Preprocessing complete. Shape of latest data: {processed_data.shape}"
    )
    logging.debug(f"Latest processed data columns: {processed_data.columns.tolist()}")
    # Latest date is implicitly the index/date column of the single row df
    # latest_data_date = processed_data['date'].iloc[0] # Get date from the row
    # logging.info(f"Latest date in processed data: {latest_data_date}")

    # 3. Prepare Observation State
    # processed_data should contain one row per ticker for the latest timestamp
    if len(processed_data["tic"].unique()) != settings.STOCK_DIM:
        logging.error(
            f"Could not get complete state data for latest timestamp. Found {len(processed_data['tic'].unique())} tickers vs expected {settings.STOCK_DIM}."
        )
        return

    try:
        account = api.get_account()
        cash_balance = float(account.cash)
        portfolio_value = float(account.portfolio_value)
        buying_power = float(account.buying_power)
        logging.info(
            f"Account Status - Portfolio Value: {portfolio_value:.2f}, Cash: {cash_balance:.2f}, Buying Power: {buying_power:.2f}"
        )
        positions = api.list_positions()
        holdings = {pos.symbol: int(pos.qty) for pos in positions}
        logging.info(f"Current Holdings: {holdings}")
    except Exception as e:
        logging.error(f"Failed to get account/position info: {e}", exc_info=True)
        return

    current_shares = [holdings.get(tic, 0) for tic in settings.TICKERS]

    # Reorder the processed_data (which has one row per ticker) according to settings.TICKERS
    latest_state_df = (
        processed_data.set_index("tic").reindex(settings.TICKERS).reset_index()
    )

    # Check for NaN values after reindexing (in case a ticker had no data)
    if latest_state_df.isnull().values.any():
        missing_tics = latest_state_df[latest_state_df.isnull().any(axis=1)][
            "tic"
        ].tolist()
        logging.error(
            f"Missing data for tickers in the latest state after reindex: {missing_tics}. Check data fetch/preprocess."
        )
        # Decide how to handle: skip cycle, fill NaNs? For now, skip.
        return

    try:
        # Use INDICATORS_WITH_TURBULENCE list from settings
        feature_values = (
            latest_state_df[settings.INDICATORS_WITH_TURBULENCE]
            .values.flatten()
            .tolist()
        )
    except KeyError as e:
        logging.error(
            f"Missing expected indicator column in latest data: {e}. Available: {latest_state_df.columns.tolist()}"
        )
        return

    state = np.array([cash_balance] + current_shares + feature_values, dtype=np.float32)

    expected_len = settings.STATE_SPACE
    if len(state) != expected_len:
        logging.error(
            f"State length mismatch! Expected {expected_len}, got {len(state)}"
        )
        logging.error(
            f"Cash: 1, Shares: {settings.STOCK_DIM}, Features: {settings.STOCK_DIM * len(settings.INDICATORS_WITH_TURBULENCE)}"
        )
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
    # orders_to_submit = [] # Replaced by logic below

    try:
        latest_quotes = api.get_latest_quotes(settings.TICKERS)
        # Use ask price (ap) for buy cost estimate, bid price (bp) for sell value estimate (though not strictly needed for market sell)
        current_ask_prices = {tic: q.ap for tic, q in latest_quotes.items()}
    except Exception as e:
        logging.warning(
            f"Could not get latest quotes: {e}. Will attempt fallback using latest close price for estimates."
        )
        current_ask_prices = {}  # Initialize empty, will populate with fallbacks later

    logging.info("Calculating desired trades based on action...")
    # Use latest close prices from state df as a fallback if quotes are missing
    latest_close_prices = latest_state_df.set_index("tic")["close"].to_dict()

    # Separate buys and sells intentions
    buy_order_requests = []  # Store as {'symbol': tic, 'qty': qty, 'price_est': price}
    sell_order_requests = []  # Store as {'symbol': tic, 'qty': qty}

    for i, tic in enumerate(settings.TICKERS):
        intended_change = actions_intended_shares[i]
        current_holding = holdings.get(tic, 0)

        # Get price estimate for buys: prefer latest ask, fallback to latest close
        price_estimate = current_ask_prices.get(tic)
        if price_estimate is None:
            price_estimate = latest_close_prices.get(tic)
            if price_estimate:
                logging.debug(
                    f"Using latest close price ({price_estimate:.2f}) as buy estimate for {tic} (quote unavailable)."
                )
            else:
                logging.warning(
                    f"Could not get price estimate for {tic}. Skipping trade check for this ticker."
                )
                continue  # Skip if no price available

        if intended_change > 0:  # Intend to Buy
            qty_to_buy = intended_change
            logging.info(
                f"TRADE PREP (Buy Intention): {qty_to_buy} {tic} @ market (Holding: {current_holding}, Est Price: {price_estimate:.2f})"
            )
            # Ensure price_estimate is a float before adding
            if isinstance(price_estimate, (int, float)):
                buy_order_requests.append(
                    {
                        "symbol": tic,
                        "qty": qty_to_buy,
                        "price_est": float(price_estimate),
                    }
                )
            else:
                logging.warning(
                    f"Invalid price estimate type ({type(price_estimate)}) for {tic}. Skipping buy intention."
                )

        elif intended_change < 0:  # Intend to Sell
            qty_to_sell = min(
                abs(intended_change), current_holding
            )  # Can only sell what you have
            if qty_to_sell > 0:
                logging.info(
                    f"TRADE PREP (Sell): {qty_to_sell} {tic} @ market (Holding: {current_holding})"
                )
                sell_order_requests.append(
                    {"symbol": tic, "qty": int(qty_to_sell)}
                )  # Ensure int qty

    # --- Buying Power Check and Buy Order Preparation ---
    total_buy_cost = sum(req["qty"] * req["price_est"] for req in buy_order_requests)
    logging.info(
        f"Estimated total cost for buy intentions: {total_buy_cost:.2f}. Available cash: {cash_balance:.2f}"
    )

    # Scale down buy orders if necessary (use 99% of cash for safety margin)
    # Use buying_power for the check, applying a small safety margin (e.g., 99%)
    buying_power_limit = buying_power * 0.99
    final_buy_orders = []
    # Compare total cost against the buying power limit
    if total_buy_cost > buying_power_limit and total_buy_cost > 0:
        scale_factor = buying_power_limit / total_buy_cost
        logging.warning(
            f"Insufficient buying power ({buying_power:.2f}) for estimated buy cost ({total_buy_cost:.2f}). Scaling buy orders by factor {scale_factor:.4f}."
        )
        for req in buy_order_requests:
            scaled_qty = math.floor(
                req["qty"] * scale_factor
            )  # Use floor to ensure integer qty <= available cash
            if scaled_qty > 0:
                logging.info(
                    f"SCALED BUY ORDER: {scaled_qty} {req['symbol']} (Original: {req['qty']})"
                )
                final_buy_orders.append(
                    {
                        "symbol": req["symbol"],
                        "qty": int(scaled_qty),
                        "side": "buy",
                        "type": "market",
                        "time_in_force": "day",
                    }
                )
            else:
                logging.info(
                    f"Skipping buy for {req['symbol']} after scaling resulted in zero quantity."
                )
    else:
        # No scaling needed, prepare original buy orders
        logging.info("Sufficient buying power for all intended buy orders.")
        for req in buy_order_requests:
            final_buy_orders.append(
                {
                    "symbol": req["symbol"],
                    "qty": int(req["qty"]),
                    "side": "buy",
                    "type": "market",
                    "time_in_force": "day",
                }
            )

    # --- Prepare final sell orders ---
    final_sell_orders = [
        {
            "symbol": req["symbol"],
            "qty": req["qty"],
            "side": "sell",
            "type": "market",
            "time_in_force": "day",
        }
        for req in sell_order_requests
    ]

    # Combine sell and (potentially scaled) buy orders
    # Consider submitting sells first to potentially free up cash, though market orders execute quickly
    orders_to_submit = final_sell_orders + final_buy_orders

    # 6. Submit Orders
    if not orders_to_submit:
        logging.info("No orders to submit for this cycle.")
    else:
        logging.info(f"Submitting {len(orders_to_submit)} orders...")
        for order_data in orders_to_submit:
            try:
                logging.info(f"Submitting order: {order_data}")
                submitted_order = api.submit_order(**order_data)
                logging.info(
                    f"Order submitted for {order_data['symbol']}. ID: {submitted_order.id}, Status: {submitted_order.status}"
                )
                time.sleep(0.1)  # Small delay between submissions
            except tradeapi.rest.APIError as e:
                # Log specific Alpaca errors
                logging.error(
                    f"Alpaca API Error submitting order for {order_data['symbol']}: {e} - Status Code: {e.status_code} - Order Data: {order_data}"
                )
                # You might want to check e.status_code for specific handling (e.g., 403 insufficient funds)
            except Exception as e:
                logging.error(
                    f"Unexpected error submitting order for {order_data['symbol']}: {e}",
                    exc_info=True,
                )

    logging.info("--- Trading Logic Cycle Finished ---")


# --- Main Execution ---
if __name__ == "__main__":
    # --- Configure Argument Parser ---
    # Need to parse args *before* configuring logging if log level depends on args
    parser = argparse.ArgumentParser(description="Alpaca Trading Agent - Paper Trader")
    parser.add_argument(
        "--log-level",
        default="INFO",  # Default logging level
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level for the paper trading script.",
    )
    # Parse only known arguments specific to this script
    # Use try-except in case argparse wasn't imported successfully
    try:
        args, _ = parser.parse_known_args()
        log_level_arg = args.log_level
    except NameError:
        log_level_arg = "INFO"  # Fallback if argparse failed
        logging.error("Argparse module not loaded, using default log level 'INFO'.")

    # --- Configure Logging ---
    # Configure file-based logging using the argument or default
    # Use try-except in case logging utils weren't imported
    try:
        # Pass log_level as a keyword argument
        configure_file_logging(log_level_arg)
        add_console_logging(log_level_arg)  # ADDED THIS CALL
        logging.info(
            f"--- Paper Trading Logging Initialized (Level: {log_level_arg.upper()}) ---"
        )
    except Exception as e:
        # Use basic print if logging setup itself fails
        print(f"ERROR setting up logging: {e}", file=sys.stderr)
        # Fallback basic config to console if setup fails
        logging.basicConfig(
            level=log_level_arg.upper(),
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        logging.error(f"Failed to configure logging via utility: {e}", exc_info=True)
        # sys.exit(1) # Optionally exit

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
    logging.info(
        f"Scheduling trading logic to run daily at {TRADE_SCHEDULE_TIME} America/New_York."
    )
    try:
        schedule.every().day.at(TRADE_SCHEDULE_TIME, "America/New_York").do(
            run_trading_logic, api=alpaca_api, model=trading_model
        )
    except Exception as e:
        logging.error(f"Failed to schedule trading logic: {e}", exc_info=True)
        sys.exit(1)

    # # Optional: Run once immediately
    # logging.info("Running initial trading logic cycle...")
    # run_trading_logic(api=alpaca_api, model=trading_model)
    # logging.info("Initial cycle finished.")

    # Keep the script running
    logging.info("Scheduler started. Waiting for scheduled jobs...")
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every 60 seconds
    except KeyboardInterrupt:
        logging.info("Shutdown signal received.")
    except Exception as e:
        logging.error(f"Scheduler loop encountered an error: {e}", exc_info=True)
    finally:
        logging.info("--- Alpaca Paper Trading Agent Shutdown ---")
