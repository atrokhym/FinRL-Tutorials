#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates  # Add for date formatting
import datetime
import matplotlib  # Add base matplotlib import
import argparse
import os
import gym
from numpy import random as rd
import pickle
import hashlib
import shutil  # Added for directory removal
from sqlalchemy import create_engine  # Added for database connection

import torch
import torch.serialization  # Added for safe_globals
import torch.nn as nn  # Added for Sequential class
from torch.distributions.normal import Normal  # Added for safe_globals
import ray
from torch.utils.tensorboard import (
    SummaryWriter,
)  # Added import for TensorBoard logging

import random

# Assuming Stable Baselines3 uses PyTorch by default
#import torch

# --- Seed for Reproducibility ---
SEED = 42  # You can change this value
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
# If using CUDA (GPU)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # if use multi-GPU
    # These settings can enforce determinism but might impact performance
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
# Set PYTHONHASHSEED environment variable (optional, affects hash randomization)
# Note: This needs to be set *before* Python starts for full effect,
# so setting it here might only have partial impact depending on execution context.
# Consider setting it in the shell environment if strict hash reproducibility is needed.
# import os
# os.environ['PYTHONHASHSEED'] = str(SEED)

print(f"Global random seeds set to: {SEED}")



# Attempt to import the specific class needed for safe loading
try:
    from elegantrl.agents.net import ActorPPO
except ImportError:
    print(
        "Warning: Could not import ActorPPO from elegantrl.agents.net. Model loading might fail if this specific class is required."
    )

    # Define a placeholder if import fails, though loading will likely still error
    # if the actual class definition is needed by torch.load
    class ActorPPO:
        pass


# from finrl import config # Assuming config might be needed later, commented out for now
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split

# from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv # Original env
# Using NP version as in original script
from local_stocktrading_env import StockTradingEnv  # Use local copy
from finrl.agents.stablebaselines3.models import DRLAgent as DRLAgent_sb3
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback # Added BaseCallback
from finrl.agents.rllib.models import DRLAgent as DRLAgent_rllib
from finrl.agents.elegantrl.models import DRLAgent as DRLAgent_erl

from finrl.meta.data_processor import DataProcessor
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline

# from finrl.config_tickers import DOW_30_TICKER # Import DOW tickers

# import plotly.graph_objs as go # Removed Plotly


# --- Constants ---
RAW_DATA_CACHE_DIR = "./data_cache/raw"
PROCESSED_DATA_CACHE_DIR = "./data_cache/processed"

# # Default Dates (can be overridden by command-line args)
# TRAIN_START_DATE = "2014-01-01"
# TRAIN_END_DATE = "2020-12-31"
# # TEST_START_DATE = "2021-01-01"
# # TEST_END_DATE = "2021-12-31"
# TEST_START_DATE = "2023-01-01"  # Unchanged
# TEST_END_DATE = "2024-12-31"  # Adjusted to avoid potential download issues

TRAIN_START_DATE = '2015-01-01'
TRAIN_END_DATE = '2021-12-31'
VALIDATION_START_DATE = '2022-01-01'
VALIDATION_END_DATE = '2022-12-31'
TEST_START_DATE = '2023-01-01'
TEST_END_DATE = '2025-04-15' 


# Default Indicators (can be overridden by command-line args)
INDICATORS = [
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "dx_30",
    "close_30_sma",
    "close_60_sma",
]


# Default Tickers (can be overridden by command-line args)
TECH_TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "NVDA",
    "TSLA",
    "INTC",
    "CSCO",
    "ADBE",
]
# Use DOW_30_TICKER for eRL example as in original script, TECH_TICKERS for others
DEFAULT_TICKER_LIST = TECH_TICKERS

# Default DRL Params (can be overridden by command-line args)
ERL_PARAMS = {
    "learning_rate": 3e-5,
    "batch_size": 2048,
    "gamma": 0.985,
    "seed": 312,
    "net_dimension": 512,
    "target_step": 5000,
    "eval_gap": 60,
    "eval_times": 1,
}
RLlib_PARAMS = {"lr": 5e-6, "train_batch_size": 1000, "gamma": 0.99}
SAC_PARAMS = {
    "batch_size": 128,
    "buffer_size": 100000,
    "learning_rate": 0.0001,
    "learning_starts": 100,
    "ent_coef": "auto_0.1",
}
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.00025,
    "batch_size": 128,
}
TD3_PARAMS = {"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001}
DDPG_PARAMS = {"batch_size": 128, "buffer_size": 50000, "learning_rate": 0.001}
A2C_PARAMS = {"n_steps": 5, "ent_coef": 0.01, "learning_rate": 3e-5}

# Map model names to their default params
AGENT_PARAMS_MAP = {
    "sac": SAC_PARAMS,
    # Note: eRL and RLlib also use 'ppo', specific params handled in run_train
    "ppo": PPO_PARAMS,
    "td3": TD3_PARAMS,
    "ddpg": DDPG_PARAMS,
    "a2c": A2C_PARAMS,
}

# --- Stock Trading Environment Definition (Copied from original script) ---
# It might be better to import this if it's stable in the library,
# but copying ensures it matches the version used in the notebook.


# Removed local StockTradingEnv definition (lines 134-298)
# Now using: from local_stocktrading_env import StockTradingEnv

# Removed leftover methods (get_state, sigmoid_sign) from deleted local class definition


# --- Data Functions ---


# Import necessary libraries for database access
from sqlalchemy import create_engine
import pandas as pd  # Ensure pandas is imported if not already

# --- Constants for DB connection ---
DB_CONNECTION_STRING = "postgresql://postgres:postgres@localhost:5432/postgres"
RAW_DATA_TABLE = "all_raw_data"
# ---


def get_data(
    start_date,
    end_date,
    ticker_list,
    data_source,  # Keep for DataProcessor instantiation, though not used for fetching
    time_interval,  # Keep for potential future use, though not used for fetching
    cache_dir=RAW_DATA_CACHE_DIR,  # No longer used
    **kwargs,
):
    """Fetches and cleans data from the PostgreSQL database."""
    print(
        f"Fetching data from PostgreSQL table '{RAW_DATA_TABLE}' for {len(ticker_list)} tickers..."
    )
    print(f"Date range: {start_date} to {end_date}")

    data = pd.DataFrame()  # Initialize empty dataframe

    try:
        engine = create_engine(DB_CONNECTION_STRING)
        # Construct the SQL query
        # Ensure column names match your table structure
        # Assuming timestamp column is compatible with string comparison
        query = f"""
        SELECT timestamp, tic, open, high, low, close, volume
        FROM {RAW_DATA_TABLE}
        WHERE tic IN ({str(ticker_list)[1:-1]}) -- Creates 'AAPL', 'MSFT', ...
        AND timestamp >= '{start_date}'
        AND timestamp <= '{end_date}'
        ORDER BY tic, timestamp;
        """
        # print(f"Executing query: {query}") # Optional: Debug query
        data = pd.read_sql(query, engine)

        if data.empty:
            print(
                "Warning: No data returned from database for the specified tickers/date range."
            )
        else:
            print(
                f"Successfully fetched {len(data)} rows for {len(data['tic'].unique())} tickers from database."
            )

            # Convert timestamp column to datetime objects (important for cleaning/processing)
            # Assuming the timestamp column is stored as text or compatible type
            try:
                # Attempt conversion, trying 'ISO8601' format first
                data["timestamp"] = pd.to_datetime(
                    data["timestamp"], format="ISO8601", errors="coerce"
                )
                # If conversion failed for some, try inferring format
                if data["timestamp"].isnull().any():
                    print(
                        "Warning: ISO8601 conversion failed for some timestamps, trying mixed format."
                    )
                    # Coerce errors to NaT (Not a Time)
                    data["timestamp"] = pd.to_datetime(
                        data["timestamp"], format="mixed", errors="coerce"
                    )
                # Drop rows where timestamp conversion failed completely
                original_rows = len(data)
                data.dropna(subset=["timestamp"], inplace=True)
                if len(data) < original_rows:
                    print(
                        f"Warning: Dropped {original_rows - len(data)} rows due to unparseable timestamps."
                    )
                # Optional: Localize or convert timezone if necessary, e.g., data['timestamp'] = data['timestamp'].dt.tz_localize('UTC')
                print("Timestamp column converted to datetime.")
            except Exception as e:
                print(f"Error converting 'timestamp' column to datetime: {e}")
                # Return empty df if conversion fails critically
                raise ValueError("Timestamp conversion failed") from e

            # Clean the data (using DataProcessor's clean_data method)
            # We still instantiate DP to use its cleaning logic

    except Exception as e:
        print(f"Error fetching or processing data from database: {e}")
        data = pd.DataFrame()  # Ensure empty dataframe on error

    # Remove caching logic
    # print(f"Raw data saved/loaded from cache is disabled.")

    return data


# --- VIX Data Function ---
VIX_TABLE_NAME = "vix_raw_data"  # Define VIX table name constant


def get_vix_from_db(start_date, end_date):
    """Fetches VIX data from the PostgreSQL database."""
    print(f"Fetching VIX data from PostgreSQL table '{VIX_TABLE_NAME}'...")
    print(f"Date range: {start_date} to {end_date}")

    vix_data = pd.DataFrame()
    try:
        engine = create_engine(DB_CONNECTION_STRING)
        # Query specifically for VIX ticker '^VIX'
        query = f"""
        SELECT timestamp, tic, open, high, low, close, volume
        FROM {VIX_TABLE_NAME}
        WHERE tic = '^VIX'
        AND timestamp >= '{start_date}'
        AND timestamp <= '{end_date}'
        ORDER BY timestamp;
        """
        vix_data = pd.read_sql(query, engine)

        if vix_data.empty:
            print(
                "Warning: No VIX data returned from database for the specified date range."
            )
        else:
            print(f"Successfully fetched {len(vix_data)} rows for VIX from database.")
            # Convert timestamp column (similar to get_data)
            try:
                vix_data["timestamp"] = pd.to_datetime(
                    vix_data["timestamp"], format="ISO8601", errors="coerce"
                )
                if vix_data["timestamp"].isnull().any():
                    print(
                        "Warning: ISO8601 conversion failed for some VIX timestamps, trying mixed format."
                    )
                    vix_data["timestamp"] = pd.to_datetime(
                        vix_data["timestamp"], format="mixed", errors="coerce"
                    )
                original_rows = len(vix_data)
                vix_data.dropna(subset=["timestamp"], inplace=True)
                if len(vix_data) < original_rows:
                    print(
                        f"Warning: Dropped {original_rows - len(vix_data)} VIX rows due to unparseable timestamps."
                    )
                print("VIX timestamp column converted to datetime.")
            except Exception as e:
                print(f"Error converting VIX 'timestamp' column to datetime: {e}")
                raise ValueError("VIX timestamp conversion failed") from e
    except Exception as e:
        print(f"Error fetching VIX data from database: {e}")
        vix_data = pd.DataFrame()  # Ensure empty dataframe on error

    return vix_data


# --- End VIX Data Function ---


def process_data(
    data,
    technical_indicator_list,
    start_date,
    end_date,
    if_vix=True,
    cache_dir=PROCESSED_DATA_CACHE_DIR,
    raw_data_hash=None,
    **kwargs,
):
    """Adds technical indicators, VIX, and converts to arrays, using CSV cache if available."""
    os.makedirs(cache_dir, exist_ok=True)

    # Create a unique filename for the cache
    indicator_str = "_".join(sorted(technical_indicator_list))
    vix_str = "vix" if if_vix else "novix"
    if raw_data_hash is None:
        try:
            data_to_hash = data.sort_index(axis=1)
            raw_data_hash = hashlib.md5(
                pd.util.hash_pandas_object(data_to_hash, index=True).values
            ).hexdigest()
        except Exception as hash_err:
            print(f"Error generating hash for raw data: {hash_err}")
            raw_data_hash = f"unknown_raw_data_{rd.randint(10000)}"

    filename_base = f"{raw_data_hash}_{indicator_str}_{vix_str}_processed"
    filename_hash = hashlib.md5(filename_base.encode()).hexdigest()
    cache_file = os.path.join(cache_dir, f"{filename_hash}.csv") # Use .csv extension

    loaded_from_cache = False
    processed_data = None
    vix_successfully_added = False # Initialize vix status

    # Attempt to load from CSV cache
    if os.path.exists(cache_file):
        print(f"Attempting to load processed data from CSV cache: {cache_file}")
        try:
            processed_data = pd.read_csv(cache_file)
            if 'timestamp' in processed_data.columns:
                try:
                    processed_data['timestamp'] = pd.to_datetime(processed_data['timestamp'], errors='coerce', utc=True)
                    processed_data.dropna(subset=['timestamp'], inplace=True)
                    print("Timestamp column loaded and converted to datetime (UTC).")
                except Exception as ts_err:
                    print(f"Warning: Error converting timestamp after loading CSV: {ts_err}. Re-processing.")
                    processed_data = None
            else:
                print("Warning: 'timestamp' column not found in cached CSV. Re-processing.")
                processed_data = None

            if processed_data is not None and not processed_data.empty:
                print("Processed DataFrame loaded successfully from CSV cache.")
                loaded_from_cache = True
                # Check if VIX was likely added based on columns
                vix_successfully_added = ('vix' in processed_data.columns or 'VIXY' in processed_data.columns)
                if vix_successfully_added:
                    print("VIX column found in cached data.")
            else:
                print("CSV cache loading failed or resulted in empty data. Re-processing.")
                loaded_from_cache = False

        except Exception as e:
            print(f"Error loading processed data from CSV cache {cache_file}: {e}. Re-processing.")
            loaded_from_cache = False
            processed_data = None

    # --- Data Processing Pipeline (if not loaded from cache) ---
    if not loaded_from_cache:
        print("Processing data: adding indicators...")
        processed_data = data.copy()

        # --- VIX Merging ---
        if if_vix:
            print("Attempting to add VIX data from database...")
            vix_df = get_vix_from_db(start_date, end_date)
            if vix_df is None or vix_df.empty:
                print("Warning: Failed to fetch VIX data. Proceeding without VIX.")
                vix_df = pd.DataFrame()
            else:
                print("VIX data successfully fetched.")

                # Prepare VIX data
                vix_df_indexed = None
                if "timestamp" in vix_df.columns:
                    try:
                        if not pd.api.types.is_datetime64_any_dtype(vix_df["timestamp"]):
                            vix_df["timestamp"] = pd.to_datetime(vix_df["timestamp"], utc=True, errors='coerce')
                        elif vix_df["timestamp"].dt.tz is None:
                            vix_df["timestamp"] = vix_df["timestamp"].dt.tz_localize("UTC")
                        else:
                            vix_df["timestamp"] = vix_df["timestamp"].dt.tz_convert("UTC")
                        vix_df.dropna(subset=['timestamp'], inplace=True)
                        if not vix_df.empty:
                            vix_df_indexed = vix_df.set_index("timestamp")
                    except Exception as e:
                        print(f"Error processing VIX timestamp: {e}. Skipping VIX merge.")
                else:
                    print("Warning: 'timestamp' column not found in VIX data. Skipping VIX merge.")

                # Prepare main data
                processed_data_indexed = None
                if "timestamp" in processed_data.columns:
                    try:
                        if not pd.api.types.is_datetime64_any_dtype(processed_data["timestamp"]):
                            processed_data["timestamp"] = pd.to_datetime(processed_data["timestamp"], utc=True, errors='coerce')
                        elif processed_data["timestamp"].dt.tz is None:
                            processed_data["timestamp"] = processed_data["timestamp"].dt.tz_localize("UTC")
                        else:
                            processed_data["timestamp"] = processed_data["timestamp"].dt.tz_convert("UTC")
                        processed_data.dropna(subset=['timestamp'], inplace=True)
                        if not processed_data.empty:
                            processed_data_indexed = processed_data.set_index("timestamp")
                    except Exception as e:
                        print(f"Error processing main data timestamp: {e}. Cannot merge VIX.")
                else:
                    print("CRITICAL WARNING: 'timestamp' column missing from main data. Cannot merge VIX.")

                # Perform merge
                if vix_df_indexed is not None and processed_data_indexed is not None:
                    try:
                        print("Attempting LEFT join on timestamp index...")
                        vix_to_merge = vix_df_indexed[["close"]].rename(columns={"close": "vix"})
                        merged_data = processed_data_indexed.join(vix_to_merge, how="left")
                        processed_data = merged_data.reset_index()
                        vix_successfully_added = True
                        print("VIX data merged successfully (left join).")
                        if "vix" in processed_data.columns:
                            processed_data["vix"] = processed_data["vix"].fillna(0)
                            print("Filled NaN values in 'vix' column with 0.")
                    except Exception as e:
                        print(f"Error during VIX merge: {e}. Proceeding without VIX.")
                        if processed_data_indexed is not None: # Fallback only if indexing was successful
                             processed_data = processed_data_indexed.reset_index()
                        vix_successfully_added = False
                else:
                    print("Skipping VIX merge due to data preparation issues.")
                    if processed_data_indexed is not None: # Fallback only if indexing was successful
                         processed_data = processed_data_indexed.reset_index()
                    vix_successfully_added = False
        # --- End VIX Merging ---

        # Add technical indicators
        DP_generic = DataProcessor(
            data_source="yahoofinance", start_date=start_date, end_date=end_date, **kwargs
        )
        if not processed_data.empty:
            processed_data = DP_generic.add_technical_indicator(
                processed_data, technical_indicator_list
            )
            print("Technical indicators added.")

            # --- Alignment and Filtering ---
            if not processed_data.empty:
                print("Ensuring common date range post-indicators...")
                if "timestamp" not in processed_data.columns:
                    print("Warning: 'timestamp' column missing before final date range alignment. Skipping.")
                else:
                    try:
                        if not pd.api.types.is_datetime64_any_dtype(processed_data["timestamp"]):
                            processed_data["timestamp"] = pd.to_datetime(processed_data["timestamp"], utc=True, errors='coerce')
                        elif processed_data["timestamp"].dt.tz is None:
                            processed_data["timestamp"] = processed_data["timestamp"].dt.tz_localize("UTC")
                        else:
                            processed_data["timestamp"] = processed_data["timestamp"].dt.tz_convert("UTC")
                        processed_data.dropna(subset=['timestamp'], inplace=True)

                        if not processed_data.empty:
                            start_dates = processed_data.groupby("tic")["timestamp"].min()
                            end_dates = processed_data.groupby("tic")["timestamp"].max()
                            if not start_dates.empty and not end_dates.empty:
                                max_start_date = start_dates.max()
                                min_end_date = end_dates.min()
                                processed_data = processed_data[
                                    (processed_data["timestamp"] >= max_start_date)
                                    & (processed_data["timestamp"] <= min_end_date)
                                ]
                                print(f"Final common date range set: {max_start_date} to {min_end_date}")
                            else:
                                print("Warning: Could not determine common date range (empty groups). Skipping alignment.")
                                processed_data = pd.DataFrame()
                        else:
                             print("Warning: Data empty after timestamp conversion. Skipping alignment.")
                    except Exception as e:
                        print(f"Error during final date alignment: {e}. Skipping alignment.")
            # --- End Alignment ---
        else:
            print("Skipping technical indicators on empty dataframe.")

        # --- Save Processed DataFrame to CSV Cache ---
        if not processed_data.empty:
            try:
                df_to_save = processed_data.copy()
                if 'timestamp' in df_to_save.columns and pd.api.types.is_datetime64_any_dtype(df_to_save['timestamp']):
                     df_to_save['timestamp'] = df_to_save['timestamp'].astype(str)
                df_to_save.to_csv(cache_file, index=False)
                print(f"Processed data saved to CSV cache: {cache_file}")
            except Exception as e:
                print(f"Error saving processed data to CSV cache {cache_file}: {e}")
        else:
            print("Skipping saving empty processed data to cache.")
    # --- End Data Processing Pipeline ---

    # --- Array Generation (Runs whether loaded from cache or processed fresh) ---
    print("Converting DataFrame to arrays...")
    if processed_data is None or processed_data.empty:
        print("Warning: Processed data is empty before array conversion. Returning empty arrays.")
        num_tickers = len(data['tic'].unique()) if 'data' in locals() and not data.empty and 'tic' in data.columns else 0
        num_tech_indicators = len(technical_indicator_list)
        price_array = np.empty((0, num_tickers))
        tech_dim = num_tech_indicators + (1 if vix_successfully_added else 0)
        tech_array = np.empty((0, num_tickers * tech_dim))
        turbulence_array = np.empty((0,))
    else:
        # Ensure DP_generic is instantiated if loaded from cache
        if loaded_from_cache:
            DP_generic = DataProcessor(
                data_source="yahoofinance", start_date=start_date, end_date=end_date, **kwargs
            )

        # Ensure 'tic' column exists
        if "tic" not in processed_data.columns:
            print("CRITICAL WARNING: 'tic' column missing before df_to_array! Adding fallback.")
            processed_data["tic"] = "UNKNOWN"

        # Ensure 'timestamp' column exists and is datetime for filtering/alignment
        if "timestamp" not in processed_data.columns:
             print("CRITICAL WARNING: 'timestamp' column missing before final filtering! Attempting reset_index.")
             # This might happen if loaded CSV had no timestamp and index wasn't date
             try:
                 processed_data = processed_data.reset_index() # Try to get index as column
                 if 'index' in processed_data.columns: # If index became 'index' column
                      processed_data['timestamp'] = pd.to_datetime(processed_data['index'], errors='coerce', utc=True)
                      processed_data.dropna(subset=['timestamp'], inplace=True)
                 elif 'date' in processed_data.columns: # Fallback to 'date' if it exists
                      processed_data['timestamp'] = pd.to_datetime(processed_data['date'], errors='coerce', utc=True)
                      processed_data.dropna(subset=['timestamp'], inplace=True)
                 else:
                      raise ValueError("Cannot recover timestamp.")
             except Exception as ts_recovery_err:
                  print(f"Error recovering timestamp: {ts_recovery_err}. Cannot proceed with filtering.")
                  processed_data = pd.DataFrame() # Empty DF if timestamp is unrecoverable

        # --- Explicit Date Range Filtering and Deduplication ---
        print("Applying explicit date range filter and deduplication...")
        if not processed_data.empty and "timestamp" in processed_data.columns:
            if not pd.api.types.is_datetime64_any_dtype(processed_data["timestamp"]):
                try:
                    processed_data["timestamp"] = pd.to_datetime(processed_data["timestamp"], utc=True, errors='coerce')
                    processed_data.dropna(subset=['timestamp'], inplace=True)
                except Exception as e:
                    print(f"Warning: Could not convert timestamp to datetime before filtering: {e}")

            if not processed_data.empty and pd.api.types.is_datetime64_any_dtype(processed_data["timestamp"]):
                print("Normalizing timestamp to date (midnight UTC)...")
                if processed_data["timestamp"].dt.tz is None:
                    processed_data["timestamp"] = processed_data["timestamp"].dt.tz_localize("UTC").dt.normalize()
                else:
                    processed_data["timestamp"] = processed_data["timestamp"].dt.tz_convert("UTC").dt.normalize()

                start_dt_utc = pd.Timestamp(start_date, tz="UTC").normalize()
                end_dt_utc = pd.Timestamp(end_date, tz="UTC").normalize()

                original_rows = len(processed_data)
                processed_data = processed_data[
                    (processed_data["timestamp"] >= start_dt_utc)
                    & (processed_data["timestamp"] <= end_dt_utc)
                ].copy()
                print(f"Filtered by date range [{start_date}, {end_date}]: {original_rows} -> {len(processed_data)} rows")

                original_rows = len(processed_data)
                processed_data = processed_data.drop_duplicates(subset=["timestamp", "tic"], keep="last") # Changed keep="first" to keep="last"
                print(f"Dropped duplicates (timestamp, tic): {original_rows} -> {len(processed_data)} rows")
        else:
            print("Skipping explicit filtering/deduplication: empty dataframe or missing/invalid 'timestamp'.")
        # --- END Filtering/Deduplication ---

        # --- Re-verify common dates ---
        print("Re-verifying common dates before df_to_array...")
        if not processed_data.empty and "tic" in processed_data.columns and "timestamp" in processed_data.columns:
            if not pd.api.types.is_datetime64_any_dtype(processed_data["timestamp"]):
                 try:
                     processed_data["timestamp"] = pd.to_datetime(processed_data["timestamp"], utc=True, errors='coerce')
                     processed_data.dropna(subset=['timestamp'], inplace=True)
                     print("Converted timestamp column to datetime for final alignment.")
                 except Exception as e:
                     print(f"Warning: Could not convert timestamp to datetime for final alignment: {e}")

            if not processed_data.empty and pd.api.types.is_datetime64_any_dtype(processed_data["timestamp"]):
                unique_tickers = processed_data["tic"].unique()
                if len(unique_tickers) > 0:
                    common_dates_sets = [set(processed_data[processed_data["tic"] == tic]["timestamp"]) for tic in unique_tickers]
                    final_common_dates = set.intersection(*common_dates_sets) if common_dates_sets else set()

                    if len(final_common_dates) < len(processed_data["timestamp"].unique()):
                        print(f"Discrepancy found! Filtering processed_data to {len(final_common_dates)} common dates.")
                        processed_data = processed_data[processed_data["timestamp"].isin(final_common_dates)].copy()
                    elif len(final_common_dates) == 0:
                        print("Error: No common dates found across all tickers in final verification step!")
                        processed_data = pd.DataFrame() # Set to empty if no common dates
                    else:
                        print("Final date verification passed. All tickers seem aligned.")
                else:
                    print("Skipping final date alignment check (no tickers found).")
            else:
                print("Skipping final date alignment due to non-datetime timestamp column or empty dataframe after conversion attempt.")
        else:
            print("Skipping final date alignment check (empty df or missing 'tic'/'timestamp' columns).")
        # --- End Re-verify ---

        # --- df_to_array call ---
        if processed_data.empty:
             print("Warning: Processed data became empty after alignment/filtering. Returning empty arrays.")
             num_tickers = len(data['tic'].unique()) if 'data' in locals() and not data.empty and 'tic' in data.columns else 0
             num_tech_indicators = len(technical_indicator_list)
             price_array = np.empty((0, num_tickers))
             tech_dim = num_tech_indicators + (1 if vix_successfully_added else 0)
             tech_array = np.empty((0, num_tickers * tech_dim))
             turbulence_array = np.empty((0,))
        else:
            # Ensure the DataProcessor instance knows the indicator list, even if loaded from cache
            DP_generic.tech_indicator_list = technical_indicator_list
            try:
                processed_data = processed_data.reset_index(drop=True) # Ensure index is reset
                if vix_successfully_added and "vix" in processed_data.columns:
                    print("Renaming 'vix' column to 'VIXY' for df_to_array compatibility.")
                    processed_data = processed_data.rename(columns={"vix": "VIXY"})

                print(f"DEBUG: Shape of processed_data before df_to_array: {processed_data.shape}")
                if "tic" in processed_data.columns:
                     print(f"DEBUG: Unique tickers in processed_data before df_to_array: {processed_data['tic'].unique()}")
                     print("DEBUG: Row counts per ticker before df_to_array:")
                     print(processed_data.groupby("tic").size())

                price_array, tech_array, turbulence_array = DP_generic.df_to_array(
                    processed_data, vix_successfully_added
                )
                print("Data processed and converted to arrays.")
            except Exception as e:
                print(f"Error during df_to_array: {e}") # Revert detailed error print
                num_tickers = len(processed_data["tic"].unique()) if "tic" in processed_data.columns else 0
                num_tech_indicators = len(technical_indicator_list)
                price_array = np.empty((0, num_tickers))
                tech_dim = num_tech_indicators + (1 if vix_successfully_added else 0)
                tech_array = np.empty((0, num_tickers * tech_dim))
                turbulence_array = np.empty((0,))
        # --- End df_to_array call ---

    # Return arrays and the final processed dataframe (which might be loaded or freshly processed)
    return price_array, tech_array, turbulence_array, processed_data


# --- Training Function ---


def run_train(
    start_date,
    end_date,
    ticker_list,
    data_source,
    time_interval,
    technical_indicator_list,
    drl_lib,
    env,
    model_name,
    if_vix=True,
    **kwargs,
):
    """Fetches data, processes it, and runs the training loop."""
    # Get and process data
    raw_cache_dir = kwargs.get("raw_cache_dir", RAW_DATA_CACHE_DIR)
    processed_cache_dir = kwargs.get("processed_cache_dir", PROCESSED_DATA_CACHE_DIR)
    # Get base results dir
    base_results_dir = kwargs.get("results_dir", "./results")
    # Create library-specific results directory
    lib_results_dir = os.path.join(base_results_dir, drl_lib)
    os.makedirs(lib_results_dir, exist_ok=True)
    print(f"Results for this run will be saved under: {lib_results_dir}")

    raw_data = get_data(
        start_date,
        end_date,
        ticker_list,
        data_source,
        time_interval,
        cache_dir=raw_cache_dir,
        **kwargs,
    )

    # Generate hash for raw data to use in processed data caching filename
    ticker_str = "_".join(sorted(ticker_list))
    raw_filename_base = (
        f"{data_source}_{ticker_str}_{start_date}_{end_date}_{time_interval}_raw"
    )
    raw_data_hash = hashlib.md5(raw_filename_base.encode()).hexdigest()

    price_array, tech_array, turbulence_array, _ = process_data(
        data=raw_data,
        technical_indicator_list=technical_indicator_list,
        start_date=start_date,  # Pass dates
        end_date=end_date,  # Pass dates
        if_vix=if_vix,
        cache_dir=processed_cache_dir,
        raw_data_hash=raw_data_hash,
        **kwargs,
    )

    # Check if arrays are empty (e.g., due to processing error)
    if price_array.size == 0 or tech_array.size == 0 or turbulence_array.size == 0:
        print(
            "Error: Data processing resulted in empty arrays. Cannot proceed with training."
        )
        return

    # --- DEBUG LOGS for Training ---
    print(f"DEBUG TRAIN: Tickers used: {ticker_list}")
    print(f"DEBUG TRAIN: Indicators used: {technical_indicator_list}")
    # --- END DEBUG ---
    # Setup environment config
    env_config = {
        "price_array": price_array,
        "tech_array": tech_array,
        "turbulence_array": turbulence_array,
        "if_train": True,
    }
    # Set API version based on library
    # Set API version based on library requirements
    # Stable-Baselines3 expects Gymnasium API (reset->(obs,info), step->(obs,rew,term,trunc,info))
    # ElegantRL and RLlib seem to expect the older Gym API (reset->obs, step->(obs,rew,done,info))
    if drl_lib == "stable_baselines3":
        env_config["api_version"] = "gymnasium"
    else:  # elegantrl, rllib
        env_config["api_version"] = "gym"
    env_instance = env(config=env_config)
    # --- DEBUG LOGS for Training ---
    try:
        print(
            f"DEBUG TRAIN: Training Env Observation Space Shape: {env_instance.observation_space.shape}"
        )
    except AttributeError:
        print(
            f"DEBUG TRAIN: Training Env Observation Space: {env_instance.observation_space} (Shape attribute not found)"
        )
    # --- END DEBUG ---

    # Determine default CWD base for saving models (SB3 zip, RLlib base) - Now within lib_results_dir
    default_cwd_base = os.path.join(lib_results_dir, f"trained_models") # Base within lib dir
    os.makedirs(default_cwd_base, exist_ok=True) # Ensure this exists

    # Use specified training CWD override if provided via args.train_cwd (passed in kwargs['cwd'])
    # If not provided (i.e., kwargs['cwd'] is None), use the default base.
    train_model_save_dir = kwargs.get("cwd")  # Get the potential override
    if train_model_save_dir is None:
        train_model_save_dir = default_cwd_base  # Fallback to default if not provided
    else:
        # If a custom CWD is provided, ensure it exists
        os.makedirs(train_model_save_dir, exist_ok=True)


    print(f"--- Starting Training: {drl_lib} - {model_name} ---")
    print(f"Data: {start_date} to {end_date}")
    print(f"Tickers: {ticker_list}")
    # Print the directory where the final *model* will be saved (for SB3/RLlib)
    print(f"Model Output directory (train_model_save_dir): {train_model_save_dir}")

    if drl_lib == "elegantrl":
        # Default from original script or args
        break_step = kwargs.get("break_step")
        if break_step is None:
            break_step = 1e5  # Default break_step for eRL
            print(f"Using default break_step for eRL: {int(break_step)}")
        break_step = int(break_step)  # Ensure it's an integer for path naming

        # Use default if not provided
        erl_params = kwargs.get("erl_params", ERL_PARAMS)

        # --- TensorBoard Logging Setup for ElegantRL ---
        # Logs will go into the library-specific results directory
        tb_log_path_erl = os.path.join(lib_results_dir, "tensorboard_logs")
        os.makedirs(tb_log_path_erl, exist_ok=True)

        log_name_erl = f"{model_name}_{break_step}"  # Use integer break_step
        full_erl_log_path = os.path.join(tb_log_path_erl, log_name_erl)

        # Remove existing log directory to prevent conflicts
        if os.path.exists(full_erl_log_path):
            print(f"Removing existing erl log directory: {full_erl_log_path}")
            shutil.rmtree(full_erl_log_path)

        # Initialize TensorBoard writer
        writer = SummaryWriter(log_dir=full_erl_log_path)
        print(f"TensorBoard logging initialized at: {full_erl_log_path}")
        # --- End TensorBoard Setup ---

        agent = DRLAgent_erl(
            env=env,  # Pass class, not instance for eRL
            price_array=price_array,
            tech_array=tech_array,
            turbulence_array=turbulence_array,
        )

        model = agent.get_model(model_name, model_kwargs=erl_params)

        print(
            f"ElegantRL training initiated. Logs will be saved in: {full_erl_log_path}"
        )

        # Call the agent's internal training method
        # Assuming a signature similar to SB3/RLlib integrations
        trained_model = agent.train_model(
            model=model, total_timesteps=break_step, cwd=full_erl_log_path
        )  # Use the TB log path as cwd
        print(
            f"ElegantRL training finished. Model saved potentially within agent's logic."
        )
        # Note: ElegantRL agent might handle saving and logging differently, check its implementation.
        # We assign to trained_model for consistency, but its usage might differ.

    elif drl_lib == "rllib":
        # Default from original script or args
        total_episodes = kwargs.get("total_episodes")
        if total_episodes is None:
            total_episodes = 30  # Default total_episodes for RLlib
            print(f"Using default total_episodes for RLlib: {total_episodes}")

        # Use default if not provided
        rllib_params = kwargs.get("rllib_params", RLlib_PARAMS)

        # Ensure Ray is initialized (and shutdown previous session if needed)
        if ray.is_initialized():
            print("Shutting down previous Ray session...")
            ray.shutdown()
        print("Initializing Ray...")
        ray.init(
            ignore_reinit_error=True, num_cpus=kwargs.get("ray_num_cpus", None)
        )  # Allow specifying CPU count

        agent_rllib = DRLAgent_rllib(
            env=env,  # Pass class, not instance for RLlib
            price_array=price_array,
            tech_array=tech_array,
            turbulence_array=turbulence_array,
        )

        model, model_config = agent_rllib.get_model(model_name)

        # Apply specific params
        model_config["lr"] = rllib_params["lr"]
        model_config["train_batch_size"] = rllib_params["train_batch_size"]
        model_config["gamma"] = rllib_params["gamma"]
        # Add other necessary config from rllib_params if needed

        # Use lib_results_dir as the base for RLlib logs/checkpoints
        rllib_log_dir = os.path.join(lib_results_dir, "rllib_results") # Specific subdir for RLlib outputs
        os.makedirs(rllib_log_dir, exist_ok=True)
        print(
            f"RLlib results (checkpoints, logs) will be stored within: {rllib_log_dir}"
        )

        trained_model = agent_rllib.train_model(
            model=model,
            model_name=model_name,  # Pass model_name string
            model_config=model_config,
            results_dir=rllib_log_dir,  # Specify the parent log directory
            total_episodes=total_episodes,
        )
        # Checkpoint saving is handled internally by DRLAgent_rllib/Ray Tune within results_dir
        ray.shutdown()  # Shutdown Ray after training
        print("Ray session shut down.")

    elif drl_lib == "stable_baselines3":
        # Default from original script or args
        total_timesteps = kwargs.get("total_timesteps")
        if total_timesteps is None:
            total_timesteps = 1e5  # Default total_timesteps for SB3
            print(f"Using default total_timesteps for SB3: {int(total_timesteps)}")
        total_timesteps = int(total_timesteps)  # Ensure integer

        # Get early stopping flag from kwargs
        enable_early_stopping = kwargs.get("enable_early_stopping", False)

        agent_params = kwargs.get("agent_params", AGENT_PARAMS_MAP.get(model_name, {}))

        # Ensure tensorboard_log is NOT in agent_params if passed separately
        agent_params.pop("tensorboard_log_sb3", None)

        agent = DRLAgent_sb3(env=env_instance)


        # --- Custom Callback for Early Stopping ---
        # Necessary because the default EvalCallback in older SB3 versions
        # might not support stop_train_on_no_improvement, patience, min_delta directly.
        class StopTrainingOnNoImprovementCallback(EvalCallback):
            def __init__(self, eval_env, patience: int, min_delta: float, *args, **kwargs):
                super().__init__(eval_env, *args, **kwargs)
                self.patience = patience
                self.min_delta = min_delta
                self.wait_count = 0
                # Initialize best_mean_reward to a very low value
                self.best_mean_reward = -float('inf')

            def _on_step(self) -> bool:
                # First, run the parent's _on_step to perform evaluation
                continue_training = super()._on_step()

                # If the parent callback decided to stop (e.g., due to nan), respect that
                if not continue_training:
                    return False

                # Check if an evaluation was performed in this step
                # self.last_mean_reward is updated by the parent EvalCallback after evaluation
                if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
                    current_mean_reward = self.last_mean_reward

                    if self.verbose > 0:
                        print(f"Custom Callback: Eval mean reward: {current_mean_reward:.4f}, Best mean reward: {self.best_mean_reward:.4f}")

                    # Check for improvement
                    if current_mean_reward > self.best_mean_reward + self.min_delta:
                        self.best_mean_reward = current_mean_reward
                        self.wait_count = 0
                        if self.verbose > 0:
                            print(f"Custom Callback: New best mean reward found: {self.best_mean_reward:.4f}")
                        # Save the best model explicitly if the parent didn't (though it should)
                        if self.best_model_save_path is not None:
                            self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                            if self.verbose > 0:
                                print(f"Custom Callback: Saving best model to {self.best_model_save_path}")

                    else:
                        self.wait_count += 1
                        if self.verbose > 0:
                            print(f"Custom Callback: No improvement greater than {self.min_delta}. Wait count: {self.wait_count}/{self.patience}")

                    # Check if patience is exceeded
                    if self.wait_count >= self.patience:
                        if self.verbose > 0:
                            print(f"Custom Callback: Stopping training early as no improvement observed for {self.patience} evaluations.")
                        return False # Stop training

                return True # Continue training
        # --- End Custom Callback ---

        # --- Create Evaluation Environment (only if early stopping enabled) ---
        eval_env = None
        if enable_early_stopping:
            print("Creating evaluation environment for SB3 early stopping...")
            eval_env_config = {
                "price_array": price_array,
                "tech_array": tech_array,
                "turbulence_array": turbulence_array,
                "if_train": False, # Set if_train to False for evaluation
                "api_version": "gymnasium" # SB3 uses Gymnasium API
            }
            eval_env = env(config=eval_env_config)
        # --- End Evaluation Environment ---


        # Define the base log path using the library-specific results directory
        tb_log_path_sb3 = os.path.join(lib_results_dir, "tensorboard_logs")
        os.makedirs(tb_log_path_sb3, exist_ok=True)

        # Define the specific run log name
        log_name_sb3 = f"{model_name}_{total_timesteps}"  # Use integer total_timesteps
        full_tb_log_path = os.path.join(tb_log_path_sb3, log_name_sb3)

        # ADDED: Remove existing log directory to prevent SB3 adding suffixes (_1, _2, ...)
        if os.path.exists(full_tb_log_path):
            print(f"Removing existing tensorboard log directory: {full_tb_log_path}")
            shutil.rmtree(full_tb_log_path)

        # --- Setup Custom EvalCallback (conditionally) ---
        callback_to_use = None # Default to no callback
        if enable_early_stopping and eval_env is not None:
            eval_freq = int(kwargs.get("eval_freq", 10000)) # Get eval_freq from kwargs, default 10000
            print(f"Setting up Custom StopTrainingOnNoImprovementCallback with eval_freq={eval_freq}, patience=20, min_delta=0.0")
            callback_to_use = StopTrainingOnNoImprovementCallback(
                eval_env=eval_env,
                patience=20,
                min_delta=0.0,
                # Pass other necessary args for EvalCallback
                best_model_save_path=train_model_save_dir, # Save best model here
                log_path=full_tb_log_path, # Log eval results here
                eval_freq=eval_freq,
                n_eval_episodes=5, # Number of episodes to evaluate
                deterministic=True,
                render=False,
                verbose=1
            )
        elif enable_early_stopping and eval_env is None:
             print("Warning: Early stopping enabled but failed to create eval_env. Proceeding without callback.")
        else:
            print("Early stopping disabled. Proceeding without evaluation callback.")
        # --- End Custom EvalCallback Setup ---
        # Pass agent_params AND tensorboard_log separately
        model = agent.get_model(
            model_name,
            model_kwargs=agent_params,
            tensorboard_log=tb_log_path_sb3,  # Pass the base log directory path here
        )

        # Train the model directly using model.learn(), passing the conditional callback
        # bypassing the DRLAgent.train_model wrapper which hardcodes a different callback.
        print("Calling model.learn() directly to use EvalCallback...")
        trained_model = model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=log_name_sb3,  # Use the specific, non-suffixed log name
            callback=callback_to_use # Pass the callback instance (or None)
        )
        # Print the full TensorBoard log path after training
        print(f"SB3 Tensorboard log saved to: {full_tb_log_path}")

        print("SB3 Training finished!")
        # Use the main 'train_model_save_dir' for saving the final model zip
        save_path = os.path.join(train_model_save_dir, f"{model_name}.zip")
        trained_model.save(save_path)
        print(f"SB3 Trained model saved to: {save_path}")
    else:
        raise ValueError("DRL library input is NOT supported. Please check.")

    print(f"--- Training Complete: {drl_lib} - {model_name} ---")


# --- Testing Function ---
def run_test(
    start_date,
    end_date,
    ticker_list,
    data_source,
    time_interval,
    technical_indicator_list,
    drl_lib,
    env,
    model_name,
    if_vix=True,
    **kwargs,
):
    """Fetches data, processes it, and runs the testing/prediction loop."""
    # Get and process data
    raw_cache_dir = kwargs.get("raw_cache_dir", RAW_DATA_CACHE_DIR)
    processed_cache_dir = kwargs.get("processed_cache_dir", PROCESSED_DATA_CACHE_DIR)
    # Get base results dir
    base_results_dir = kwargs.get("results_dir", "./results")
    # Create library-specific results directory
    lib_results_dir = os.path.join(base_results_dir, drl_lib)
    os.makedirs(lib_results_dir, exist_ok=True)
    print(f"Test results for this run will be saved under: {lib_results_dir}")


    raw_data = get_data(
        start_date,
        end_date,
        ticker_list,
        data_source,
        time_interval,
        cache_dir=raw_cache_dir,
        **kwargs,
    )

    # Generate hash for raw data to use in processed data caching filename
    ticker_str = "_".join(sorted(ticker_list))
    raw_filename_base = (
        f"{data_source}_{ticker_str}_{start_date}_{end_date}_{time_interval}_raw"
    )
    raw_data_hash = hashlib.md5(raw_filename_base.encode()).hexdigest()

    price_array, tech_array, turbulence_array, data_df = process_data(
        data=raw_data,
        technical_indicator_list=technical_indicator_list,
        start_date=start_date,  # Pass dates
        end_date=end_date,  # Pass dates
        if_vix=if_vix,
        cache_dir=processed_cache_dir,
        raw_data_hash=raw_data_hash,
        **kwargs,
    )  # Keep data_df for baseline

    # --- DEBUG PRINT ---
    print(f"DEBUG: process_data returned price_array shape: {price_array.shape}")
    # Removed CSCO price debug print
    # --- END DEBUG PRINT ---

    # Check if arrays are empty (e.g., due to processing error)
    # Use .size check for numpy arrays
    if price_array.size == 0 or tech_array.size == 0 or turbulence_array.size == 0:
        print(
            "Error: Data processing resulted in empty arrays. Cannot proceed with testing."
        )
        return None, None  # Return None to indicate failure

    # --- DEBUG LOGS for Testing ---
    print(f"DEBUG TEST: Tickers used: {ticker_list}")
    print(f"DEBUG TEST: Indicators used: {technical_indicator_list}")
    # --- END DEBUG ---
    # Setup environment config for testing
    env_config = {
        "price_array": price_array,
        "tech_array": tech_array,
        "turbulence_array": turbulence_array,
        "if_train": False,
    }  # Set if_train to False
    # Set API version based on library
    # Set API version based on library requirements (Consistent with run_train)
    # Stable-Baselines3 expects Gymnasium API
    # ElegantRL and RLlib seem to expect the older Gym API
    if drl_lib == "stable_baselines3":
        env_config["api_version"] = "gymnasium"
    else:  # elegantrl, rllib
        env_config["api_version"] = "gym"
    env_instance = env(config=env_config)
    # --- DEBUG PRINT: Check observation space shape ---
    try:
        print(
            f"DEBUG: Test Environment Observation Space Shape: {env_instance.observation_space.shape}"
        )
    except AttributeError:
        print(
            f"DEBUG: Test Environment Observation Space: {env_instance.observation_space} (Shape attribute not found)"
        )
    # --- END DEBUG PRINT ---

    # Load parameters
    # Determine default CWD (model loading path) based on training output structure
    # Adjust based on where train step saves models/logs
    if drl_lib == "elegantrl":
        # Use the specific log/model path structure created during training
        # Need break_step used during training to reconstruct path
        # Attempt to get it from kwargs, fallback to default used in train
        break_step = kwargs.get("break_step")
        if break_step is None:
            break_step = 1e5  # Default from train
        log_name_erl = f"{model_name}_{int(break_step)}"
        # Load from the library-specific results dir structure
        default_cwd_load = os.path.join(lib_results_dir, "tensorboard_logs", log_name_erl)
    elif drl_lib == "rllib":
        # RLlib saves checkpoints within a structure generated by Ray Tune.
        # Default points to the base dir used in training (now within lib_results_dir).
        # User likely needs to provide a more specific path via --test_cwd.
        default_cwd_load = os.path.join(lib_results_dir, "rllib_results") # Point to the base RLlib dir
    else:  # stable_baselines3
        # SB3 saves a zip file in the trained_models subdir within lib_results_dir
        default_cwd_load = os.path.join(lib_results_dir, "trained_models") # Point to the base SB3 models dir

    # Use specified test_cwd if provided (via kwargs['test_cwd']), otherwise use library-specific default
    cwd_load_path = kwargs.get("test_cwd")  # Get potential override from --test_cwd
    if cwd_load_path is None:
        cwd_load_path = default_cwd_load  # Fallback to default if --test_cwd not used

    # --- Path adjustments for loading ---
    if drl_lib == "rllib":
        # RLlib needs path to specific checkpoint dir/file.
        # 'cwd_load_path' needs refinement or user must provide exact path.
        agent_path = cwd_load_path  # Assume user provides correct path via test_cwd if default isn't right
        print(
            f"Using RLlib agent path (ensure it points to checkpoint dir/file): {agent_path}"
        )
    elif drl_lib == "stable_baselines3":
        # SB3 loads from a .zip file
        # If cwd_load_path is a dir, append the expected zip filename
        if not cwd_load_path.endswith(".zip") and os.path.isdir(cwd_load_path):
            potential_path = os.path.join(cwd_load_path, f"{model_name}.zip")
            if os.path.exists(potential_path):
                cwd_load_path = potential_path
                print(f"Found SB3 model zip: {cwd_load_path}")
            else:
                print(
                    f"Warning: SB3 model .zip not found at {potential_path}. Trying directory path {cwd_load_path}, but prediction might fail."
                )
        # If cwd_load_path already ends with .zip or isn't a dir, use it as is
        print(f"Using SB3 model path: {cwd_load_path}")
    elif drl_lib == "elegantrl":
        # ElegantRL loads from the directory where checkpoints are saved (the CWD passed during training)
        print(f"Using eRL model directory: {cwd_load_path}")
    # --- End Path adjustments ---

    print(f"--- Starting Testing: {drl_lib} - {model_name} ---")
    print(f"Data: {start_date} to {end_date}")
    print(f"Tickers: {ticker_list}")
    print(f"Loading model from: {cwd_load_path}")  # Use the adjusted path

    # Load elegantrl needs state dim, action dim and net dim
    # Use default from ERL_PARAMS if not provided via args/kwargs
    net_dimension = kwargs.get(
        "test_net_dimension", kwargs.get("net_dimension", ERL_PARAMS["net_dimension"])
    )

    episode_total_assets = None

    if drl_lib == "elegantrl":
        # --- Load Model ---
        print("Attempting to load ElegantRL model with safe_globals context...")
        actor_model = None
        try:
            # Add necessary classes for safe loading
            with torch.serialization.safe_globals(
                [ActorPPO, nn.Sequential, nn.Linear, nn.ReLU, Normal]
            ):
                actor_path = os.path.join(
                    cwd_load_path, "act.pth"
                )  # Construct path to actor file
                if not os.path.exists(actor_path):
                    raise FileNotFoundError(
                        f"Actor model file not found at: {actor_path}"
                    )
                # Load the state dict or the full model depending on how it was saved
                # Assuming torch.load loads the entire actor network instance here
                actor_model = torch.load(actor_path)
                # Determine device (use GPU 0 if available, else CPU) - adjust if needed
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                actor_model.to(device)
                actor_model.eval()  # Set model to evaluation mode
                print(
                    f"Actor model loaded successfully from {actor_path} onto {device}."
                )

        except NameError as e:
            print(
                f"Error: A required class for safe_globals not defined (import likely failed): {e}"
            )
            raise  # Re-raise error as loading is critical
        except FileNotFoundError as e:
            print(f"Error loading model: {e}")
            raise  # Re-raise error
        except Exception as e:
            print(f"Error during model loading even with safe_globals: {e}")
            raise  # Re-raise other potential errors

        # --- Custom Prediction Loop ---
        if actor_model:
            # --- Add diagnostic prints for env_instance ---
            print(f"DEBUG RS: Type of env_instance before loop: {type(env_instance)}")
            try:
                print(
                    f"DEBUG RS: Module of env_instance before loop: {env_instance.__class__.__module__}"
                )
            except AttributeError:
                print(f"DEBUG RS: Could not get module for env_instance.")
            # --- End diagnostic prints ---
            print("Starting custom prediction loop...")
            episode_total_assets = []
            # Use the env_instance created earlier
            # Gym API reset returns only state
            state = env_instance.reset()
            # Ensure initial_total_asset is accessible, handle potential AttributeError
            try:
                initial_asset = env_instance.initial_total_asset
            except AttributeError:
                initial_asset = (
                    env_instance.amount
                    + (
                        env_instance.stocks * env_instance.price_ary[env_instance.day]
                    ).sum()
                )
            episode_total_assets.append(initial_asset)  # Record initial value

            all_trades = []
            with torch.no_grad():  # Disable gradient calculations during inference
                for step in range(env_instance.max_step):
                    state_tensor = torch.as_tensor(
                        state, dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    action_tensor = actor_model(state_tensor)
                    # Assuming continuous actions based on previous logs
                    action = action_tensor.detach().cpu().numpy()[0]
                    print(f"Step {step}: action = {action}")

                    # Call step on our local environment instance
                    # Gym API step returns state, reward, done, info
                    state, reward, done, info = env_instance.step(action)

                    # Collect trade logs for this step
                    if info and "trade_log" in info and info["trade_log"]:
                        all_trades.extend(info["trade_log"])

                    # Record total asset value for this step
                    try:
                        current_total_asset = env_instance.total_asset
                    except AttributeError:
                        current_total_asset = (
                            env_instance.amount
                            + (
                                env_instance.stocks
                                * env_instance.price_ary[env_instance.day]
                            ).sum()
                        )
                    episode_total_assets.append(current_total_asset)

                    if done:
                        print(f"Episode finished at step {step}.")
                        break
            # Save all trades to CSV
            if all_trades:
                trades_df = pd.DataFrame(all_trades)
                # Save to library-specific results directory
                trades_csv_path = os.path.join(lib_results_dir, f"agent_transactions_{drl_lib}_{model_name}_{start_date}_to_{end_date}.csv")
                trades_df.to_csv(trades_csv_path, index=False)
                print(f"Agent transaction log saved to: {trades_csv_path}")
            else:
                print("No agent transactions to log.")
            print("Custom prediction loop finished.")
        else:
            print("Actor model not loaded, cannot run prediction loop.")
            # Ensure episode_total_assets is None or handled appropriately if loading failed
            episode_total_assets = None

    elif drl_lib == "rllib":
        # Ensure Ray is initialized (and shutdown previous session if needed)
        if ray.is_initialized():
            print("Shutting down previous Ray session...")
            ray.shutdown()
        print("Initializing Ray for prediction...")
        ray.init(ignore_reinit_error=True, num_cpus=kwargs.get("ray_num_cpus", None))

        # DRL_prediction needs the agent_path pointing to the checkpoint
        episode_total_assets = DRLAgent_rllib.DRL_prediction(
            model_name=model_name,  # Pass model name string
            env=env,  # Pass env class
            price_array=price_array,
            tech_array=tech_array,
            turbulence_array=turbulence_array,
            agent_path=agent_path,  # Use the determined agent path
        )
        ray.shutdown()  # Shutdown Ray after prediction
        print("Ray session shut down.")

    elif drl_lib == "stable_baselines3":
        # DRL_prediction_load_from_file expects the path to the .zip file
        # --- DEBUG LOGS for Testing ---
        print(f"DEBUG TEST: Loading SB3 model from path: {cwd_load_path}")
        # --- END DEBUG ---
        episode_total_assets = DRLAgent_sb3.DRL_prediction_load_from_file(
            model_name=model_name,  # Pass model name string
            environment=env_instance,  # Pass env instance
            cwd=cwd_load_path,  # Use the determined model path (should be .zip)
            # --- ADDED arguments for trade log saving ---
            results_dir=lib_results_dir, # Pass the library-specific results directory
            start_date=start_date,   # Pass test start date
            end_date=end_date,     # Pass test end date
            # --- END ADDED ---
            ticker_list=ticker_list # Pass the list of tickers
        )

    else:
        raise ValueError("DRL library input is NOT supported. Please check.")

    print(f"--- Testing Complete: {drl_lib} - {model_name} ---")

    # --- Calculate and Print Test Statistics --- START ADDED BLOCK ---
    if episode_total_assets is not None and data_df is not None:
        print(f"Test period length: {len(episode_total_assets)} days")
        if len(episode_total_assets) > 1:
            try:
                # Prepare DataFrame for backtest_stats
                # Ensure 'timestamp' column exists before trying to access it
                if "timestamp" in data_df.columns:
                    unique_dates = data_df.timestamp.unique()
                elif (
                    "date" in data_df.columns
                ):  # Fallback if 'timestamp' isn't there but 'date' is
                    unique_dates = data_df.date.unique()
                    print(
                        "Warning: Using 'date' column instead of 'timestamp' from test data_df."
                    )
                else:
                    print(
                        "Error: Neither 'timestamp' nor 'date' column found in test data_df. Cannot calculate stats."
                    )
                    raise KeyError("Missing date/timestamp column in test data_df")

                if len(episode_total_assets) > len(unique_dates):
                    print(
                        f"Warning: Length mismatch for stats calculation. Trimming account values."
                    )
                    account_value_results_trimmed = episode_total_assets[
                        0 : len(unique_dates)
                    ]
                    dates_to_use = unique_dates
                elif len(episode_total_assets) < len(unique_dates):
                    print(
                        f"Warning: Length mismatch for stats calculation. Trimming dates."
                    )
                    dates_to_use = unique_dates[0 : len(episode_total_assets)]
                    account_value_results_trimmed = episode_total_assets
                else:
                    dates_to_use = unique_dates
                    account_value_results_trimmed = episode_total_assets

                account_df_temp = pd.DataFrame(
                    {
                        "date": dates_to_use,
                        "account_value": account_value_results_trimmed,
                    }
                )
                # Ensure date column is datetime
                account_df_temp["date"] = pd.to_datetime(account_df_temp["date"])
                # Make date timezone naive (required by some versions/uses of backtest_stats)
                account_df_temp["date"] = account_df_temp["date"].dt.tz_localize(None)

                # --- Check for zero volatility ---
                # Calculate returns on a temporary Series to avoid modifying df index yet
                temp_returns_series = account_df_temp.set_index("date")[
                    "account_value"
                ].pct_change(1)
                if (
                    temp_returns_series.std() < 1e-6
                    or temp_returns_series.isnull().all()
                ):
                    print(
                        "\nWarning: Portfolio value did not change significantly during the test period."
                    )
                    print(
                        "Performance statistics may be zero or NaN due to lack of trades or zero net profit.\n"
                    )
                # --- End Check ---

                print(
                    "--- Calculating Test Performance Metrics (using backtest_stats) ---"
                )
                # Suppress RuntimeWarnings during calculation
                with np.errstate(divide="ignore", invalid="ignore"):
                    # MODIFIED LINE: Pass the DataFrame *and* specify the value column name
                    perf_stats_test = backtest_stats(
                        account_value=account_df_temp, value_col_name="account_value"
                    )

                try:
                    print("Attempting to save backtest statistics to file...")
                    # Save to library-specific results directory
                    stats_filename = os.path.join(
                        lib_results_dir,
                        f"perf_stats_{drl_lib}_{model_name}_{start_date}_to_{end_date}.txt",
                    )
                    # stats_filename = (
                    #     f"{MODEL_FILENAME_BASE}_backtest_stats.txt"  # Base already includes suffix
                    # )
                    # stats_path = os.path.join(BACKTEST_RESULTS_DIR, stats_filename)
                    with open(stats_filename, "w") as f:
                        f.write(perf_stats_test.to_string())  # Convert Series to string before writing
                    print(f"Successfully saved backtest stats to {stats_filename}")
                except Exception as e_save:
                    print(f"ERROR saving backtest stats to file {stats_filename}: {e_save}", exc_info=True)

                perf_stats_test_df = pd.DataFrame(perf_stats_test)
                # print(perf_stats_test_df) # Commented out to avoid double printing
                print(
                    "-------------------------------------------------------------------"
                )

            except Exception as e:
                print(f"Error calculating test performance stats: {e}")
        else:
            print("--- Test Performance Metrics ---")
            print(
                f"Initial/Final Portfolio Value: ${episode_total_assets[0]:,.2f} (Test duration too short for detailed stats)"
            )
            print("--------------------------------")
    else:
        print(
            "Warning: episode_total_assets or data_df is None. Cannot calculate performance metrics."
        )
    # --- Calculate and Print Test Statistics --- END ADDED BLOCK ---

    return episode_total_assets, data_df, ticker_list  # Return ticker_list as well


# --- Plotting Utility Functions ---


def save_performance_stats(
    account_value_df, results_dir, drl_lib, model_name, start_date, end_date
):
    """Calculates and saves performance stats to a CSV file in the library-specific directory."""
    print("Calculating backtest stats for saving...")
    # Create library-specific results directory if it doesn't exist (robustness)
    lib_results_dir = os.path.join(results_dir, drl_lib)
    os.makedirs(lib_results_dir, exist_ok=True)
    try:
        # Prepare DataFrame for saving stats
        account_value_df_save = account_value_df.copy()
        # Ensure date column is datetime and timezone naive
        account_value_df_save["date"] = pd.to_datetime(account_value_df_save["date"])
        account_value_df_save["date"] = account_value_df_save["date"].dt.tz_localize(
            None
        )

        # --- Check for zero volatility ---
        temp_returns_series_save = account_value_df_save.set_index("date")[
            "account_value"
        ].pct_change(1)
        if (
            temp_returns_series_save.std() < 1e-6
            or temp_returns_series_save.isnull().all()
        ):
            print(
                "\nWarning (for saving): Portfolio value did not change significantly. Stats may be zero or NaN.\n"
            )
        # --- End Check ---

        # Suppress RuntimeWarnings during calculation
        with np.errstate(divide="ignore", invalid="ignore"):
            # Pass DataFrame and specify value column name
            perf_stats_all = backtest_stats(
                account_value=account_value_df_save, value_col_name="account_value"
            )

        perf_stats_all_df = pd.DataFrame(perf_stats_all)
        # Save to library-specific results directory
        stats_filename = os.path.join(
            lib_results_dir,
            f"perf_stats_{drl_lib}_{model_name}_{start_date}_to_{end_date}.csv",
        )
        perf_stats_all_df.to_csv(stats_filename)
        print(f"Performance stats saved to: {stats_filename}")
    except Exception as e:
        print(f"Error calculating/saving backtest stats: {e}")


def plot_agent_vs_baseline(
    account_value_df,
    baseline_all_tickers_df,
    time_interval,
    results_dir,
    drl_lib,
    model_name,
    start_date,
    end_date,
):
    """Generates a plot comparing agent cumulative return vs. an equal-weighted baseline, saved to library-specific directory."""
    print("Generating Agent vs. Baseline plot...")
    # Create library-specific results directory if it doesn't exist (robustness)
    lib_results_dir = os.path.join(results_dir, drl_lib)
    os.makedirs(lib_results_dir, exist_ok=True)
    try:
        # --- Calculate Equal-Weighted Baseline ---
        print("Calculating equal-weighted baseline return...")
        # Pivot table to get closing prices with dates as index and tickers as columns
        baseline_pivot = baseline_all_tickers_df.pivot_table(
            index="date", columns="tic", values="close"
        )

        # Calculate daily returns for each ticker
        ticker_returns = baseline_pivot.pct_change()  # Daily returns for each ticker

        # Calculate mean daily return (equal-weighted portfolio)
        equal_weighted_daily_return = ticker_returns.mean(axis=1)

        # Calculate cumulative return for the equal-weighted baseline
        baseline_cumulative_return_series = (
            1 + equal_weighted_daily_return
        ).cumprod() - 1
        baseline_cumulative_return_series = baseline_cumulative_return_series.fillna(
            0
        )  # Fill first NaN

        # Convert to DataFrame for merging
        baseline_eq_wgt_df = baseline_cumulative_return_series.reset_index()
        baseline_eq_wgt_df.columns = ["date", "baseline_cumulative_return"]
        # Ensure baseline date is datetime and normalized
        baseline_eq_wgt_df["date"] = pd.to_datetime(baseline_eq_wgt_df["date"], errors='coerce')
        baseline_eq_wgt_df = baseline_eq_wgt_df.dropna(subset=["date"]) # Drop failed conversions
        baseline_eq_wgt_df["date"] = baseline_eq_wgt_df["date"].dt.normalize() # Keep only date part
        # --- End Baseline Calculation ---

        # Align dates - merge agent results with the calculated baseline cumulative return
        # Ensure account_value_df date is also UTC for merging (baseline data is already UTC)
        account_value_df["date"] = pd.to_datetime(account_value_df["date"], utc=True)
        # Normalize agent date as well before merging
        account_value_df["date"] = account_value_df["date"].dt.normalize() # Keep only date part
        merged_df = pd.merge(
            account_value_df, baseline_eq_wgt_df, on="date", how="inner"
        )

        if merged_df.empty:
            print(
                "Error: Merged DataFrame for Agent vs Baseline plot is empty. Skipping plot."
            )
            return

        # Calculate cumulative returns
        merged_df["agent_cumulative_return"] = (
            merged_df["account_value"] / merged_df["account_value"].iloc[0]
        ) - 1

        # --- Matplotlib Plotting ---
        fig, ax = plt.subplots(figsize=(15, 7))

        # Plot Agent Return
        ax.plot(
            merged_df["date"],
            merged_df["agent_cumulative_return"],
            label="Agent",
            linewidth=2,
        )

        # Plot Baseline Return
        ax.plot(
            merged_df["date"],
            merged_df["baseline_cumulative_return"],
            label="Equal-Weighted Baseline",
            linestyle="--",
            linewidth=2,
        )

        # Formatting
        ax.set_title("Agent Cumulative Return vs Equal-Weighted Baseline", fontsize=16)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Cumulative Return", fontsize=12)
        ax.legend(loc="upper left", fontsize=10)
        ax.grid(True)

        # Improve date formatting on x-axis
        fig.autofmt_xdate()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

        plt.tight_layout()

        # Define PNG filename within library-specific directory
        png_filename = os.path.join(
            lib_results_dir,
            f"backtest_plot_{drl_lib}_{model_name}_{start_date}_to_{end_date}_vs_equal_wgt.png",
        )

        # Save the figure
        plt.savefig(png_filename)
        print(f"Agent vs Baseline plot saved as PNG: {png_filename}")
        plt.close(fig)  # Close the figure

        # --- Plot and save the agent's account value (equity curve) ---
        fig2, ax2 = plt.subplots(figsize=(15, 7))
        ax2.plot(
            merged_df["date"],
            merged_df["account_value"],
            label="Agent Account Value",
            linewidth=2,
            color="tab:blue"
        )
        ax2.set_title("Agent Account Value Over Time", fontsize=16)
        ax2.set_xlabel("Date", fontsize=12)
        ax2.set_ylabel("Account Value", fontsize=12)
        ax2.legend(loc="upper left", fontsize=10)
        ax2.grid(True)
        fig2.autofmt_xdate()
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.tight_layout()
        # Save account value curve to library-specific directory
        account_value_png_filename = os.path.join(
            lib_results_dir,
            f"account_value_curve_{drl_lib}_{model_name}_{start_date}_to_{end_date}.png",
        )
        plt.savefig(account_value_png_filename)
        print(f"Agent account value curve saved as PNG: {account_value_png_filename}")
        plt.close(fig2)

    except Exception as plot_err:
        print(f"Error generating Agent vs Baseline plot: {plot_err}")


def plot_individual_prices000(
    baseline_all_tickers_df, results_dir, drl_lib, model_name, start_date, end_date
):
    """Generates a plot showing the closing prices of individual tickers."""
    print("Generating individual ticker performance plot...")
    try:
        matplotlib.rcdefaults()  # Reset rcParams to default
        plt.style.use("seaborn-v0_8-darkgrid")  # Explicitly set style
        # Pivot table to get closing prices with dates as index and tickers as columns
        # Ensure 'date' column exists and is datetime
        if "date" not in baseline_all_tickers_df.columns:
            baseline_all_tickers_df["date"] = pd.to_datetime(
                baseline_all_tickers_df["timestamp"], utc=True
            )

        baseline_pivot = baseline_all_tickers_df.pivot_table(
            index="date", columns="tic", values="close"
        )

        fig, ax = plt.subplots(figsize=(15, 7))

        # Plot closing price for each ticker using baseline_pivot
        for ticker in baseline_pivot.columns:
            ax.plot(baseline_pivot.index, baseline_pivot[ticker], label=ticker)

        # Formatting
        ax.set_title("Individual Ticker Closing Prices (Test Period)", fontsize=16)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Closing Price", fontsize=12)
        ax.legend(loc="upper left", fontsize="small")
        ax.grid(True)

        # Improve date formatting on x-axis
        fig.autofmt_xdate()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

        plt.tight_layout()

        # Define PNG filename
        individual_png_filename = os.path.join(
            results_dir,
            f"individual_ticker_prices_{drl_lib}_{model_name}_{start_date}_to_{end_date}.png",
        )

        # Save the figure
        plt.savefig(individual_png_filename)
        print(f"Individual ticker plot saved as PNG: {individual_png_filename}")
        plt.close(fig)  # Close the figure
        plt.style.use("default")  # Reset style after plotting

    except Exception as e:
        print(f"Error generating individual ticker plot: {e}")


# --- End Plotting Utility Functions ---

def plot_individual_prices(baseline_all_tickers_df, results_dir, drl_lib, model_name, start_date, end_date):
    """Generates a plot showing the closing prices of individual tickers, saved to library-specific directory."""
    print("Generating individual ticker performance plot...")
    # Create library-specific results directory if it doesn't exist (robustness)
    lib_results_dir = os.path.join(results_dir, drl_lib)
    os.makedirs(lib_results_dir, exist_ok=True)
    try:
        # Check for required columns
        required_columns = ['date', 'tic', 'close']
        if not all(col in baseline_all_tickers_df.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain columns: {required_columns}")

        # Ensure 'date' is in datetime format and timezone-aware (UTC)
        baseline_all_tickers_df['date'] = pd.to_datetime(baseline_all_tickers_df['date'], utc=True)

        # Convert start_date and end_date to datetime and make them timezone-aware (UTC)
        start_date_dt = pd.to_datetime(start_date, utc=True)
        end_date_dt = pd.to_datetime(end_date, utc=True)

        # --- Get definitive list of valid trading dates for the plot period ---
        print(f"Fetching raw data for plot period ({start_date} to {end_date}) to get valid trading dates...")
        # Assume get_data needs list of tickers - use unique tickers from input df
        tickers_in_df = baseline_all_tickers_df['tic'].unique().tolist()
        if not tickers_in_df:
             print("Warning: No tickers found in input DataFrame for fetching valid dates.")
             # Fallback: use the filtered data directly, might still have gaps
             valid_trading_dates = pd.to_datetime(baseline_all_tickers_df['date'].unique(), utc=True)
        else:
             raw_plot_period_data = get_data(
                 start_date=start_date, # Use original string dates for get_data
                 end_date=end_date,
                 ticker_list=tickers_in_df,
                 data_source="yahoofinance", # Placeholder, not used by DB fetch
                 time_interval="1D" # Placeholder, not used by DB fetch
             )
             if raw_plot_period_data.empty:
                 print("Warning: Failed to fetch raw data for plot period. Using dates from input df.")
                 valid_trading_dates = pd.to_datetime(baseline_all_tickers_df['date'].unique(), utc=True)
             else:
                 # Ensure the timestamp column is datetime
                 raw_plot_period_data['timestamp'] = pd.to_datetime(raw_plot_period_data['timestamp'], utc=True, errors='coerce')
                 raw_plot_period_data.dropna(subset=['timestamp'], inplace=True)
                 valid_trading_dates = pd.to_datetime(raw_plot_period_data['timestamp'].unique(), utc=True)
                 print(f"Found {len(valid_trading_dates)} unique valid trading dates for plot period.")
        # --- End fetching valid dates ---

        # Filter the input DataFrame (potentially from cache) to ONLY these valid dates AND the original date range
        print("Filtering input DataFrame to valid trading dates and specified range...")
        mask = (
            (baseline_all_tickers_df['date'] >= start_date_dt) &
            (baseline_all_tickers_df['date'] <= end_date_dt) &
            (baseline_all_tickers_df['date'].isin(valid_trading_dates))
        )
        df_filtered = baseline_all_tickers_df.loc[mask]

        # Handle empty data
        if df_filtered.empty:
            print(f"No data available for the date range {start_date} to {end_date}")
            return

        # Get unique tickers
        tickers = df_filtered['tic'].unique()

        # Set default Matplotlib style
        plt.style.use('default')

        # Create the line plot
        fig, ax = plt.subplots(figsize=(12, 6))
        print("Plotting data directly from filtered DataFrame...")
        for ticker in tickers:
            ticker_df = df_filtered[df_filtered['tic'] == ticker].sort_values('date')
            # Plot using the date column and close column directly
            ax.plot(ticker_df['date'], ticker_df['close'], label=ticker, linestyle='-')

        # Customize the plot
        ax.set_title(f"Individual Stock Prices from {start_date_dt.date()} to {end_date_dt.date()}") # Use datetime dates
        ax.set_xlabel("Date")
        ax.set_ylabel("Closing Price")
        ax.legend(loc='upper left')
        ax.grid(True)
        fig.autofmt_xdate()

        # Generate and save the plot to library-specific directory
        filename = f"{drl_lib}_{model_name}_{start_date_dt.date()}_to_{end_date_dt.date()}_individual_prices.png"
        save_path = os.path.join(lib_results_dir, filename)
        plt.savefig(save_path)
        plt.close()

        print(f"Individual ticker plot saved as PNG: {save_path}")

    except Exception as e:
        print(f"Error generating individual ticker plot: {e}")


def plot_individual_prices_styled(
    baseline_all_tickers_df, results_dir, drl_lib, model_name, start_date, end_date
):
    """
    Generates a plot showing the closing prices of individual tickers,
    styled to match the provided example image (white background, thick black border).
    """
    print("Generating styled individual ticker performance plot...")
    import matplotlib.dates as mdates

    try:
        # Data preparation (same as original)
        if "date" not in baseline_all_tickers_df.columns:
            baseline_all_tickers_df["date"] = pd.to_datetime(
                baseline_all_tickers_df["timestamp"], utc=True
            )

        baseline_pivot = baseline_all_tickers_df.pivot_table(
            index="date", columns="tic", values="close"
        )

        # Ensure the index is sorted and is datetime for proper line plotting
        baseline_pivot = baseline_pivot.sort_index()
        if not pd.api.types.is_datetime64_any_dtype(baseline_pivot.index):
            baseline_pivot.index = pd.to_datetime(baseline_pivot.index)

        # Plotting
        fig, ax = plt.subplots(figsize=(16, 8), facecolor="black")
        ax.set_facecolor("white")

        # Plot each ticker as a LINE (not bar, not area)
        for ticker in baseline_pivot.columns:
            ax.plot(
                baseline_pivot.index,
                baseline_pivot[ticker],
                label=ticker,
                linewidth=2,
                marker=None,  # No markers, just lines
                linestyle="-",
            )
        print("Plotted as line chart for tickers:", list(baseline_pivot.columns))

        # Title and labels
        ax.set_title("Stock Closing Prices During Backtest Period", fontsize=18, pad=15)
        ax.set_xlabel("Date", fontsize=14)
        ax.set_ylabel("Closing Price", fontsize=14)

        # Legend inside plot
        ax.legend(loc="upper left", fontsize=12, frameon=False)

        # Grid and ticks
        ax.grid(
            True, which="major", linestyle="-", linewidth=0.5, color="gray", alpha=0.3
        )
        ax.tick_params(axis="both", which="major", labelsize=12)

        # Date formatting
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate()

        # Add thick black border
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(6)

        plt.tight_layout()

        # Save the figure
        styled_png_filename = os.path.join(
            results_dir,
            f"styled_individual_ticker_prices_{drl_lib}_{model_name}_{start_date}_to_{end_date}.png",
        )
        plt.savefig(styled_png_filename, facecolor=fig.get_facecolor())
        print(f"Styled individual ticker plot saved as PNG: {styled_png_filename}")
        plt.close(fig)

    except Exception as e:
        print(f"Error generating styled individual ticker plot: {e}")


# --- Main Plotting Function (Orchestrator) ---
def build_plot(
    account_value_df,
    baseline_ticker_list,  # Corrected parameter name and position
    start_date,  # Correct position
    end_date,  # Correct position
    time_interval,
    results_dir="./results",
    drl_lib="unknown",
    model_name="unknown",
):
    """Generates backtesting plots and saves stats by orchestrating helper functions.
       Assumes results_dir is the BASE directory. Sub-functions handle lib-specific dirs."""
    print("--- Generating Plots and Stats ---")
    # Ensure the BASE results directory exists
    os.makedirs(results_dir, exist_ok=True)

    # Ensure DataFrame has 'date' and 'account_value' columns
    if account_value_df is None or not all(
        col in account_value_df.columns for col in ["date", "account_value"]
    ):
        print(
            "Error: Cannot build plot. account_value_df is None or missing required columns."
        )
        return

    if account_value_df.empty:
        print("Warning: account_value_df is empty. Skipping plot generation.")
        return

    # Make sure date column is datetime and sort
    account_value_df["date"] = pd.to_datetime(account_value_df["date"])
    account_value_df = account_value_df.sort_values(by="date").reset_index(drop=True)

    print(
        f"Plotting period: {account_value_df['date'].min()} to {account_value_df['date'].max()}"
    )
    print(
        f"Using baseline tickers for equal-weighted portfolio: {baseline_ticker_list}"
    )

    # 1. Save Backtest Stats
    save_performance_stats(
        account_value_df=account_value_df,
        results_dir=results_dir,
        drl_lib=drl_lib,
        model_name=model_name,
        start_date=start_date,  # Use original test start/end for filename consistency
        end_date=end_date,
    )

    # 2. Fetch Baseline Data for Plotting
    print("Fetching data for baseline tickers...")
    # Ensure baseline dates match account value dates exactly for plotting
    baseline_start_date = account_value_df.loc[0, "date"].strftime("%Y-%m-%d")
    baseline_end_date = account_value_df.loc[
        len(account_value_df) - 1, "date"
    ].strftime("%Y-%m-%d")

    try:
        baseline_all_tickers_df = get_data(
            start_date=baseline_start_date,
            end_date=baseline_end_date,
            ticker_list=baseline_ticker_list,
            data_source="yahoofinance",  # data_source arg is kept but not used by get_data for fetching
            time_interval=time_interval,
        )

        if baseline_all_tickers_df.empty:
            print("Error: Failed to fetch baseline ticker data. Cannot generate plots.")
            return

        # Ensure 'timestamp' is datetime and rename to 'date' for consistency
        if "timestamp" not in baseline_all_tickers_df.columns:
            print(
                "Error: Baseline data missing 'timestamp' column. Cannot generate plots."
            )
            return
        baseline_all_tickers_df["date"] = pd.to_datetime(
            baseline_all_tickers_df["timestamp"], utc=True
        )
        baseline_all_tickers_df = baseline_all_tickers_df.sort_values(
            by=["tic", "date"]
        )

        # 3. Generate Plots using fetched baseline data
        plot_agent_vs_baseline(
            account_value_df=account_value_df.copy(), # Pass a copy to avoid modification issues
            baseline_all_tickers_df=baseline_all_tickers_df.copy(),
            time_interval=time_interval,
            results_dir=results_dir,
            drl_lib=drl_lib,
            model_name=model_name,
            start_date=start_date, # Use original test start/end for filename consistency
            end_date=end_date
        )

        plot_individual_prices(
            baseline_all_tickers_df=baseline_all_tickers_df.copy(),
            results_dir=results_dir,
            drl_lib=drl_lib,
            model_name=model_name,
            start_date=start_date,  # Use original test start/end for filename consistency
            end_date=end_date,
        )

    except Exception as data_err:
        print(f"Error fetching or processing baseline data for plots: {data_err}")

    print("--- Plotting Complete ---")


# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(
        description="Run FinRL training, testing, or plotting."
    )
    parser.add_argument(
        "--step",
        type=str,
        default="all",
        choices=["get_data", "process_data", "train", "test", "plot", "test_plot", "all"], # Added test_plot
        help="Which step(s) to run.",
    )
    parser.add_argument(
        "--drl_lib",
        type=str,
        default="stable_baselines3",
        choices=["elegantrl", "rllib", "stable_baselines3"],
        help="DRL library to use.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="ppo",
        help="Name of the DRL model (e.g., ppo, sac, td3).",
    )
    parser.add_argument(
        "--env",
        default=StockTradingEnv,
        help="Trading environment class (default: StockTradingEnv).",
    )  # Keep default for now

    # Data Args
    parser.add_argument("--train_start_date", type=str, default=TRAIN_START_DATE)
    parser.add_argument("--train_end_date", type=str, default=TRAIN_END_DATE)
    parser.add_argument("--test_start_date", type=str, default=TEST_START_DATE)
    parser.add_argument("--test_end_date", type=str, default=TEST_END_DATE)
    parser.add_argument(
        "--ticker_list",
        nargs="+",
        default=None,
        help="List of stock tickers (defaults based on DRL lib).",
    )
    parser.add_argument("--data_source", type=str, default="yahoofinance")
    parser.add_argument("--time_interval", type=str, default="1D")
    parser.add_argument("--technical_indicators", nargs="+", default=INDICATORS)
    parser.add_argument("--if_vix", type=bool, default=True)

    # Training Args
    parser.add_argument(
        "--train_total_timesteps",
        type=float,
        default=None,  # Default to None, let function handle defaults
        help="Total timesteps for SB3 training (e.g., 1e4). Ignored if early stopping enabled.",
    )
    parser.add_argument(
        "--eval_freq", # Added argument for evaluation frequency
        type=int,
        default=10000, # Default evaluation frequency
        help="Evaluate the agent every n steps (used only if --enable_early_stopping is set).",
    )
    parser.add_argument(
        "--train_total_episodes",
        type=int,
        default=None,  # Default to None, let function handle defaults
        help="Total episodes for RLlib training (e.g., 30).",
    )
    parser.add_argument(
        "--train_break_step",
        type=float,
        default=None,  # Default to None, let function handle defaults
        help="Break step for eRL training (e.g., 1e5).",
    )
    parser.add_argument(
        "--train_cwd",
        type=str,
        default=None,
        help="Directory to save trained models (overrides default). Best model saved here if early stopping enabled.",
    )

    # Testing Args
    parser.add_argument(
        "--test_cwd",
        type=str,
        default=None,
        help="Directory/path to load trained models from (overrides default). Use 'best_model.zip' if early stopping was enabled during training.",
    )
    parser.add_argument(
        "--test_net_dimension",
        type=int,
        default=None,
        help="Net dimension for eRL testing (overrides default).",
    )
    parser.add_argument(
        "--enable_early_stopping",
        action="store_true", # Makes it a boolean flag
        help="Enable early stopping for SB3 based on evaluation reward.",
    )

    # Plotting Args (baseline ticker removed)
    # parser.add_argument("--plot_baseline_ticker", type=str, default="^DJI") # Removed
    # Directory for saving plots/stats
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument(
        "--raw_cache_dir",
        type=str,
        default=RAW_DATA_CACHE_DIR,
        help="Directory for raw data cache.",
    )
    parser.add_argument(
        "--processed_cache_dir",
        type=str,
        default=PROCESSED_DATA_CACHE_DIR,
        help="Directory for processed data cache.",
    )

    # Ray Args (for RLlib)
    parser.add_argument(
        "--ray_num_cpus", type=int, default=None, help="Number of CPUs for Ray to use."
    )

    args = parser.parse_args()

    # Create cache and results directories if they don't exist
    os.makedirs(args.raw_cache_dir, exist_ok=True)
    os.makedirs(args.processed_cache_dir, exist_ok=True)
    # Ensure the BASE results dir exists. Library-specific dirs created later.
    os.makedirs(args.results_dir, exist_ok=True)

    # Set default ticker list based on DRL library if not provided
    if args.ticker_list is None:
        args.ticker_list = DEFAULT_TICKER_LIST
        # if args.drl_lib == 'elegantrl':
        #     args.ticker_list = DOW_30_TICKER
        # else:
        #     args.ticker_list = DEFAULT_TICKER_LIST

    # Prepare kwargs for functions
    kwargs = {
        # CWDs are now handled more specifically within train/test steps
        "erl_params": ERL_PARAMS,
        "rllib_params": RLlib_PARAMS,
        "agent_params": AGENT_PARAMS_MAP.get(args.model_name, {}),
        # Pass command-line args for training steps/episodes/timesteps
        # If None, the run_train/run_test functions will use their internal defaults
        "break_step": args.train_break_step,
        "total_episodes": args.train_total_episodes,
        "total_timesteps": args.train_total_timesteps, # Note: SB3 ignores this if callback stops early
        "eval_freq": args.eval_freq, # Added eval_freq
        # Pass command-line args for testing net_dimension
        "test_net_dimension": args.test_net_dimension,  # For eRL test
        "ray_num_cpus": args.ray_num_cpus,
        "raw_cache_dir": args.raw_cache_dir,
        "processed_cache_dir": args.processed_cache_dir,
        "results_dir": args.results_dir,  # Pass results_dir
        # Pass explicit CWD overrides if provided. Crucially, pass None if not provided.
        "cwd": args.train_cwd,  # Base CWD override for training
        "test_cwd": args.test_cwd,  # CWD override for testing/loading
        "enable_early_stopping": args.enable_early_stopping, # Pass the flag
    }
    # Filter out None values *except* for 'cwd' and 'test_cwd' which signal user intent
    # Also keep eval_freq even if it's the default
    kwargs_filtered = {
        k: v for k, v in kwargs.items() if k in ["cwd", "test_cwd", "eval_freq", "enable_early_stopping"] or v is not None
    }

    # --- Execute Steps ---
    account_value_results = None
    test_data_df = None  # To store data df from test run for plotting
    test_ticker_list = None  # To store ticker list from test run for plotting

    if args.step in ["get_data", "all"]:
        print("\n=== Step: Get Data ===")
        # Example: Get training data (will cache)
        get_data(
            args.train_start_date,
            args.train_end_date,
            args.ticker_list,
            args.data_source,
            args.time_interval,
            cache_dir=args.raw_cache_dir,
            # Pass relevant args if get_data uses them (e.g., API keys via kwargs)
            **{
                k: v
                for k, v in kwargs_filtered.items()
                if k
                not in [
                    "cwd",
                    "test_cwd",
                    "break_step",
                    "total_episodes",
                    "total_timesteps",
                    "test_net_dimension",
                ]
            },
        )
        # Example: Get testing data (will cache)
        get_data(
            args.test_start_date,
            args.test_end_date,
            args.ticker_list,
            args.data_source,
            args.time_interval,
            cache_dir=args.raw_cache_dir,
            # Pass relevant args
            **{
                k: v
                for k, v in kwargs_filtered.items()
                if k
                not in [
                    "cwd",
                    "test_cwd",
                    "break_step",
                    "total_episodes",
                    "total_timesteps",
                    "test_net_dimension",
                ]
            },
        )

    if args.step in ["process_data", "all"]:
        print("\n=== Step: Process Data ===")
        # Example: Process training data (will cache)
        raw_train_data = get_data(
            args.train_start_date,
            args.train_end_date,
            args.ticker_list,
            args.data_source,
            args.time_interval,
            cache_dir=args.raw_cache_dir,
            # Pass relevant args
            **{
                k: v
                for k, v in kwargs_filtered.items()
                if k
                not in [
                    "cwd",
                    "test_cwd",
                    "break_step",
                    "total_episodes",
                    "total_timesteps",
                    "test_net_dimension",
                ]
            },
        )
        # Generate hash for raw data filename
        ticker_str_train = "_".join(sorted(args.ticker_list))
        raw_filename_base_train = f"{args.data_source}_{ticker_str_train}_{args.train_start_date}_{args.train_end_date}_{args.time_interval}_raw"
        raw_data_hash_train = hashlib.md5(raw_filename_base_train.encode()).hexdigest()
        process_data(
            data=raw_train_data,
            technical_indicator_list=args.technical_indicators,
            start_date=args.train_start_date,
            end_date=args.train_end_date,  # Pass dates
            if_vix=args.if_vix,
            cache_dir=args.processed_cache_dir,
            raw_data_hash=raw_data_hash_train,
            **kwargs_filtered,  # Pass filtered kwargs
        )

        # Example: Process testing data (will cache)
        raw_test_data = get_data(
            args.test_start_date,
            args.test_end_date,
            args.ticker_list,
            args.data_source,
            args.time_interval,
            cache_dir=args.raw_cache_dir,
            # Pass relevant args
            **{
                k: v
                for k, v in kwargs_filtered.items()
                if k
                not in [
                    "cwd",
                    "test_cwd",
                    "break_step",
                    "total_episodes",
                    "total_timesteps",
                    "test_net_dimension",
                ]
            },
        )
        # Generate hash for raw data filename
        ticker_str_test = "_".join(sorted(args.ticker_list))
        raw_filename_base_test = f"{args.data_source}_{ticker_str_test}_{args.test_start_date}_{args.test_end_date}_{args.time_interval}_raw"
        raw_data_hash_test = hashlib.md5(raw_filename_base_test.encode()).hexdigest()
        process_data(
            data=raw_test_data,
            technical_indicator_list=args.technical_indicators,
            start_date=args.test_start_date,
            end_date=args.test_end_date,  # Pass dates
            if_vix=args.if_vix,
            cache_dir=args.processed_cache_dir,
            raw_data_hash=raw_data_hash_test,
            **kwargs_filtered,  # Pass filtered kwargs
        )

    if args.step in ["train", "all"]:
        print("\n=== Step: Train Model ===")
        # run_train handles cwd logic internally now based on kwargs['cwd']
        run_train(
            start_date=args.train_start_date,
            end_date=args.train_end_date,
            ticker_list=args.ticker_list,
            data_source=args.data_source,
            time_interval=args.time_interval,
            technical_indicator_list=args.technical_indicators,
            drl_lib=args.drl_lib,
            env=args.env,
            model_name=args.model_name,
            if_vix=args.if_vix,
            **kwargs_filtered,  # Pass filtered kwargs
        )

    if args.step in ["test", "all"]:
        print("\n=== Step: Test Model ===")
        # run_test handles cwd logic internally based on kwargs['test_cwd']
        account_value_results, test_data_df, test_ticker_list = (
            run_test(  # Capture ticker_list
                start_date=args.test_start_date,
                end_date=args.test_end_date,
                ticker_list=args.ticker_list,  # Pass the list used for testing
                data_source=args.data_source,
                time_interval=args.time_interval,
                technical_indicator_list=args.technical_indicators,
                drl_lib=args.drl_lib,
                env=args.env,
                model_name=args.model_name,
                if_vix=args.if_vix,
                **kwargs_filtered,  # Pass filtered kwargs
            )
        )
        # If test wasn't run in 'all' step, use the default/arg ticker list for plotting baseline
        if test_ticker_list is None:
            test_ticker_list = args.ticker_list

    if args.step in ["plot", "all"]:
        print("\n=== Step: Build Plot ===")
        if account_value_results is None and args.step == "plot":
            print("Plotting requires test results. Running test step first...")
            # Rerun test step to get results
            account_value_results, test_data_df, test_ticker_list = (
                run_test(  # Capture ticker_list
                    start_date=args.test_start_date,
                    end_date=args.test_end_date,
                    ticker_list=args.ticker_list,  # Pass the list used for testing
                    data_source=args.data_source,
                    time_interval=args.time_interval,
                    technical_indicator_list=args.technical_indicators,
                    drl_lib=args.drl_lib,
                    env=args.env,
                    model_name=args.model_name,
                    if_vix=args.if_vix,
                    **kwargs_filtered,  # Pass filtered kwargs
                )
            )
            # If test wasn't run previously, use the default/arg ticker list for plotting baseline
            if test_ticker_list is None:
                test_ticker_list = args.ticker_list

        if account_value_results is not None and test_data_df is not None:
            # Prepare DataFrame for plotting
            # Ensure dates align; use dates from the test_data_df which corresponds to the price_array length

            # Determine the correct date column name ('timestamp' or 'date')
            if "timestamp" in test_data_df.columns:
                date_col_name = "timestamp"
            elif "date" in test_data_df.columns:
                date_col_name = "date"
                print("Warning: Using 'date' column from test_data_df for plotting.")
            else:
                print(
                    "Error: Neither 'timestamp' nor 'date' column found in test data_df for plotting."
                )
                # Use return instead of exit in case this is called as part of a larger process
                return  # Cannot proceed

            unique_plot_dates = test_data_df[date_col_name].unique()

            if len(account_value_results) > len(unique_plot_dates):
                print(
                    f"Warning: Length mismatch for plotting ({len(account_value_results)} vs {len(unique_plot_dates)}). Trimming account values."
                )
                # This usually happens because the environment runs one step extra for the final state
                account_value_results_plot = account_value_results[
                    0 : len(unique_plot_dates)
                ]
                dates_plot = unique_plot_dates
            elif len(account_value_results) < len(unique_plot_dates):
                print(
                    f"Warning: Length mismatch for plotting ({len(account_value_results)} vs {len(unique_plot_dates)}). Trimming dates."
                )
                dates_plot = unique_plot_dates[0 : len(account_value_results)]
                account_value_results_plot = account_value_results
            else:
                dates_plot = unique_plot_dates
                account_value_results_plot = account_value_results

            account_df = pd.DataFrame(
                {"date": dates_plot, "account_value": account_value_results_plot}
            )

            build_plot(
                account_value_df=account_df,
                baseline_ticker_list=test_ticker_list,  # Pass the list of tickers
                start_date=args.test_start_date,  # Use test dates for context
                end_date=args.test_end_date,
                time_interval=args.time_interval,
                results_dir=args.results_dir,
                drl_lib=args.drl_lib,  # Pass drl_lib
                model_name=args.model_name,  # Pass model_name
            )
        else:
            print("Cannot build plot: Missing account value results or test data.")

    elif args.step == "test_plot":
        print("\n=== Step: Test Plot Function (with dummy data) ===")
        # --- Test Data and Parameters (similar to test_plot_style.py) ---
        print("Running test plot step from within refactored_script.py...")

        # 1. Dummy DataFrame
        dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'])
        tickers = ['DUMMY_A', 'DUMMY_B']
        data = []
        for tic in tickers:
            prices = np.random.rand(len(dates)) * 50 + 75 # Random prices for dummy test
            for i, date in enumerate(dates):
                 data.append({'date': date, 'tic': tic, 'close': prices[i]})

        dummy_df = pd.DataFrame(data)
        print("\nDummy DataFrame created for test plot:")
        print(dummy_df.head())

        # 2. Dummy Parameters
        test_results_dir = "./results_test_in_main" # Use a different dir to avoid conflicts
        test_drl_lib = "maintestlib"
        test_model_name = "maintestmodel"
        test_start_date = "2023-01-01"
        test_end_date = "2023-01-05"

        print(f"\nTest Parameters:")
        print(f"  Results Dir: {test_results_dir}")
        print(f"  DRL Lib: {test_drl_lib}")
        print(f"  Model Name: {test_model_name}")
        print(f"  Start Date: {test_start_date}")
        print(f"  End Date: {test_end_date}")

        # 3. Call the function (ensure it uses the definition from *this* script)
        # It will save to results_test_in_main/maintestlib_maintestmodel_..._individual_prices.png
        plot_individual_prices(
            baseline_all_tickers_df=dummy_df,
            results_dir=test_results_dir,
            drl_lib=test_drl_lib,
            model_name=test_model_name,
            start_date=test_start_date,
            end_date=test_end_date
        )
        print("\nTest plot step finished.")
        # --- End Test Plot Step ---

    print("\n=== Script Finished ===")


if __name__ == "__main__":
    main()
