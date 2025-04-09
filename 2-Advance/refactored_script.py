#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import argparse
import os
import gym
from numpy import random as rd
import pickle
import hashlib

import torch
import ray

# from finrl import config # Assuming config might be needed later, commented out for now
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split

# from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv # Original env
# Using NP version as in original script
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent as DRLAgent_sb3
from finrl.agents.rllib.models import DRLAgent as DRLAgent_rllib
from finrl.agents.elegantrl.models import DRLAgent as DRLAgent_erl

from finrl.meta.data_processor import DataProcessor
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline

# from finrl.config_tickers import DOW_30_TICKER # Import DOW tickers

import plotly.graph_objs as go


# --- Constants ---
RAW_DATA_CACHE_DIR = "./data_cache/raw"
PROCESSED_DATA_CACHE_DIR = "./data_cache/processed"

# Default Dates (can be overridden by command-line args)
TRAIN_START_DATE = "2014-01-01"
TRAIN_END_DATE = "2020-07-30"
TEST_START_DATE = "2020-08-01"
TEST_END_DATE = "2021-10-01"

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


class StockTradingEnv(gym.Env):
    def __init__(
        self,
        config,
        initial_account=1e6,
        gamma=0.99,
        turbulence_thresh=99,
        min_stock_rate=0.1,
        max_stock=1e2,
        initial_capital=1e6,
        buy_cost_pct=1e-3,
        sell_cost_pct=1e-3,
        reward_scaling=2**-11,
        initial_stocks=None,
    ):
        price_ary = config["price_array"]
        tech_ary = config["tech_array"]
        turbulence_ary = config["turbulence_array"]
        if_train = config["if_train"]
        # n = price_ary.shape[0] # Unused variable n
        self.price_ary = price_ary.astype(np.float32)
        self.tech_ary = tech_ary.astype(np.float32)
        self.turbulence_ary = turbulence_ary

        self.tech_ary = self.tech_ary * 2**-7
        self.turbulence_bool = (turbulence_ary > turbulence_thresh).astype(np.float32)
        self.turbulence_ary = (
            self.sigmoid_sign(turbulence_ary, turbulence_thresh) * 2**-5
        ).astype(np.float32)

        stock_dim = self.price_ary.shape[1]
        self.gamma = gamma
        self.max_stock = max_stock
        self.min_stock_rate = min_stock_rate
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.initial_capital = initial_capital
        self.initial_stocks = (
            np.zeros(stock_dim, dtype=np.float32)
            if initial_stocks is None
            else initial_stocks
        )

        # reset()
        self.day = None
        self.amount = None
        self.stocks = None
        self.total_asset = None
        self.gamma_reward = None
        self.initial_total_asset = None

        # environment information
        self.env_name = "StockEnv"
        self.state_dim = 1 + 2 + 3 * stock_dim + self.tech_ary.shape[1]
        # amount + (turbulence, turbulence_bool) + (price, stock, cd) * stock_dim + tech_dim
        self.stocks_cd = None
        self.action_dim = stock_dim
        self.max_step = self.price_ary.shape[0] - 1
        self.if_train = if_train
        self.if_discrete = False
        # Target return; used for potential early stopping but not implemented here
        self.target_return = 5.0
        self.episode_return = 0.0

        self.observation_space = gym.spaces.Box(
            low=-3000, high=3000, shape=(self.state_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(self.action_dim,), dtype=np.float32
        )

    def reset(self):
        self.day = 0
        price = self.price_ary[self.day]

        if self.if_train:
            self.stocks = (
                self.initial_stocks + rd.randint(0, 64, size=self.initial_stocks.shape)
            ).astype(np.float32)
            self.stocks_cd = np.zeros_like(self.stocks)
            self.amount = (
                self.initial_capital * rd.uniform(0.95, 1.05)
                - (self.stocks * price).sum()
            )
        else:
            self.stocks = self.initial_stocks.astype(np.float32)
            self.stocks_cd = np.zeros_like(self.stocks)
            self.amount = self.initial_capital

        self.total_asset = self.amount + (self.stocks * price).sum()
        self.initial_total_asset = self.total_asset
        self.gamma_reward = 0.0
        return self.get_state(price)  # state

    def step(self, actions):
        actions = (actions * self.max_stock).astype(int)

        self.day += 1
        price = self.price_ary[self.day]
        self.stocks_cd += 1

        if self.day >= len(self.turbulence_bool):  # Add boundary check
            # Handle case where day exceeds turbulence data length (should ideally not happen if data aligns)
            print(
                f"Warning: Day index {self.day} exceeds turbulence data length {len(self.turbulence_bool)}. Assuming no turbulence."
            )
            is_turbulent = False
        else:
            is_turbulent = self.turbulence_bool[self.day]

        if not is_turbulent:
            min_action = int(self.max_stock * self.min_stock_rate)  # stock_cd
            for index in np.where(actions < -min_action)[0]:  # sell_index:
                if price[index] > 0:  # Sell only if current asset is > 0
                    sell_num_shares = min(self.stocks[index], -actions[index])
                    self.stocks[index] -= sell_num_shares
                    self.amount += (
                        price[index] * sell_num_shares * (1 - self.sell_cost_pct)
                    )
                    self.stocks_cd[index] = 0
            for index in np.where(actions > min_action)[0]:  # buy_index:
                # Buy only if the price is > 0 (no missing data in this particular date)
                if price[index] > 0:
                    buy_num_shares = min(self.amount // price[index], actions[index])
                    self.stocks[index] += buy_num_shares
                    self.amount -= (
                        price[index] * buy_num_shares * (1 + self.buy_cost_pct)
                    )
                    self.stocks_cd[index] = 0

        else:  # sell all when turbulence
            self.amount += (self.stocks * price).sum() * (1 - self.sell_cost_pct)
            self.stocks[:] = 0
            self.stocks_cd[:] = 0

        state = self.get_state(price)
        total_asset = self.amount + (self.stocks * price).sum()
        reward = (total_asset - self.total_asset) * self.reward_scaling
        self.total_asset = total_asset

        self.gamma_reward = self.gamma_reward * self.gamma + reward
        done = self.day == self.max_step
        if done:
            reward = self.gamma_reward
            self.episode_return = total_asset / self.initial_total_asset

        return state, reward, done, dict()

    def get_state(self, price):
        amount = np.array(max(self.amount, 1e4) * (2**-12), dtype=np.float32)
        scale = np.array(2**-6, dtype=np.float32)

        # Boundary checks for arrays accessed by self.day
        # Use min to avoid index out of bounds on the last step
        day_idx = min(self.day, self.max_step)

        turbulence_val = (
            self.turbulence_ary[day_idx] if day_idx < len(self.turbulence_ary) else 0.0
        )
        turbulence_bool_val = (
            self.turbulence_bool[day_idx]
            if day_idx < len(self.turbulence_bool)
            else 0.0
        )
        tech_val = (
            self.tech_ary[day_idx]
            if day_idx < len(self.tech_ary)
            else np.zeros(self.tech_ary.shape[1])
        )  # Default to zeros if index issue

        return np.hstack(
            (
                amount,
                turbulence_val,
                turbulence_bool_val,
                price * scale,
                self.stocks * scale,
                self.stocks_cd,
                tech_val,
            )
        )  # state.astype(np.float32)

    @staticmethod
    def sigmoid_sign(ary, thresh):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x * np.e)) - 0.5

        return sigmoid(ary / thresh) * thresh


# --- Data Functions ---


def get_data(
    start_date,
    end_date,
    ticker_list,
    data_source,
    time_interval,
    cache_dir=RAW_DATA_CACHE_DIR,
    **kwargs,
):
    """Downloads and cleans data, using cache if available."""
    os.makedirs(cache_dir, exist_ok=True)
    # Create a unique filename based on parameters
    ticker_str = "_".join(sorted(ticker_list))
    filename_base = (
        f"{data_source}_{ticker_str}_{start_date}_{end_date}_{time_interval}_raw"
    )
    # Use hash for potentially long filenames
    filename_hash = hashlib.md5(filename_base.encode()).hexdigest()
    cache_file = os.path.join(cache_dir, f"{filename_hash}.pkl")

    if os.path.exists(cache_file):
        print(f"Loading raw data from cache: {cache_file}")
        try:
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
            print("Raw data loaded from cache.")
            return data
        except Exception as e:
            print(
                f"Error loading raw data from cache {cache_file}: {e}. Re-downloading."
            )

    print(f"Fetching data from {data_source} for {ticker_list}...")
    DP = DataProcessor(data_source, **kwargs)
    data = DP.download_data(ticker_list, start_date, end_date, time_interval)
    data = DP.clean_data(data)
    print("Data fetched and cleaned.")

    # Save to cache
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(data, f)
        print(f"Raw data saved to cache: {cache_file}")
    except Exception as e:
        print(f"Error saving raw data to cache {cache_file}: {e}")

    return data


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
    """Adds technical indicators, VIX, and converts to arrays, using cache if available."""
    os.makedirs(cache_dir, exist_ok=True)

    # Create a unique filename
    indicator_str = "_".join(sorted(technical_indicator_list))
    vix_str = "vix" if if_vix else "novix"
    # Use hash of raw data combined with processing params for uniqueness
    if raw_data_hash is None:
        # Attempt to create a hash from the DataFrame if raw_data_hash wasn't passed
        try:
            # Ensure consistent hashing by sorting columns first
            data_to_hash = data.sort_index(axis=1)
            raw_data_hash = hashlib.md5(
                pd.util.hash_pandas_object(data_to_hash, index=True).values
            ).hexdigest()
        except Exception as hash_err:
            print(f"Error generating hash for raw data: {hash_err}")
            # Fallback if hashing fails
            raw_data_hash = f"unknown_raw_data_{rd.randint(10000)}"

    filename_base = f"{raw_data_hash}_{indicator_str}_{vix_str}_processed"
    filename_hash = hashlib.md5(filename_base.encode()).hexdigest()
    cache_file = os.path.join(cache_dir, f"{filename_hash}.pkl")

    if os.path.exists(cache_file):
        print(f"Loading processed data from cache: {cache_file}")
        try:
            with open(cache_file, "rb") as f:
                cached_data = pickle.load(f)
            print("Processed data loaded from cache.")
            # Return data df as well, assuming it was cached
            return (
                cached_data["price_array"],
                cached_data["tech_array"],
                cached_data["turbulence_array"],
                cached_data["data_df"],
            )
        except Exception as e:
            print(
                f"Error loading processed data from cache {cache_file}: {e}. Re-processing."
            )

    print("Processing data: adding indicators and converting to arrays...")
    processed_data = data.copy()  # Start with a copy of the raw data

    vix_successfully_added = False
    if if_vix:
        print("Attempting to add VIX data manually...")
        # --- Instantiate VIX DataProcessor ---
        # Create a copy of kwargs to avoid modifying the original dict passed down
        dp_kwargs = kwargs.copy()
        # Explicitly remove keys that might conflict with constructor arguments
        dp_kwargs.pop("start_date", None)
        dp_kwargs.pop("end_date", None)
        vix_dp = DataProcessor(
            data_source="yahoofinance",
            start_date=start_date,
            end_date=end_date,
            **dp_kwargs,
        )

        # --- VIX Data Caching and Download ---
        raw_cache_dir = kwargs.get(
            "raw_cache_dir", RAW_DATA_CACHE_DIR
        )  # Get raw cache dir
        time_interval = kwargs.get("time_interval", "1D")
        vix_cache_filename = f"vix_raw_{start_date}_{end_date}_{time_interval}.pkl"
        vix_cache_path = os.path.join(raw_cache_dir, vix_cache_filename)

        vix_df = None
        if os.path.exists(vix_cache_path):
            print(f"Loading raw VIX data from cache: {vix_cache_path}")
            try:
                with open(vix_cache_path, "rb") as f:
                    vix_df = pickle.load(f)
                print("Raw VIX data loaded from cache.")
            except Exception as e:
                print(
                    f"Error loading raw VIX data from cache {vix_cache_path}: {e}. Re-downloading."
                )
                vix_df = None  # Ensure download happens if cache load fails

        if vix_df is None:  # If not loaded from cache or cache load failed
            print("Downloading raw VIX data...")
            # Use the already created vix_dp instance
            vix_df = vix_dp.download_data(["^VIX"], start_date, end_date, time_interval)

            if vix_df is not None and not vix_df.empty:
                # Clean data immediately after download
                print("Cleaning freshly downloaded VIX data...")
                vix_df = vix_dp.clean_data(vix_df)  # MOVED CLEANING HERE
                # Diagnostic print
                print(f"Columns after clean_data: {vix_df.columns.tolist()}")
                # Diagnostic print
                print(f"Index type after clean_data: {type(vix_df.index)}")

                # Save the *cleaned* raw VIX data to cache
                try:
                    # Ensure dir exists
                    os.makedirs(raw_cache_dir, exist_ok=True)
                    with open(vix_cache_path, "wb") as f:
                        pickle.dump(vix_df, f)  # Save cleaned data
                    print(f"Cleaned raw VIX data saved to cache: {vix_cache_path}")
                except Exception as e:
                    print(
                        f"Error saving cleaned raw VIX data to cache {vix_cache_path}: {e}"
                    )
            else:
                print("VIX download failed or returned empty dataframe.")
                vix_df = (
                    pd.DataFrame()
                )  # Ensure vix_df is an empty df if download fails
        else:
            # If loaded from cache, assume it's already clean (or cleaning causes issues)
            print("Skipping VIX cleaning as data was loaded from cache.")

        # --- VIX Data is now either loaded from cache (assumed clean) or downloaded and cleaned ---
        # --- Merge VIX Data ---
        # --- VIX Merge Refined ---
        vix_df_indexed = None
        processed_data_indexed = None

        # Prepare VIX data
        if vix_df is not None and not vix_df.empty:
            if "timestamp" not in vix_df.columns:
                vix_df = vix_df.reset_index()
            if "timestamp" in vix_df.columns:
                try:
                    if not pd.api.types.is_datetime64_any_dtype(vix_df["timestamp"]):
                        vix_df["timestamp"] = pd.to_datetime(
                            vix_df["timestamp"], utc=True
                        )
                    elif vix_df["timestamp"].dt.tz is None:
                        vix_df["timestamp"] = vix_df["timestamp"].dt.tz_localize("UTC")
                    else:
                        vix_df["timestamp"] = vix_df["timestamp"].dt.tz_convert("UTC")
                    vix_df_indexed = vix_df.set_index("timestamp")
                except Exception as e:
                    print(f"Error processing VIX timestamp: {e}. Skipping VIX merge.")
            else:
                print(
                    "Warning: 'timestamp' column not found in VIX data. Skipping VIX merge."
                )

        # Prepare main data
        if "timestamp" not in processed_data.columns:
            processed_data = processed_data.reset_index()
        if "timestamp" in processed_data.columns:
            try:
                if not pd.api.types.is_datetime64_any_dtype(
                    processed_data["timestamp"]
                ):
                    processed_data["timestamp"] = pd.to_datetime(
                        processed_data["timestamp"], utc=True
                    )
                elif processed_data["timestamp"].dt.tz is None:
                    processed_data["timestamp"] = processed_data[
                        "timestamp"
                    ].dt.tz_localize("UTC")
                else:
                    processed_data["timestamp"] = processed_data[
                        "timestamp"
                    ].dt.tz_convert("UTC")
                processed_data_indexed = processed_data.set_index("timestamp")
            except Exception as e:  # Added except for the try starting line 376
                print(f"Error processing main data timestamp: {e}. Cannot merge VIX.")
                processed_data_indexed = None  # Ensure it's None if error occurs
        else:
            print(
                "CRITICAL WARNING: 'timestamp' column missing from main data. Cannot merge VIX."
            )

        # Perform merge if both prepared successfully
        if vix_df_indexed is not None and processed_data_indexed is not None:
            try:
                print("Attempting INNER join on timestamp index...")
                vix_to_merge = vix_df_indexed[["close"]].rename(
                    columns={"close": "vix"}
                )
                # Use original processed_data_indexed (main data) and join vix_to_merge
                merged_data = processed_data_indexed.join(vix_to_merge, how="inner")
                processed_data = (
                    merged_data.reset_index()
                )  # Reset index after successful merge
                vix_successfully_added = True
                print("VIX data merged successfully.")
            except Exception as e:
                print(f"Error during VIX merge: {e}. Proceeding without VIX.")
                processed_data = (
                    processed_data_indexed.reset_index()
                )  # Fallback to original data
                vix_successfully_added = False
        else:
            print("Skipping VIX merge due to data preparation issues.")
            if processed_data_indexed is not None:
                # Ensure index is reset if merge skipped
                processed_data = processed_data_indexed.reset_index()
            # If processed_data_indexed is None, processed_data is already the original copy
            vix_successfully_added = False
    # --- End VIX Merge Refined ---
    # Removed the misplaced except block from here

    # Add technical indicators using a generic processor instance
    DP_generic = DataProcessor(
        data_source="yahoofinance", start_date=start_date, end_date=end_date, **kwargs
    )
    if not processed_data.empty:
        processed_data = DP_generic.add_technical_indicator(
            processed_data, technical_indicator_list
        )
        print("Technical indicators added.")

        # ADDED: Ensure common date range across all tickers AFTER adding indicators
        if not processed_data.empty:
            print("Ensuring common date range post-indicators...")
            # Ensure 'timestamp' column exists and is datetime
            if "timestamp" not in processed_data.columns:
                print(
                    "Warning: 'timestamp' column missing before final date range alignment. Skipping."
                )
            else:
                # Standardize timestamp to UTC datetime
                try:
                    if not pd.api.types.is_datetime64_any_dtype(
                        processed_data["timestamp"]
                    ):
                        processed_data["timestamp"] = pd.to_datetime(
                            processed_data["timestamp"], utc=True
                        )
                    elif processed_data["timestamp"].dt.tz is None:
                        processed_data["timestamp"] = processed_data[
                            "timestamp"
                        ].dt.tz_localize("UTC")
                    else:
                        processed_data["timestamp"] = processed_data[
                            "timestamp"
                        ].dt.tz_convert("UTC")

                    # Find common date range based on 'timestamp'
                    start_dates = processed_data.groupby("tic")["timestamp"].min()
                    end_dates = processed_data.groupby("tic")["timestamp"].max()
                    max_start_date = start_dates.max()
                    min_end_date = end_dates.min()

                    # Filter the dataframe
                    processed_data = processed_data[
                        (processed_data["timestamp"] >= max_start_date)
                        & (processed_data["timestamp"] <= min_end_date)
                    ]
                    print(
                        f"Final common date range set: {max_start_date} to {min_end_date}"
                    )

                    # Check row counts after final alignment
                    # print("Checking row counts per ticker AFTER FINAL alignment:")
                    # print(processed_data.groupby("tic").size())

                except Exception as e:
                    print(
                        f"Error during final date alignment: {e}. Skipping alignment."
                    )
        # END ADDED
    else:
        print("Skipping technical indicators on empty dataframe.")
    print(f"Columns after adding indicators: {processed_data.columns.tolist()}")

    # Let df_to_array handle turbulence internally
    print("Calling df_to_array (will handle turbulence)...")

    # Ensure 'tic' column exists if dataframe is not empty
    if not processed_data.empty and "tic" not in processed_data.columns:
        print("CRITICAL WARNING: 'tic' column missing before df_to_array!")
        # Add a dummy 'tic' column if it's missing, otherwise df_to_array will fail
        # This might indicate a problem earlier in the data loading/cleaning
        # This is a fallback, ideally 'tic' should always be present
        processed_data["tic"] = "UNKNOWN"

    print(f"Columns just before df_to_array: {processed_data.columns.tolist()}")

    # Convert to array
    if processed_data.empty:
        print("Warning: Processed data is empty. Returning empty arrays.")
        num_tickers = (
            len(data["tic"].unique()) if "tic" in data.columns and not data.empty else 0
        )
        num_tech_indicators = len(technical_indicator_list)
        price_array = np.empty((0, num_tickers))
        tech_dim = num_tech_indicators + (1 if vix_successfully_added else 0)
        tech_array = np.empty((0, num_tickers * tech_dim))
        turbulence_array = np.empty((0,))
    else:
        # Ensure processed_data index is reset before calling df_to_array
        # This is crucial if the merge operation above changed the index
        # Use drop=True if index is not needed as column
        processed_data = processed_data.reset_index(drop=True)

        # ADDED: Re-verify common dates just before array conversion
        print("Re-verifying common dates before df_to_array...")
        # Use the DataFrame that will be passed to df_to_array
        df_to_align = processed_data
        final_common_dates = None
        if not df_to_align.empty and "tic" in df_to_align.columns and "timestamp" in df_to_align.columns:
            # Ensure timestamp column is datetime type for proper comparison
            if not pd.api.types.is_datetime64_any_dtype(df_to_align["timestamp"]):
                 try:
                     # Attempt conversion, assuming UTC is appropriate or timezone is consistent
                     df_to_align["timestamp"] = pd.to_datetime(df_to_align["timestamp"], utc=True)
                     print("Converted timestamp column to datetime for final alignment.")
                 except Exception as e:
                     print(f"Warning: Could not convert timestamp to datetime for final alignment: {e}")
                     # Proceed without alignment if conversion fails

            if pd.api.types.is_datetime64_any_dtype(df_to_align["timestamp"]): # Check again after conversion attempt
                unique_tickers = df_to_align["tic"].unique()
                if len(unique_tickers) > 0: # Proceed only if there are tickers
                    for tic in unique_tickers:
                        tic_dates = set(df_to_align[df_to_align["tic"] == tic]["timestamp"])
                        if final_common_dates is None:
                            final_common_dates = tic_dates
                        else:
                            final_common_dates.intersection_update(tic_dates)

                    if final_common_dates is not None and len(final_common_dates) < len(df_to_align['timestamp'].unique()):
                        print(f"Discrepancy found! Filtering processed_data to {len(final_common_dates)} common dates.")
                        # Filter the main dataframe 'processed_data' based on common dates found in 'df_to_align'
                        processed_data = processed_data[processed_data["timestamp"].isin(final_common_dates)].copy()
                        # Optional: Print new counts
                        # print("Checking row counts per ticker AFTER second alignment:")
                        # print(processed_data.groupby("tic").size())
                    elif final_common_dates is None or len(final_common_dates) == 0:
                         # This case handles when intersection results in empty set or was never initialized properly
                         print("Error: No common dates found across all tickers in final verification step!")
                         raise ValueError("No common dates found before df_to_array")
                    else:
                        print("Final date verification passed. All tickers seem aligned.")
                else:
                    print("Skipping final date alignment check (no tickers found).")
            else:
                print("Skipping final date alignment due to non-datetime timestamp column after conversion attempt.")
        else:
            print("Skipping final date alignment check (empty df or missing 'tic'/'timestamp' columns).")

        # Removed the stray try: statement that was here.
        try:
            # WORKAROUND: Rename 'vix' column to 'VIXY' if VIX was added,
            # as df_to_array/add_turbulence might expect 'VIXY'
            if vix_successfully_added and "vix" in processed_data.columns:
                print("Renaming 'vix' column to 'VIXY' for df_to_array compatibility.")
                processed_data = processed_data.rename(columns={"vix": "VIXY"})

            # df_to_array internally calls add_turbulence if needed
            # ADDED: Check row counts per ticker (moved inside try block)
            # print("Checking row counts per ticker before df_to_array:")
            # print(processed_data.groupby("tic").size())
            # END ADDED
            price_array, tech_array, turbulence_array = DP_generic.df_to_array(
                processed_data, vix_successfully_added
            )
            print("Data processed and converted to arrays.")
        except Exception as e:
            print(f"Error during df_to_array: {e}")
            # Fallback to empty arrays if conversion fails
            num_tickers = (
                len(processed_data["tic"].unique())
                if "tic" in processed_data.columns
                else 0
            )
            num_tech_indicators = len(technical_indicator_list)
            price_array = np.empty((0, num_tickers))
            tech_dim = num_tech_indicators + (1 if vix_successfully_added else 0)
            tech_array = np.empty((0, num_tickers * tech_dim))
            turbulence_array = np.empty((0,))
            # ADDED: Check row counts per ticker
            # print("Checking row counts per ticker before df_to_array:")
            # print(processed_data.groupby("tic").size())
            # END ADDED
            price_array, tech_array, turbulence_array = DP_generic.df_to_array(
                processed_data, vix_successfully_added
            )
            print("Data processed and converted to arrays.")
        except Exception as e:
            print(f"Error during df_to_array: {e}")
            # Fallback to empty arrays if conversion fails
            num_tickers = (
                len(processed_data["tic"].unique())
                if "tic" in processed_data.columns
                else 0
            )
            num_tech_indicators = len(technical_indicator_list)
            price_array = np.empty((0, num_tickers))
            tech_dim = num_tech_indicators + (1 if vix_successfully_added else 0)
            tech_array = np.empty((0, num_tickers * tech_dim))
            turbulence_array = np.empty((0,))

    # Save to cache (including arrays and the final processed dataframe)
    data_to_cache = {
        "price_array": price_array,
        "tech_array": tech_array,
        "turbulence_array": turbulence_array,
        "data_df": processed_data,  # Cache the final df
    }
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(data_to_cache, f)
        print(f"Processed data saved to cache: {cache_file}")
    except Exception as e:
        print(f"Error saving processed data to cache {cache_file}: {e}")

    # Return the processed df
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

    # Setup environment config
    env_config = {
        "price_array": price_array,
        "tech_array": tech_array,
        "turbulence_array": turbulence_array,
        "if_train": True,
    }
    env_instance = env(config=env_config)

    # Read DRL parameters
    cwd = kwargs.get("cwd", f"./trained_models/{drl_lib}_{model_name}")
    os.makedirs(cwd, exist_ok=True)  # Ensure directory exists

    print(f"--- Starting Training: {drl_lib} - {model_name} ---")
    print(f"Data: {start_date} to {end_date}")
    print(f"Tickers: {ticker_list}")
    print(f"Output directory: {cwd}")

    if drl_lib == "elegantrl":
        # Default from original script
        break_step = kwargs.get("break_step", 1e5)
        # Use default if not provided
        erl_params = kwargs.get("erl_params", ERL_PARAMS)

        agent = DRLAgent_erl(
            env=env,  # Pass class, not instance for eRL
            price_array=price_array,
            tech_array=tech_array,
            turbulence_array=turbulence_array,
        )

        model = agent.get_model(model_name, model_kwargs=erl_params)
        trained_model = agent.train_model(
            model=model, cwd=cwd, total_timesteps=break_step
        )
        # Note: eRL agent doesn't explicitly save the model in train_model, it happens internally

    elif drl_lib == "rllib":
        # Default from original script
        total_episodes = kwargs.get("total_episodes", 30)
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

        trained_model = agent_rllib.train_model(
            model=model,
            model_name=model_name,  # Pass model_name string
            model_config=model_config,
            total_episodes=total_episodes,
        )
        # Save the checkpoint
        checkpoint_path = trained_model.save(cwd)
        print(f"RLlib model checkpoint saved to: {checkpoint_path}")
        ray.shutdown()  # Shutdown Ray after training
        print("Ray session shut down.")

    elif drl_lib == "stable_baselines3":
        # Default from original script
        total_timesteps = kwargs.get("total_timesteps", 1e4)
        # Get agent params based on model_name, allow override via kwargs
        agent_params = kwargs.get("agent_params", AGENT_PARAMS_MAP.get(model_name, {}))

        agent = DRLAgent_sb3(env=env_instance)  # Pass instance for SB3

        model = agent.get_model(model_name, model_kwargs=agent_params)
        trained_model = agent.train_model(
            model=model, tb_log_name=model_name, total_timesteps=total_timesteps
        )
        print("SB3 Training finished!")
        save_path = os.path.join(cwd, f"{model_name}.zip")
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

    # Check if arrays are empty (e.g., due to processing error)
    if price_array.size == 0 or tech_array.size == 0 or turbulence_array.size == 0:
        print(
            "Error: Data processing resulted in empty arrays. Cannot proceed with testing."
        )
        return None, None  # Return None to indicate failure

    # Setup environment config for testing
    env_config = {
        "price_array": price_array,
        "tech_array": tech_array,
        "turbulence_array": turbulence_array,
        "if_train": False,
    }  # Set if_train to False
    env_instance = env(config=env_config)

    # Load parameters
    # Default cwd based on training output structure
    default_cwd = f"./trained_models/{drl_lib}_{model_name}"
    cwd = kwargs.get("cwd", default_cwd)
    # Specific path adjustments for different libraries if needed
    if drl_lib == "rllib":
        # RLlib needs path to specific checkpoint, not just the directory
        # Assume latest checkpoint if not specified, or require exact path via kwargs
        # This part might need refinement based on how checkpoints are saved/named
        # For now, assume 'cwd' passed via kwargs points to the correct checkpoint dir/file
        agent_path = cwd  # Requires user to provide correct checkpoint path
        print(f"Using RLlib agent path: {agent_path}")
    elif drl_lib == "stable_baselines3":
        # SB3 usually saves as a .zip file
        if not cwd.endswith(".zip"):
            # Assume standard naming convention if directory is given
            potential_path = os.path.join(cwd, f"{model_name}.zip")
            if os.path.exists(potential_path):
                cwd = potential_path
            else:
                # Fallback or raise error if zip not found
                print(
                    f"Warning: SB3 model .zip file not found at {potential_path}. Trying directory path {cwd}, but DRL_prediction might fail."
                )
        print(f"Using SB3 model path: {cwd}")
    else:  # elegantrl
        print(f"Using eRL model directory: {cwd}")

    print(f"--- Starting Testing: {drl_lib} - {model_name} ---")
    print(f"Data: {start_date} to {end_date}")
    print(f"Tickers: {ticker_list}")
    print(f"Loading model from: {cwd}")

    # Load elegantrl needs state dim, action dim and net dim
    # Use default from ERL_PARAMS
    net_dimension = kwargs.get("net_dimension", ERL_PARAMS["net_dimension"])

    episode_total_assets = None

    if drl_lib == "elegantrl":
        episode_total_assets = DRLAgent_erl.DRL_prediction(
            model_name=model_name,
            cwd=cwd,
            net_dimension=net_dimension,
            environment=env_instance,
        )

    elif drl_lib == "rllib":
        # Ensure Ray is initialized (and shutdown previous session if needed)
        if ray.is_initialized():
            print("Shutting down previous Ray session...")
            ray.shutdown()
        print("Initializing Ray for prediction...")
        ray.init(ignore_reinit_error=True, num_cpus=kwargs.get("ray_num_cpus", None))

        episode_total_assets = DRLAgent_rllib.DRL_prediction(
            model_name=model_name,  # Pass model name string
            env=env,  # Pass env class
            price_array=price_array,
            tech_array=tech_array,
            turbulence_array=turbulence_array,
            agent_path=agent_path,
        )  # Use the determined agent path
        ray.shutdown()  # Shutdown Ray after prediction
        print("Ray session shut down.")

    elif drl_lib == "stable_baselines3":
        episode_total_assets = DRLAgent_sb3.DRL_prediction_load_from_file(
            model_name=model_name,  # Pass model name string
            environment=env_instance,  # Pass env instance
            cwd=cwd,
        )  # Use the determined model path

    else:
        raise ValueError("DRL library input is NOT supported. Please check.")

    print(f"--- Testing Complete: {drl_lib} - {model_name} ---")
    if episode_total_assets is not None:
        print(f"Test period length: {len(episode_total_assets)}")
    else:
        print("Warning: episode_total_assets is None.")

    return episode_total_assets, data_df  # Return data_df for baseline plotting


# --- Plotting Function ---


def build_plot(
    account_value_df,
    baseline_ticker,
    start_date,
    end_date,
    time_interval,
    results_dir="./results",
):
    """Generates backtesting plots and saves stats."""
    print("--- Generating Plots and Stats ---")
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

    # Make sure date column is datetime
    account_value_df["date"] = pd.to_datetime(account_value_df["date"])
    account_value_df = account_value_df.sort_values(by="date").reset_index(drop=True)

    print(
        f"Plotting period: {account_value_df['date'].min()} to {account_value_df['date'].max()}"
    )
    print(f"Using baseline ticker: {baseline_ticker}")

    # 1. Backtest Stats
    print("Calculating backtest stats...")
    try:
        perf_stats_all = backtest_stats(account_value=account_value_df)
        perf_stats_all_df = pd.DataFrame(perf_stats_all)
        stats_filename = os.path.join(
            results_dir,
            f"perf_stats_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.csv",
        )
        perf_stats_all_df.to_csv(stats_filename)
        print(f"Performance stats saved to: {stats_filename}")
        print(perf_stats_all_df)
    except Exception as e:
        print(f"Error calculating backtest stats: {e}")

    # 2. Backtest Plot vs Baseline
    print("Generating backtest plot...")
    # Ensure baseline dates match account value dates exactly for plotting
    baseline_start_date = account_value_df.loc[0, "date"].strftime("%Y-%m-%d")
    baseline_end_date = account_value_df.loc[
        len(account_value_df) - 1, "date"
    ].strftime("%Y-%m-%d")

    # Removed the try block calling finrl.plot.backtest_plot to ensure
    # only the manual method using get_data (with caching) is used.
    print("Generating plot manually using Plotly (ensures baseline caching)...")
    try:
            # Fetch baseline data manually for comparison
            baseline_df = get_data(
                baseline_start_date,
                baseline_end_date,
                [baseline_ticker],
                "yahoofinance",
                time_interval,
            )
            # ADDED: Rename 'timestamp' to 'date' if necessary
            if "timestamp" in baseline_df.columns and "date" not in baseline_df.columns:
                print("Renaming 'timestamp' column to 'date' in baseline_df for plotting.")
                baseline_df = baseline_df.rename(columns={"timestamp": "date"})
            elif "date" not in baseline_df.columns:
                 print("Error: Baseline DataFrame missing 'date' or 'timestamp' column.")
                 # Handle error appropriately, maybe skip plotting or raise exception
                 raise KeyError("Baseline DataFrame missing date/timestamp column needed for plotting.")

            # Ensure the 'date' column exists before proceeding
            if "date" in baseline_df.columns:
                baseline_df["date"] = pd.to_datetime(baseline_df["date"], utc=True)
                baseline_df = baseline_df.sort_values(by="date").reset_index(drop=True)
            else:
                # This case should ideally be caught by the check above, but as a safeguard:
                print("Error: 'date' column still missing after checks. Cannot proceed with plotting.")
                raise KeyError("'date' column missing in baseline_df")

            # Align dates - merge ensures only common dates are kept
            merged_df = pd.merge(
                account_value_df, baseline_df[["date", "close"]], on="date", how="inner"
            )

            if merged_df.empty:
                print(
                    "Error: Merged DataFrame for plotting is empty. Cannot generate plot."
                )
                return

            # Calculate cumulative returns
            merged_df["agent_cumulative_return"] = (
                merged_df["account_value"] / merged_df["account_value"].iloc[0]
            ) - 1
            merged_df["baseline_cumulative_return"] = (
                merged_df["close"] / merged_df["close"].iloc[0]
            ) - 1

            # Create Plotly figure
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=merged_df["date"],
                    y=merged_df["agent_cumulative_return"],
                    mode="lines",
                    name="Agent",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=merged_df["date"],
                    y=merged_df["baseline_cumulative_return"],
                    mode="lines",
                    name=baseline_ticker,
                )
            )

            fig.update_layout(
                title="Agent Cumulative Return vs Baseline",
                xaxis_title="Date",
                yaxis_title="Cumulative Return",
                legend_title="Legend",
                paper_bgcolor="rgba(1,1,0,0)",
                plot_bgcolor="rgba(1, 1, 0, 0)",
            )
            fig.update_xaxes(
                showline=True,
                linecolor="black",
                showgrid=True,
                gridwidth=1,
                gridcolor="LightSteelBlue",
                mirror=True,
            )
            fig.update_yaxes(
                showline=True,
                linecolor="black",
                showgrid=True,
                gridwidth=1,
                gridcolor="LightSteelBlue",
                mirror=True,
            )
            fig.update_yaxes(
                zeroline=True, zerolinewidth=1, zerolinecolor="LightSteelBlue"
            )

            plotly_filename = os.path.join(
                results_dir,
                f"backtest_plotly_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.html",
            )
            fig.write_html(plotly_filename)
            print(f"Plotly plot saved to: {plotly_filename}")
            # fig.show() # Don't show automatically

    except Exception as plot_err:
        print(f"Error generating manual Plotly plot: {plot_err}")

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
        choices=["get_data", "process_data", "train", "test", "plot", "all"],
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
        default=None,
        help="Total timesteps for SB3 training.",
    )
    parser.add_argument(
        "--train_total_episodes",
        type=int,
        default=None,
        help="Total episodes for RLlib training.",
    )
    parser.add_argument(
        "--train_break_step",
        type=float,
        default=None,
        help="Break step for eRL training.",
    )
    parser.add_argument(
        "--train_cwd", type=str, default=None, help="Directory to save trained models."
    )

    # Testing Args
    parser.add_argument(
        "--test_cwd",
        type=str,
        default=None,
        help="Directory/path to load trained models from.",
    )
    parser.add_argument(
        "--test_net_dimension",
        type=int,
        default=None,
        help="Net dimension for eRL testing.",
    )

    # Plotting Args
    parser.add_argument("--plot_baseline_ticker", type=str, default="^DJI")
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

    # Create cache directories if they don't exist
    os.makedirs(args.raw_cache_dir, exist_ok=True)
    os.makedirs(args.processed_cache_dir, exist_ok=True)

    # Set default ticker list based on DRL library if not provided
    if args.ticker_list is None:
        args.ticker_list = DEFAULT_TICKER_LIST
        # if args.drl_lib == 'elegantrl':
        #     args.ticker_list = DOW_30_TICKER
        # else:
        #     args.ticker_list = DEFAULT_TICKER_LIST

    # Prepare kwargs for functions
    kwargs = {
        "cwd": args.train_cwd,  # For training
        "erl_params": ERL_PARAMS,  # Allow potential future override via args
        "rllib_params": RLlib_PARAMS,  # Allow potential future override via args
        "agent_params": AGENT_PARAMS_MAP.get(args.model_name, {}),  # For SB3
        "break_step": args.train_break_step,  # For eRL train
        "total_episodes": args.train_total_episodes,  # For RLlib train
        "total_timesteps": args.train_total_timesteps,  # For SB3 train
        "net_dimension": args.test_net_dimension,  # For eRL test
        "ray_num_cpus": args.ray_num_cpus,  # For RLlib train/test
        "raw_cache_dir": args.raw_cache_dir,  # Pass cache dirs
        "processed_cache_dir": args.processed_cache_dir,
        # 'time_interval' removed - passed explicitly where needed
    }
    # Filter out None values from kwargs to avoid overriding defaults unnecessarily
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    # --- Execute Steps ---
    account_value_results = None
    test_data_df = None  # To store data df from test run for plotting

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
        )
        # Example: Get testing data (will cache)
        get_data(
            args.test_start_date,
            args.test_end_date,
            args.ticker_list,
            args.data_source,
            args.time_interval,
            cache_dir=args.raw_cache_dir,
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
            **kwargs,
        )

        # Example: Process testing data (will cache)
        raw_test_data = get_data(
            args.test_start_date,
            args.test_end_date,
            args.ticker_list,
            args.data_source,
            args.time_interval,
            cache_dir=args.raw_cache_dir,
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
            **kwargs,
        )

    if args.step in ["train", "all"]:
        print("\n=== Step: Train Model ===")
        # Update kwargs with specific training CWD if provided
        train_kwargs = kwargs.copy()
        if args.train_cwd:
            train_kwargs["cwd"] = args.train_cwd
        else:  # Default CWD if not specified
            train_kwargs["cwd"] = f"./trained_models/{args.drl_lib}_{args.model_name}"

        # Ensure time_interval is not passed twice (explicitly and via kwargs)
        train_kwargs.pop("time_interval", None)
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
            **train_kwargs,
        )  # Pass potentially updated kwargs

    if args.step in ["test", "all"]:
        print("\n=== Step: Test Model ===")
        # Update kwargs with specific testing CWD if provided
        test_kwargs = kwargs.copy()
        if args.test_cwd:
            test_kwargs["cwd"] = args.test_cwd
        # Default CWD if not specified (points to default training output)
        else:
            test_kwargs["cwd"] = f"./trained_models/{args.drl_lib}_{args.model_name}"
            # Special handling for RLlib checkpoint path might be needed here if default is used
            if args.drl_lib == "rllib":
                print(
                    f"Warning: Using default test CWD for RLlib: {test_kwargs['cwd']}. Ensure this directory contains the correct checkpoint structure or provide --test_cwd."
                )
            elif args.drl_lib == "stable_baselines3":
                # Check if default zip exists
                potential_path = os.path.join(
                    test_kwargs["cwd"], f"{args.model_name}.zip"
                )
                if os.path.exists(potential_path):
                    test_kwargs["cwd"] = potential_path
                else:
                    print(
                        f"Warning: Default SB3 model {potential_path} not found. Using directory {test_kwargs['cwd']}."
                    )

        account_value_results, test_data_df = run_test(
            start_date=args.test_start_date,
            end_date=args.test_end_date,
            ticker_list=args.ticker_list,
            data_source=args.data_source,
            time_interval=args.time_interval,
            technical_indicator_list=args.technical_indicators,
            drl_lib=args.drl_lib,
            env=args.env,
            model_name=args.model_name,
            if_vix=args.if_vix,
            **test_kwargs,
        )  # Pass potentially updated kwargs

    if args.step in ["plot", "all"]:
        print("\n=== Step: Build Plot ===")
        if account_value_results is None and args.step == "plot":
            print("Plotting requires test results. Running test step first...")
            # Update kwargs with specific testing CWD if provided
            test_kwargs = kwargs.copy()
            if args.test_cwd:
                test_kwargs["cwd"] = args.test_cwd
            else:  # Default CWD if not specified
                test_kwargs["cwd"] = (
                    f"./trained_models/{args.drl_lib}_{args.model_name}"
                )
                # Add path checks similar to the 'test' step above
                if args.drl_lib == "rllib":
                    print(
                        f"Warning: Using default test CWD for RLlib: {test_kwargs['cwd']}. Ensure this directory contains the correct checkpoint structure or provide --test_cwd."
                    )
                elif args.drl_lib == "stable_baselines3":
                    potential_path = os.path.join(
                        test_kwargs["cwd"], f"{args.model_name}.zip"
                    )
                    if os.path.exists(potential_path):
                        test_kwargs["cwd"] = potential_path
                    else:
                        print(
                            f"Warning: Default SB3 model {potential_path} not found. Using directory {test_kwargs['cwd']}."
                        )

            account_value_results, test_data_df = run_test(
                start_date=args.test_start_date,
                end_date=args.test_end_date,
                ticker_list=args.ticker_list,
                data_source=args.data_source,
                time_interval=args.time_interval,
                technical_indicator_list=args.technical_indicators,
                drl_lib=args.drl_lib,
                env=args.env,
                model_name=args.model_name,
                if_vix=args.if_vix,
                **test_kwargs,
            )

        if account_value_results is not None and test_data_df is not None:
            # Prepare DataFrame for plotting
            # Ensure dates align; use dates from the test_data_df which corresponds to the price_array length
            if len(account_value_results) > len(test_data_df.timestamp.unique()):
                print(
                    f"Warning: Length mismatch between account values ({len(account_value_results)}) and unique dates in test data ({len(test_data_df.date.unique())}). Trimming account values."
                )
                # This usually happens because the environment runs one step extra for the final state
                account_value_results = account_value_results[
                    0 : len(test_data_df.date.unique())
                ]
            elif len(account_value_results) < len(test_data_df.timestamp.unique()):
                print(
                    f"Warning: Length mismatch between account values ({len(account_value_results)}) and unique dates in test data ({len(test_data_df.date.unique())}). Trimming dates."
                )
                dates = test_data_df.date.unique()[0 : len(account_value_results)]
            else:
                dates = test_data_df.timestamp.unique()

            account_df = pd.DataFrame(
                {"date": dates, "account_value": account_value_results}
            )

            build_plot(
                account_value_df=account_df,
                baseline_ticker=args.plot_baseline_ticker,
                start_date=args.test_start_date,  # Use test dates for context
                end_date=args.test_end_date,
                time_interval=args.time_interval,
                results_dir=args.results_dir,
            )
        else:
            print("Cannot build plot: Missing account value results or test data.")

    print("\n=== Script Finished ===")


if __name__ == "__main__":
    main()
