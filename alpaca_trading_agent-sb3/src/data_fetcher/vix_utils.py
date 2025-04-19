# alpaca_trading_agent/src/data_fetcher/vix_utils.py

import pandas as pd
import yfinance as yf
import os
import logging
from datetime import datetime, timedelta

VIX_TICKER = "^VIX"
# Determine project root based on this file's location
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
VIX_CACHE_FILE = os.path.join(DATA_DIR, "vix_data.csv")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)


def get_vix_data(start_date_str: str, end_date_str: str) -> pd.DataFrame:
    """
    Fetches VIX data, using a cache file if possible.

    Args:
        start_date_str: Start date in 'YYYY-MM-DD' format.
        end_date_str: End date in 'YYYY-MM-DD' format.

    Returns:
        pandas.DataFrame: DataFrame with 'date' (index, timezone-naive) and 'vix' column.
                         Returns an empty DataFrame on failure.
    """
    logging.info(f"Attempting to get VIX data from {start_date_str} to {end_date_str}")
    vix_df = pd.DataFrame()
    required_start_dt = pd.to_datetime(start_date_str)
    required_end_dt = pd.to_datetime(end_date_str)
    cache_exists = os.path.exists(VIX_CACHE_FILE)
    cache_valid = False

    if cache_exists:
        try:
            logging.info(f"Loading VIX data from cache: {VIX_CACHE_FILE}")
            # Load using 'Date' as the index column, expecting a clean CSV saved below
            cached_vix = pd.read_csv(VIX_CACHE_FILE, index_col="Date", parse_dates=True)

            # Ensure index is DatetimeIndex and name it 'date'
            if not isinstance(cached_vix.index, pd.DatetimeIndex):
                cached_vix.index = pd.to_datetime(cached_vix.index)
            cached_vix.index.name = "date"  # Standardize index name

            cached_start = cached_vix.index.min()
            cached_end = cached_vix.index.max()
            logging.debug(
                f"Cache date range: {cached_start.date()} to {cached_end.date()}"
            )

            # Check if cache covers the required date range and is reasonably recent
            if cached_start <= required_start_dt and cached_end >= (
                required_end_dt - timedelta(days=1)
            ):
                logging.info("VIX cache covers the required date range.")
                cache_valid = True
                # Filter cache for the required range *before* returning
                vix_df = cached_vix[
                    (cached_vix.index >= required_start_dt)
                    & (cached_vix.index <= required_end_dt)
                ].copy()
            else:
                logging.warning(
                    "VIX cache does not cover the required date range or is outdated. Will download fresh data."
                )
        except Exception as e:
            logging.error(
                f"Error loading VIX cache file: {e}. Will download fresh data.",
                exc_info=False,
            )
            cache_valid = (
                False  # Force download if cache is corrupted or format is wrong
            )

    if not cache_valid:
        logging.info(f"Downloading VIX data ({VIX_TICKER}) using yfinance...")
        try:
            # Download data
            download_start = (required_start_dt - timedelta(days=5)).strftime(
                "%Y-%m-%d"
            )
            download_end = (required_end_dt + timedelta(days=1)).strftime("%Y-%m-%d")
            vix_data = yf.download(
                VIX_TICKER, start=download_start, end=download_end, progress=False
            )

            if vix_data.empty:
                logging.error(f"Failed to download VIX data for {VIX_TICKER}.")
                return pd.DataFrame()

            # --- Handle Potential MultiIndex Columns from yf.download ---
            if isinstance(vix_data.columns, pd.MultiIndex):
                logging.debug(
                    "Downloaded VIX data has MultiIndex columns. Flattening..."
                )
                vix_data.columns = [
                    (
                        "_".join(map(str, col)).strip("_")
                        if isinstance(col, tuple)
                        else col
                    )
                    for col in vix_data.columns.values
                ]
                logging.debug(f"Columns after flattening: {vix_data.columns.tolist()}")

            # Ensure index is DatetimeIndex and name it 'Date' initially
            vix_data.index.name = "Date"
            if not isinstance(vix_data.index, pd.DatetimeIndex):
                vix_data.index = pd.to_datetime(vix_data.index)

            # Select the correct 'Close' column (potentially prefixed) and rename to 'vix'
            close_col_name = f"Close_{VIX_TICKER}"  # Adjusted column name
            if close_col_name not in vix_data.columns:
                # Fallback check in case yfinance changes behavior or prefix is different
                fallback_close_col = next(
                    (col for col in vix_data.columns if col.startswith("Close")), None
                )
                if fallback_close_col:
                    logging.warning(
                        f"Could not find exact column '{close_col_name}'. Using fallback: '{fallback_close_col}'"
                    )
                    close_col_name = fallback_close_col
                else:
                    logging.error(
                        f"Neither '{close_col_name}' nor any column starting with 'Close' found in downloaded VIX data. Columns: {vix_data.columns.tolist()}"
                    )
                    return pd.DataFrame()

            vix_df_full = vix_data[[close_col_name]].rename(
                columns={close_col_name: "vix"}
            )

            # Fill any missing values
            vix_df_full = vix_df_full.ffill().bfill()

            logging.info(
                f"Successfully processed downloaded VIX data. Shape: {vix_df_full.shape}"
            )

            # --- Explicit Column Cache Saving (Corrected) ---
            try:
                # Ensure index is named 'Date' before reset
                vix_df_full.index.name = "Date"
                # Reset index TO MAKE 'Date' A COLUMN
                vix_df_to_save = vix_df_full.reset_index()
                logging.debug(
                    f"Columns before saving cache: {vix_df_to_save.columns.tolist()}"
                )  # Should be ['Date', 'vix']
                # Save ONLY the 'Date' and 'vix' columns, explicitly without pandas index
                vix_df_to_save[["Date", "vix"]].to_csv(VIX_CACHE_FILE, index=False)
                logging.info(
                    f"Saved fresh VIX data to cache: {VIX_CACHE_FILE} with explicit columns ['Date', 'vix']"
                )
            except Exception as e:
                logging.error(f"Failed to save VIX data to cache: {e}", exc_info=True)
            # --- End Explicit Column Cache Saving ---

            # Filter for the originally required date range *after* saving full cache
            vix_df = vix_df_full[
                (vix_df_full.index >= required_start_dt)
                & (vix_df_full.index <= required_end_dt)
            ].copy()
            # Rename index AFTER filtering, just before returning
            vix_df.index.name = "date"

        except Exception as e:
            logging.error(
                f"An error occurred during VIX data download/processing: {e}",
                exc_info=True,
            )
            return pd.DataFrame()  # Return empty DataFrame on failure

    if vix_df.empty:
        logging.warning("Returning empty DataFrame for VIX data.")
    else:
        logging.info(f"Returning VIX data. Shape: {vix_df.shape}")
        # Ensure the final returned DataFrame index is timezone-naive 'date'
        if not isinstance(vix_df.index, pd.DatetimeIndex):
            vix_df.index = pd.to_datetime(vix_df.index)
        if vix_df.index.tz is not None:
            vix_df.index = vix_df.index.tz_localize(None)
        vix_df.index.name = "date"  # Ensure index name is 'date'

    return vix_df


if __name__ == "__main__":
    # Example usage:
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    start = "2023-01-01"
    end = "2023-12-31"

    # Test cache miss
    if os.path.exists(VIX_CACHE_FILE):
        os.remove(VIX_CACHE_FILE)
        logging.info("Removed existing cache file for test.")

    print("\n--- Attempt 1: Cache Miss ---")
    vix1 = get_vix_data(start, end)
    if not vix1.empty:
        print(vix1.head())
    else:
        print("Failed")

    # Test cache hit
    print("\n--- Attempt 2: Cache Hit ---")
    vix2 = get_vix_data(start, end)
    if not vix2.empty:
        print(vix2.head())
    else:
        print("Failed")

    # Check cache file content after hit
    if os.path.exists(VIX_CACHE_FILE):
        print("\n--- Cache File Content (after hit) ---")
        try:
            with open(VIX_CACHE_FILE, "r") as f:
                # Read and print first few lines to check header
                for i, line in enumerate(f):
                    print(line.strip())
                    if i >= 5:
                        break  # Print header + 5 data lines
        except Exception as e:
            print(f"Error reading cache file: {e}")

    # Test cache partial hit / update needed
    print("\n--- Attempt 3: Cache Update ---")
    extended_end = "2024-01-15"
    vix3 = get_vix_data(start, extended_end)
    if not vix3.empty:
        print(vix3.tail())
    else:
        print("Failed")
