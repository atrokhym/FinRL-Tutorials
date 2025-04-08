# Configuration settings for Alpaca trading agent

import os
from dotenv import load_dotenv

load_dotenv()

# Alpaca API Configuration
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY', '')
ALPACA_API_SECRET = os.getenv('ALPACA_API_SECRET', '')
ALPACA_API_BASE_URL = os.getenv('ALPACA_API_BASE_URL', 'https://paper-api.alpaca.markets')  # Use paper trading by default
DATA_SOURCE = 'alpaca'

# Trading Parameters
TECH_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'INTC', 'CSCO', 'ADBE']
TICKERS = TECH_TICKERS  # Use all tickers by default

# Technical Indicators (compatible with stockstats library)
INDICATORS = [
    'close', 'high', 'low', 'trade_count', 'open', 'volume', 'vwap', # Basic Price/Volume
    'macd', 'rsi_14', 'cci_14', 'boll_ub', 'boll_lb'  # Standard Technical Indicators
    # 'turbulence' indicator is calculated separately in preprocess_data.py
]
INDICATORS_WITH_TURBULENCE = INDICATORS + ['turbulence'] # Define a list including turbulence

# Training/Testing Periods
TRAIN_START_DATE = "2015-01-01"
TRAIN_END_DATE = "2022-12-31"
TEST_START_DATE = "2023-01-01"
TEST_END_DATE = "2024-12-31"
TIME_INTERVAL = '1D'  # Maps to Alpaca timeframe

# Paper Trading Period
PAPER_TRADE_START_DATE = "2025-01-01"

# Model paths
TRAINED_MODEL_DIR = './models/papertrading_alpaca_erl'
RETRAINED_MODEL_DIR = './models/papertrading_alpaca_erl_retrain'

# Environment settings
INITIAL_ACCOUNT_BALANCE = 100000
STOCK_DIM = len(TICKERS)
# Use the list *including* turbulence for state space calculation
NUM_STOCK_FEATURES = len(INDICATORS_WITH_TURBULENCE)
STATE_SPACE = 1 + STOCK_DIM + STOCK_DIM * NUM_STOCK_FEATURES
ACTION_SPACE = STOCK_DIM
TOTAL_TIMESTEPS = 500000
#20000
#100,000, 500,000# Default total timesteps for training

# Agent parameters (ElegantRL) - NOTE: Project currently uses Stable Baselines 3 (PPO)
# ERL_PARAMS = {
#     "learning_rate": 3e-5,
#     "batch_size": 2048,
#     "gamma": 0.985,
#     "seed": 312,
#     "net_dimension": [128, 64],
#     "target_step": 5000,
#     "eval_gap": 30,
#     "eval_times": 1
# }

# Trading cost, slippage, and position limits
TRANSACTION_COST_PERCENT = 0.001  # 0.1% per trade
SLIPPAGE_PERCENT = 0.0005      # 0.05% per trade (applied to execution price)
MAX_STOCK_POSITION = 100        # Example: Maximum shares per stock
TURBULENCE_THRESHOLD = 30       # Example: Threshold for market turbulence index

# Walk-Forward Backtesting Parameters
WF_START_DATE = "2015-01-01"    # Start date for the entire walk-forward analysis
WF_END_DATE = "2024-12-31"      # End date for the entire walk-forward analysis
WF_TRAIN_WINDOW_YEARS = 5       # Initial training window length in years
WF_STEP_YEARS = 1               # Step forward size (also the test period length) in years
