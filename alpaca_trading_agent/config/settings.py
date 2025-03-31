# Configuration settings for Alpaca trading agent

import os
from dotenv import load_dotenv

load_dotenv()

# Alpaca API Configuration
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY', '')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY', '')
ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')  # Use paper trading by default
DATA_SOURCE = 'alpaca'

# Trading Parameters
TECH_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'INTC', 'CSCO', 'ADBE']
TICKERS = TECH_TICKERS  # Use all tickers by default

# Technical Indicators
INDICATORS = [
    'close', 'high', 'low', 'trade_count', 'open', 'volume', 'vwap',
    'macd', 'rsi_14', 'cci_14', 'boll_ub', 'boll_lb'
]

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
NUM_STOCK_FEATURES = len(INDICATORS)
STATE_SPACE = 1 + STOCK_DIM + STOCK_DIM * NUM_STOCK_FEATURES
ACTION_SPACE = STOCK_DIM

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

# Trading cost and position limits
MAX_STOCK_POSITION = 100
TRADE_COST_PCT = 0.001  # 0.1% trading cost
TURBULENCE_THRESHOLD = 30
