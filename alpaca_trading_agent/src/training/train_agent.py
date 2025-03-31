# alpaca_trading_agent/src/training/train_agent.py

import pandas as pd
import numpy as np
import os
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure as sb3_configure_logger

# --- Configuration and Environment Loading ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_DIR = os.path.join(PROJECT_ROOT, 'config')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')

# Add relevant directories to sys.path
import sys
sys.path.insert(0, CONFIG_DIR)
sys.path.insert(0, os.path.join(SRC_DIR, 'environment')) # To import trading_env

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    import settings
    from trading_env import create_env # Import our environment creation function
except ImportError as e:
    logging.error(f"Error importing configuration or environment: {e}.")
    logging.error("Ensure config/settings.py and src/environment/trading_env.py exist.")
    sys.exit(1)

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, 'logs'), exist_ok=True) # For SB3 logs
os.makedirs(os.path.join(RESULTS_DIR, 'models'), exist_ok=True) # For trained models

# --- Training Parameters ---
# These can be moved to settings.py or a separate training config file later
MODEL_ALGO = "PPO" # Algorithm to use (e.g., PPO, A2C, DDPG)
TOTAL_TIMESTEPS = 20000 # Reduce for faster testing, increase for better training (e.g., 100k, 500k)
MODEL_FILENAME = f"{MODEL_ALGO}_{settings.TIME_INTERVAL}_{TOTAL_TIMESTEPS}"
LOG_FILENAME = f"{MODEL_ALGO}_{settings.TIME_INTERVAL}_{TOTAL_TIMESTEPS}_log"

# --- Load Processed Data ---
processed_train_filename = f"train_processed_{settings.TRAIN_START_DATE}_{settings.TRAIN_END_DATE}.csv"
processed_train_filepath = os.path.join(DATA_DIR, processed_train_filename)

try:
    train_df = pd.read_csv(processed_train_filepath)
    # Convert date column to datetime
    train_df['date'] = pd.to_datetime(train_df['date'])
    # Sort data
    train_df = train_df.sort_values(by=['date', 'tic'])
    # Reset index to default integer index, required by this env version
    train_df = train_df.reset_index(drop=True)
    logging.info(f"Loaded processed training data: {processed_train_filepath}")
except FileNotFoundError:
    logging.error(f"Processed training data file not found: {processed_train_filepath}")
    logging.error("Please run the preprocessing script first.")
    sys.exit(1)
except Exception as e:
    logging.error(f"Error loading processed training data: {e}")
    sys.exit(1)

# --- Create Training Environment ---
logging.info("Creating training environment...")
# Stable Baselines 3 requires a vectorized environment
# We use DummyVecEnv for a single environment instance
try:
    env_train = DummyVecEnv([lambda: create_env(train_df)])
    logging.info("Training environment created successfully.")
except Exception as e:
    logging.error(f"Failed to create training environment: {e}", exc_info=True)
    sys.exit(1)

# --- Configure Agent ---
logging.info(f"Configuring agent: {MODEL_ALGO}")

# Set up Stable Baselines 3 logger
log_path = os.path.join(RESULTS_DIR, 'logs', LOG_FILENAME)
sb3_logger = sb3_configure_logger(log_path, ["stdout", "csv", "tensorboard"])

# Define PPO model parameters (can be tuned)
# See SB3 documentation for details: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
model_params = {
    "policy": "MlpPolicy",
    "env": env_train,
    "n_steps": 2048, # Number of steps to run for each environment per update
    "batch_size": 64, # Minibatch size
    "n_epochs": 10, # Number of epoch when optimizing the surrogate loss
    "gamma": 0.99, # Discount factor
    "gae_lambda": 0.95, # Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    "clip_range": 0.2, # Clipping parameter, it can be a function
    "ent_coef": 0.0, # Entropy coefficient for the loss calculation
    "vf_coef": 0.5, # Value function coefficient for the loss calculation
    "max_grad_norm": 0.5, # The maximum value for the gradient clipping
    "verbose": 1, # Verbosity level: 0=no output, 1=info, 2=debug
    "seed": 42, # Seed for the pseudo random generators
    "device": "auto" # "auto", "cpu", "cuda"
    # Add other PPO parameters as needed
}

if MODEL_ALGO == "PPO":
    model = PPO(**model_params)
    model.set_logger(sb3_logger)
# Add elif blocks here for other algorithms like A2C, DDPG if needed
# elif MODEL_ALGO == "A2C":
#     model = A2C(...)
else:
    logging.error(f"Unsupported algorithm: {MODEL_ALGO}")
    sys.exit(1)

logging.info(f"Agent configured: {MODEL_ALGO} with policy {model_params['policy']}")

# --- Train Agent ---
logging.info(f"Starting training for {TOTAL_TIMESTEPS} timesteps...")
try:
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        log_interval=1, # Log training info every 'log_interval' updates
        reset_num_timesteps=True # Reset timesteps counter at the beginning of training
    )
    logging.info("Training finished.")
except Exception as e:
    logging.error(f"Error during model training: {e}", exc_info=True)
    sys.exit(1)

# --- Save Trained Model ---
model_save_path = os.path.join(RESULTS_DIR, 'models', f"{MODEL_FILENAME}.zip")
logging.info(f"Saving trained model to: {model_save_path}")
try:
    model.save(model_save_path)
    logging.info("Model saved successfully.")
except Exception as e:
    logging.error(f"Failed to save model: {e}", exc_info=True)

logging.info("--- Training Script Finished ---")
