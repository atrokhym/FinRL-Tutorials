# alpaca_trading_agent/src/training/train_agent.py

import pandas as pd
import numpy as np
import os
import logging
import json # Added for loading params
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
# Check for Walk-Forward model suffix
model_suffix = os.environ.get('WF_MODEL_SUFFIX', '')
if model_suffix:
    logging.info(f"Using Walk-Forward model suffix: {model_suffix}")

MODEL_FILENAME = f"{MODEL_ALGO}_{settings.TIME_INTERVAL}_{TOTAL_TIMESTEPS}{model_suffix}"
LOG_FILENAME = f"{MODEL_ALGO}_{settings.TIME_INTERVAL}_{TOTAL_TIMESTEPS}{model_suffix}_log"

# --- Load Processed Data (and filter for Walk-Forward) ---
full_processed_filename = "full_processed_combined.csv"
full_processed_filepath = os.path.join(DATA_DIR, full_processed_filename)

# Check for Walk-Forward override environment variables
train_start_override = os.environ.get('WF_OVERRIDE_TRAIN_START')
train_end_override = os.environ.get('WF_OVERRIDE_TRAIN_END')

try:
    full_df = pd.read_csv(full_processed_filepath)
    full_df['date'] = pd.to_datetime(full_df['date'])
    full_df = full_df.sort_values(by=['date', 'tic']).reset_index(drop=True)
    logging.info(f"Loaded full processed data: {full_processed_filepath}")

    # Filter data for the current training window
    if train_start_override and train_end_override:
        logging.info(f"Applying Walk-Forward date overrides: Train Start={train_start_override}, Train End={train_end_override}")
        train_start_dt = pd.to_datetime(train_start_override)
        train_end_dt = pd.to_datetime(train_end_override)
        train_df = full_df[(full_df['date'] >= train_start_dt) & (full_df['date'] <= train_end_dt)].reset_index(drop=True)
    else:
        # Default behavior: Use TRAIN_START_DATE and TRAIN_END_DATE from settings
        logging.info("No Walk-Forward overrides found. Using default TRAIN dates from settings.")
        train_start_dt = pd.to_datetime(settings.TRAIN_START_DATE)
        train_end_dt = pd.to_datetime(settings.TRAIN_END_DATE)
        train_df = full_df[(full_df['date'] >= train_start_dt) & (full_df['date'] <= train_end_dt)].reset_index(drop=True)

    if train_df.empty:
         logging.error(f"Training data is empty after filtering for dates {train_start_dt} to {train_end_dt}. Check date ranges.")
         sys.exit(1)

    logging.info(f"Using training data for period: {train_df['date'].min()} to {train_df['date'].max()}. Shape: {train_df.shape}")

except FileNotFoundError:
    logging.error(f"Full processed data file not found: {full_processed_filepath}")
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

# Set up Stable Baselines 3 logger (using potentially suffixed name)
log_path = os.path.join(RESULTS_DIR, 'logs', LOG_FILENAME)
sb3_logger = sb3_configure_logger(log_path, ["stdout", "csv", "tensorboard"])
logging.info(f"SB3 log path set to: {log_path}")

# --- Load Tuned Hyperparameters (using potentially suffixed name) ---
best_params_filename = f"best_ppo_params{model_suffix}.json"
best_params_path = os.path.join(CONFIG_DIR, best_params_filename)
tuned_params = {}
try:
    with open(best_params_path, 'r') as f:
        tuned_params = json.load(f)
    logging.info(f"Loaded best hyperparameters from: {best_params_path}")
    # Remove metadata if it exists
    tuned_params.pop('_study_name', None)
    tuned_params.pop('_best_trial_number', None)
    tuned_params.pop('_best_value', None)
    tuned_params.pop('_window_train_start', None) # Remove WF info if present
    tuned_params.pop('_window_train_end', None)
except FileNotFoundError:
    logging.warning(f"Best parameters file not found at {best_params_path}. Using default parameters.")
except json.JSONDecodeError:
    logging.error(f"Error decoding JSON from {best_params_path}. Using default parameters.")
except Exception as e:
    logging.error(f"Error loading best parameters from {best_params_path}: {e}. Using default parameters.")

# --- Define PPO model parameters ---
# Start with default/fixed parameters
model_params = {
    "policy": "MlpPolicy",
    "env": env_train,
    "n_epochs": 10,
    "gae_lambda": 0.95,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "verbose": 1,
    "seed": 42,
    "device": "auto"
}

# Update with tuned parameters if loaded, otherwise use defaults from original script
default_tuned = {
    "n_steps": 2048,
    "batch_size": 64,
    "gamma": 0.99,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "learning_rate": 3e-4 # Default PPO learning rate in SB3
}

for key, default_value in default_tuned.items():
    model_params[key] = tuned_params.get(key, default_value) # Use tuned value or default

# Log the final parameters being used
logging.info("Using the following PPO parameters:")
for key, value in model_params.items():
    # Don't log the env object
    if key != "env":
        logging.info(f"  {key}: {value}")


# Original hardcoded params for reference (now replaced by loaded/default logic above)
# model_params_original = {
#     "policy": "MlpPolicy",
#     "env": env_train,
#     "n_steps": 2048, # Number of steps to run for each environment per update
#     "batch_size": 64, # Minibatch size
#     "n_epochs": 10, # Number of epoch when optimizing the surrogate loss
#     "gamma": 0.99, # Discount factor
#     "gae_lambda": 0.95, # Factor for trade-off of bias vs variance for Generalized Advantage Estimator
#     "clip_range": 0.2, # Clipping parameter, it can be a function
#     "ent_coef": 0.0, # Entropy coefficient for the loss calculation
#     "vf_coef": 0.5, # Value function coefficient for the loss calculation
#     "max_grad_norm": 0.5, # The maximum value for the gradient clipping
#     "verbose": 1, # Verbosity level: 0=no output, 1=info, 2=debug
#     "seed": 42, # Seed for the pseudo random generators
#     "device": "auto" # "auto", "cpu", "cuda"
#     # Add other PPO parameters as needed
# }

if MODEL_ALGO == "PPO":
    # Create model using the combined parameters
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

# --- Save Trained Model (using potentially suffixed name) ---
model_save_filename = f"{MODEL_FILENAME}.zip"
model_save_path = os.path.join(RESULTS_DIR, 'models', model_save_filename)
logging.info(f"Saving trained model to: {model_save_path}")
try:
    model.save(model_save_path)
    logging.info("Model saved successfully.")
except Exception as e:
    logging.error(f"Failed to save model: {e}", exc_info=True)

logging.info("--- Training Script Finished ---")
