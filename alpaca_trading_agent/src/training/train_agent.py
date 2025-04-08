# alpaca_trading_agent/src/training/train_agent.py

import pandas as pd
import numpy as np
import os
import logging
import json # Added for loading params
import argparse # Added for command-line arguments
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure as sb3_configure_logger
from stable_baselines3.common.callbacks import BaseCallback
# REMOVED: from stable_baselines3.common.callbacks import TensorboardCallback as SB3TensorboardCallback

# --- Configuration and Environment Loading ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_DIR = os.path.join(PROJECT_ROOT, 'config')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')

# Add relevant directories to sys.path
import sys
sys.path.insert(0, CONFIG_DIR)
sys.path.insert(0, SRC_DIR) # Add src directory itself for utils and environment
# Logging setup will be done explicitly using the shared utility
from utils.logging_setup import configure_file_logging
# Keep import of custom callback file, but don't use the callback itself
from utils.custom_callbacks import TensorboardCallback

try:
    import settings
    from environment.trading_env import create_env # Import from environment subdirectory
except ImportError as e:
    logging.error(f"Error importing configuration or environment: {e}.")
    logging.error("Ensure config/settings.py and src/environment/trading_env.py exist.")
    sys.exit(1)

# --- Command Line Arguments ---
def parse_args():
    parser = argparse.ArgumentParser(description="Train a PPO agent for stock trading.")
    parser.add_argument('--timesteps', type=int, default=settings.TOTAL_TIMESTEPS, help='Total number of training timesteps.')
    parser.add_argument('--model-suffix', type=str, default="", help='Suffix to append to model and log filenames (e.g., for walk-forward).')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Logging level.')
    parser.add_argument('--hyperparams', type=str, default="", help='Path to JSON file with hyperparameters to override defaults.')
    return parser.parse_args()

# --- Main Training Function ---
def train_model(total_timesteps: int, model_suffix: str = "", log_level: str = "INFO", hyperparams_path: str = ""):
    """
    Trains the PPO agent.

    Args:
        total_timesteps (int): The total number of samples (env steps) to train on.
        model_suffix (str): Suffix for model/log filenames. Defaults to "".
        log_level (str): Logging level. Defaults to "INFO".
        hyperparams_path (str): Path to hyperparameter JSON file. Defaults to "".
    """
    # --- Configure Logging for this script ---
    configure_file_logging(log_level)

    logging.info("--- Starting Training Script ---")
    logging.info(f"Total Timesteps: {total_timesteps}")
    if model_suffix:
        logging.info(f"Using Model Suffix: {model_suffix}")
    logging.info(f"Log Level: {log_level}")
    if hyperparams_path:
        logging.info(f"Using Hyperparameters File: {hyperparams_path}")

    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, 'logs'), exist_ok=True) # For SB3 logs
    os.makedirs(os.path.join(RESULTS_DIR, 'models'), exist_ok=True) # For trained models

    # --- Model and Log Filenames ---
    MODEL_ALGO = "PPO" # Keep algorithm fixed for now
    MODEL_FILENAME = f"{MODEL_ALGO}_{settings.TIME_INTERVAL}_{total_timesteps}{model_suffix}"
    LOG_FILENAME = f"{MODEL_ALGO}_{settings.TIME_INTERVAL}_{total_timesteps}{model_suffix}_log"

    # --- Load Processed Data ---
    processed_filename = "processed_data.csv" # Use the standardized name
    processed_filepath = os.path.join(DATA_DIR, processed_filename)

    try:
        train_df = pd.read_csv(processed_filepath)
        train_df['date'] = pd.to_datetime(train_df['date'])
        train_df = train_df.sort_values(by=['date', 'tic']).reset_index(drop=True)
        logging.info(f"Loaded processed data from: {processed_filepath}")

        # --- Filter Data for Combined Training + Validation Period ---
        logging.info(f"Filtering data for final training: {settings.TRAIN_START_DATE} to {settings.VALIDATION_END_DATE}")
        train_df = train_df[
            (train_df['date'] >= settings.TRAIN_START_DATE) &
            (train_df['date'] <= settings.VALIDATION_END_DATE)
        ].reset_index(drop=True)
        logging.info(f"Final training data period: {train_df['date'].min()} to {train_df['date'].max()}. Shape: {train_df.shape}")
        # --- End Filtering ---

        if train_df.empty:
            logging.error("Filtered training data is empty. Check date ranges and preprocessing step.")
            sys.exit(1)

    except FileNotFoundError:
        logging.error(f"Processed data file not found: {processed_filepath}")
        logging.error("Please run the preprocessing script first.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading processed training data: {e}")
        sys.exit(1)

    # --- Create Training Environment ---
    logging.info("Creating training environment...")
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
    # Configure logger for stdout, csv, AND tensorboard
    sb3_logger = sb3_configure_logger(log_path, ["stdout", "csv", "tensorboard"])
    logging.info(f"SB3 log path set to: {log_path}")

    # --- Load Hyperparameters ---
    loaded_params = {}
    params_source_msg = "Using default parameters."
    default_tuned_filename = f"best_ppo_params{model_suffix}.json"
    default_tuned_path = os.path.join(CONFIG_DIR, default_tuned_filename)

    paths_to_try = []
    if hyperparams_path:
        paths_to_try.append(hyperparams_path)
    paths_to_try.append(default_tuned_path)

    for path in paths_to_try:
        try:
            with open(path, 'r') as f:
                loaded_params = json.load(f)
            logging.info(f"Loaded hyperparameters from: {path}")
            params_source_msg = f"Using parameters from: {path}"
            loaded_params.pop('_study_name', None)
            loaded_params.pop('_best_trial_number', None)
            loaded_params.pop('_best_value', None)
            loaded_params.pop('_window_train_start', None)
            loaded_params.pop('_window_train_end', None)
            break
        except FileNotFoundError:
            logging.debug(f"Hyperparameter file not found at {path}. Trying next.")
            continue
        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON from {path}. Check file format.")
            continue
        except Exception as e:
            logging.error(f"Error loading parameters from {path}: {e}.")
            continue

    # --- Define PPO model parameters ---
    policy_kwargs = None
    net_arch_str = loaded_params.get("net_arch")
    if net_arch_str == "small":
        policy_kwargs = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))
        logging.info("Using 'small' network architecture: [64, 64]")
    elif net_arch_str == "medium":
        policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
        logging.info("Using 'medium' network architecture: [128, 128]")
    else:
        logging.info("Using default network architecture (likely [64, 64]).")

    model_params = {
        "policy": "MlpPolicy",
        "env": env_train,
        "n_epochs": 10,
        "gae_lambda": 0.95,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "verbose": 1,
        "seed": 42,
        "device": "auto",
        "tensorboard_log": log_path, # Use the SB3 log path for TensorBoard
        "policy_kwargs": policy_kwargs
    }

    default_tuned = {
        "n_steps": 2048,
        "batch_size": 64,
        "gamma": 0.99,
        "clip_range": 0.2,
        "ent_coef": 0.0,
        "learning_rate": 3e-4
    }

    loaded_params.pop("net_arch", None) # Remove net_arch key if it existed

    for key, default_value in default_tuned.items():
        model_params[key] = loaded_params.get(key, default_value)

    logging.info(params_source_msg)
    logging.info("Using the following PPO parameters:")
    for key, value in model_params.items():
        if key not in ["env", "policy_kwargs"]:
            logging.info(f"  {key}: {value}")
        elif key == "policy_kwargs" and value is not None:
             logging.info(f"  policy_kwargs: Custom network architecture applied.")

    if MODEL_ALGO == "PPO":
        model = PPO(**model_params)
        model.set_logger(sb3_logger) # Set the logger configured for all formats
    else:
        logging.error(f"Unsupported algorithm: {MODEL_ALGO}")
        sys.exit(1)

    logging.info(f"Agent configured: {MODEL_ALGO} with policy {model_params['policy']}")

    # --- Train Agent ---
    logging.info(f"Starting training for {total_timesteps} timesteps...")
    try:
        log_interval_freq = 1 # Log frequently to TensorBoard
        logging.info(f"Setting model.learn log_interval to: {log_interval_freq}")
        
        # Create a custom callback that directly logs the metrics we want
        class DirectLoggingCallback(BaseCallback):
            def __init__(self, verbose=0):
                super().__init__(verbose)
            
            def _on_step(self) -> bool:
                return True
            
            def _on_rollout_end(self) -> bool:
                # Get the losses directly from the model
                if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'name_to_value'):
                    values = self.model.logger.name_to_value
                    
                    # Log policy loss
                    if 'train/policy_gradient_loss' in values:
                        self.logger.record("train/policy_loss", values['train/policy_gradient_loss'])
                    
                    # Log value loss
                    if 'train/value_loss' in values:
                        self.logger.record("train/value_loss", values['train/value_loss'])
                    
                    # Log entropy loss
                    if 'train/entropy_loss' in values:
                        self.logger.record("train/entropy_loss", values['train/entropy_loss'])
                
                return True
        
        # Use the direct logging callback
        direct_callback = DirectLoggingCallback(verbose=1)
        
        model.learn(
            total_timesteps=total_timesteps,
            log_interval=log_interval_freq,
            reset_num_timesteps=True,
            callback=direct_callback
        )
        logging.info("Training finished.")
    except Exception as e:
        logging.error(f"Error during model training: {e}", exc_info=True)
    finally:
        if 'env_train' in locals() and hasattr(env_train, 'close'):
             env_train.close()
             logging.info("Training environment closed.")

    # --- Save Trained Model ---
    model_save_filename = f"{MODEL_FILENAME}.zip"
    model_save_path = os.path.join(RESULTS_DIR, 'models', model_save_filename)
    logging.info(f"Saving trained model to: {model_save_path}")
    try:
        model.save(model_save_path)
        logging.info("Model saved successfully.")
    except NameError:
         logging.error("Model variable not defined (likely due to training error). Cannot save model.")
    except Exception as e:
        logging.error(f"Failed to save model: {e}", exc_info=True)

    logging.info("--- Training Script Finished ---")

if __name__ == "__main__":
    args = parse_args()
    train_model(
        total_timesteps=args.timesteps,
        model_suffix=args.model_suffix,
        log_level=args.log_level,
        hyperparams_path=args.hyperparams
    )
