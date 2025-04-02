# alpaca_trading_agent/src/training/tune_agent.py

import pandas as pd
import numpy as np
import os
import logging
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure as sb3_configure_logger
import time
import json # Added for JSON output

import random
# Assuming Stable Baselines3 uses PyTorch by default
import torch

# --- Seed for Reproducibility ---
SEED = 42 # You can change this value
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
# If using CUDA (GPU)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED) # if use multi-GPU
    # These settings can enforce determinism but might impact performance
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
# Set PYTHONHASHSEED environment variable (optional, affects hash randomization)
# Note: This needs to be set *before* Python starts for full effect,
# so setting it here might only have partial impact depending on execution context.
# Consider setting it in the shell environment if strict hash reproducibility is needed.
# import os
# os.environ['PYTHONHASHSEED'] = str(SEED)

logging.info(f"Global random seeds set to: {SEED}")


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

# Configure logging (consider setting matplotlib higher here too)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

try:
    import settings
    from trading_env import create_env # Import our environment creation function
except ImportError as e:
    logging.error(f"Error importing configuration or environment: {e}.")
    logging.error("Ensure config/settings.py and src/environment/trading_env.py exist.")
    sys.exit(1)

# --- Tuning Parameters ---
N_TRIALS = 50 # Number of Optuna trials to run (Increased from 20)
TUNING_TIMESTEPS = 10000 # Timesteps per trial (reduced for faster tuning)
VALIDATION_SPLIT_RATIO = 0.8 # Use 80% for training, 20% for validation within the trial

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, 'tuning'), exist_ok=True) # For tuning results/logs

# --- Load and Filter Data for Current Window ---
# Load the combined processed file
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

    # Filter data if override dates are provided (for Walk-Forward)
    if train_start_override and train_end_override:
        logging.info(f"Applying Walk-Forward date overrides: Train Start={train_start_override}, Train End={train_end_override}")
        train_start_dt = pd.to_datetime(train_start_override)
        train_end_dt = pd.to_datetime(train_end_override)
        # Filter the DataFrame for the specific training window
        current_window_df = full_df[(full_df['date'] >= train_start_dt) & (full_df['date'] <= train_end_dt)].reset_index(drop=True)
        logging.info(f"Filtered data for current window. Shape: {current_window_df.shape}, Date range: {current_window_df['date'].min()} to {current_window_df['date'].max()}")
    else:
        # Default behavior: Use TRAIN_START_DATE and TRAIN_END_DATE from settings
        logging.info("No Walk-Forward overrides found. Using default TRAIN dates from settings.")
        train_start_dt = pd.to_datetime(settings.TRAIN_START_DATE)
        train_end_dt = pd.to_datetime(settings.TRAIN_END_DATE)
        current_window_df = full_df[(full_df['date'] >= train_start_dt) & (full_df['date'] <= train_end_dt)].reset_index(drop=True)
        logging.info(f"Filtered data using settings dates. Shape: {current_window_df.shape}, Date range: {current_window_df['date'].min()} to {current_window_df['date'].max()}")


    # Split the *current window's data* into training and validation sets for tuning evaluation
    unique_dates = current_window_df['date'].unique()
    if len(unique_dates) < 2: # Need at least 2 unique dates to split
         logging.error("Not enough unique dates in the current data window to perform train/validation split for tuning.")
         sys.exit(1)

    split_index = int(len(unique_dates) * VALIDATION_SPLIT_RATIO)
    # Ensure split_index is at least 1 to have some validation data
    split_index = max(1, split_index)
    # Ensure split index is not beyond the last date
    split_index = min(split_index, len(unique_dates) - 1)

    split_date = unique_dates[split_index]

    train_split_df = current_window_df[current_window_df['date'] < split_date].reset_index(drop=True)
    validation_df = current_window_df[current_window_df['date'] >= split_date].reset_index(drop=True)

    if train_split_df.empty or validation_df.empty:
        logging.error("Train or validation split resulted in empty DataFrame. Check date range and split ratio.")
        sys.exit(1)

    logging.info(f"Data split for tuning within window: Train ({len(train_split_df['date'].unique())} days), Validation ({len(validation_df['date'].unique())} days)")

except FileNotFoundError:
    logging.error(f"Full processed data file not found: {full_processed_filepath}")
    sys.exit(1)
except Exception as e:
    logging.error(f"Error loading or splitting data: {e}")
    sys.exit(1)


# --- Objective Function for Optuna ---
def objective(trial: optuna.Trial) -> float:
    """
    Objective function for Optuna hyperparameter optimization.
    Trains a PPO model with suggested parameters and evaluates it on a validation set.
    """
    logging.info(f"\n--- Starting Optuna Trial {trial.number} ---")
    trial_start_time = time.time()

    # 1. Suggest Hyperparameters
    # Define search space (adjust ranges as needed)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    gamma = trial.suggest_float("gamma", 0.9, 0.9999)
    ent_coef = trial.suggest_float("ent_coef", 0.0, 0.1)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3])

    # --- Add net_arch tuning ---
    net_arch_str = trial.suggest_categorical("net_arch", ["small", "medium"]) # Keep it simple initially: [64, 64] vs [128, 128]
    if net_arch_str == "small":
        # Default SB3 PPO MlpPolicy architecture
        policy_kwargs = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))
    elif net_arch_str == "medium":
        policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
    # Add more options like "large" ([256, 256]) if desired
    # else: # large
    #     policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
    # --- End net_arch tuning ---

    # 2. Create Training Environment
    try:
        env_train_trial = DummyVecEnv([lambda: create_env(train_split_df)])
    except Exception as e:
        logging.exception(f"Trial {trial.number}: Failed during training environment creation.") # Use logging.exception
        # Return a very low value or raise optuna.TrialPruned() if env creation fails
        return -np.inf # Indicate failure

    # 3. Configure Agent
    model_params = {
        "policy": "MlpPolicy",
        "env": env_train_trial,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": 10, # Keep fixed or tune as well?
        "gamma": gamma,
        "gae_lambda": 0.95, # Keep fixed or tune?
        "clip_range": clip_range,
        "ent_coef": ent_coef,
        "vf_coef": 0.5, # Keep fixed or tune?
        "max_grad_norm": 0.5, # Keep fixed or tune?
        "learning_rate": learning_rate,
        "verbose": 0, # Reduce verbosity during tuning
        "seed": SEED + trial.number, # Use different seed per trial (Use defined SEED)
        "device": "auto",
        "policy_kwargs": policy_kwargs # Pass the suggested network architecture
    }

    try:
        model = PPO(**model_params)
        # Disable SB3 logging during tuning to avoid clutter/conflicts
        # model.set_logger(sb3_configure_logger(f"./tuning_logs/trial_{trial.number}", ["stdout", "csv"]))
    except Exception as e:
        logging.exception(f"Trial {trial.number}: Failed during PPO model configuration.") # Use logging.exception
        return -np.inf # Indicate failure

    # 4. Train Agent
    try:
        model.learn(total_timesteps=TUNING_TIMESTEPS, log_interval=None) # No intermediate logging needed
        logging.info(f"Trial {trial.number}: Training finished.")
    except Exception as e:
        logging.exception(f"Trial {trial.number}: Error during model training.") # Use logging.exception
        return -np.inf # Indicate failure

    # 5. Evaluate Agent on Validation Set
    try:
        env_validation = create_env(validation_df) # Create single instance for eval
        obs, info = env_validation.reset()
        terminated = False
        truncated = False
        total_reward_validation = 0.0
        account_value_validation = env_validation.initial_amount

        while not (terminated or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env_validation.step(action)
            total_reward_validation += reward
            if terminated or truncated:
                # Get final account value from the environment's memory
                if hasattr(env_validation, 'asset_memory') and env_validation.asset_memory:
                    account_value_validation = env_validation.asset_memory[-1]
                else: # Fallback if asset_memory isn't available/reliable
                    account_value_validation = info.get('total_asset', env_validation.initial_amount) # Use info if available
                break

        logging.info(f"Trial {trial.number}: Validation finished. Final Account Value: {account_value_validation:.2f}")

    except Exception as e:
        logging.exception(f"Trial {trial.number}: Error during validation.") # Use logging.exception
        return -np.inf # Indicate failure
    finally:
        # Ensure environments are closed
        if 'env_train_trial' in locals():
            env_train_trial.close()
        if 'env_validation' in locals() and hasattr(env_validation, 'close'):
             env_validation.close()


    trial_duration = time.time() - trial_start_time
    logging.info(f"--- Optuna Trial {trial.number} Finished (Duration: {trial_duration:.2f}s) ---")

    # 6. Return Metric (Final portfolio value on validation set)
    # Handle cases where validation might result in NaN or Inf
    if np.isnan(account_value_validation) or np.isinf(account_value_validation):
        return -np.inf

    return account_value_validation


# --- Run Optuna Study ---
if __name__ == "__main__":
    # Check for Walk-Forward model suffix
    model_suffix = os.environ.get('WF_MODEL_SUFFIX', '')
    if model_suffix:
        logging.info(f"Using Walk-Forward model suffix: {model_suffix}")

    study_name = f"ppo-tuning-{settings.TIME_INTERVAL}-{int(time.time())}{model_suffix}"
    db_filename = f"{study_name}.db"
    storage_path = f"sqlite:///{os.path.join(RESULTS_DIR, 'tuning', db_filename)}" # Store results in a DB

    logging.info(f"Starting Optuna study: {study_name}")
    logging.info(f"Number of trials: {N_TRIALS}")
    logging.info(f"Timesteps per trial: {TUNING_TIMESTEPS}")
    logging.info(f"Results will be saved to: {storage_path}")

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_path, # Use DB storage
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(), # Add a pruner
        load_if_exists=True # Resume study if DB exists
    )

    try:
        study.optimize(objective, n_trials=N_TRIALS, timeout=None) # Set timeout in seconds if needed
    except KeyboardInterrupt:
        logging.warning("Tuning stopped manually via KeyboardInterrupt.")
    except Exception as e:
        logging.error(f"An error occurred during the Optuna study: {e}", exc_info=True)

    logging.info("\n--- Optuna Study Finished ---")
    logging.info(f"Number of finished trials: {len(study.trials)}")

    try:
        best_trial = study.best_trial
        logging.info(f"Best trial number: {best_trial.number}")
        logging.info(f"Best value (Validation Account Value): {best_trial.value:.2f}")
        logging.info("Best hyperparameters:")
        for key, value in best_trial.params.items():
            logging.info(f"  {key}: {value}")

        # Save best params to JSON file in config directory, adding suffix if present
        best_params_filename = f"best_ppo_params{model_suffix}.json"
        best_params_path = os.path.join(CONFIG_DIR, best_params_filename)
        best_params_dict = best_trial.params
        # Add study info if desired
        best_params_dict['_study_name'] = study_name
        best_params_dict['_window_train_start'] = os.environ.get('WF_OVERRIDE_TRAIN_START') # Log WF dates if available
        best_params_dict['_window_train_end'] = os.environ.get('WF_OVERRIDE_TRAIN_END')
        best_params_dict['_best_trial_number'] = best_trial.number
        # Handle potential -Infinity before saving to JSON
        best_value = best_trial.value
        if best_value == -np.inf:
            logging.warning("Best trial resulted in -Infinity, saving value as null in JSON.")
            best_params_dict['_best_value'] = None # Replace -Infinity with None for JSON compatibility
        else:
            best_params_dict['_best_value'] = best_value

        try:
            with open(best_params_path, 'w') as f:
                json.dump(best_params_dict, f, indent=4) # Use indent for readability
            logging.info(f"Best parameters saved to: {best_params_path}")
        except IOError as e:
            logging.error(f"Failed to save best parameters to {best_params_path}: {e}", exc_info=True)
        except Exception as e:
             logging.error(f"An unexpected error occurred while saving best parameters: {e}", exc_info=True)


    except optuna.exceptions.OptunaError as e:
         logging.warning(f"Could not retrieve best trial, study might be empty or failed: {e}")
    except Exception as e:
         logging.error(f"Error retrieving or saving best trial info: {e}", exc_info=True)

    logging.info("--- Tuning Script Finished ---")
