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
N_TRIALS = 20 # Number of Optuna trials to run
TUNING_TIMESTEPS = 10000 # Timesteps per trial (reduced for faster tuning)
VALIDATION_SPLIT_RATIO = 0.8 # Use 80% for training, 20% for validation within the trial

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, 'tuning'), exist_ok=True) # For tuning results/logs

# --- Load and Split Data ---
processed_train_filename = f"train_processed_{settings.TRAIN_START_DATE}_{settings.TRAIN_END_DATE}.csv"
processed_train_filepath = os.path.join(DATA_DIR, processed_train_filename)

try:
    full_train_df = pd.read_csv(processed_train_filepath)
    full_train_df['date'] = pd.to_datetime(full_train_df['date'])
    full_train_df = full_train_df.sort_values(by=['date', 'tic']).reset_index(drop=True)
    logging.info(f"Loaded full training data: {processed_train_filepath}")

    # Split data into training and validation sets for tuning evaluation
    unique_dates = full_train_df['date'].unique()
    split_index = int(len(unique_dates) * VALIDATION_SPLIT_RATIO)
    split_date = unique_dates[split_index]

    train_split_df = full_train_df[full_train_df['date'] < split_date].reset_index(drop=True)
    validation_df = full_train_df[full_train_df['date'] >= split_date].reset_index(drop=True)

    logging.info(f"Data split for tuning: Train ({len(train_split_df['date'].unique())} days), Validation ({len(validation_df['date'].unique())} days)")

except FileNotFoundError:
    logging.error(f"Processed training data file not found: {processed_train_filepath}")
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
    # net_arch = trial.suggest_categorical("net_arch", ["small", "medium"]) # Example: Needs mapping
    # policy_kwargs = dict(net_arch=[dict(pi=[64, 64], vf=[64, 64])]) # Example

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
        "seed": 42 + trial.number, # Use different seed per trial
        "device": "auto"
        # "policy_kwargs": policy_kwargs # If tuning net_arch
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
    study_name = f"ppo-tuning-{settings.TIME_INTERVAL}-{int(time.time())}"
    storage_path = f"sqlite:///{RESULTS_DIR}/tuning/{study_name}.db" # Store results in a DB

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

        # Save best params to JSON file in config directory
        best_params_path = os.path.join(CONFIG_DIR, "best_ppo_params.json") # Save in config dir
        best_params_dict = best_trial.params
        # Add study info if desired
        best_params_dict['_study_name'] = study_name
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
