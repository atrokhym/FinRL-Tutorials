#!/bin/bash

# Script to automate the daily retraining and optional trading pipeline
# Ensure this script is run from the project root (FinRL-Tutorials)
# or adjust paths accordingly.

set -x
set -e


# --- Configuration ---
LOG_LEVEL="INFO" # Set desired log level for the pipeline steps
PYTHON_EXE="python" # Use 'python' or specify the full path to your virtual env python if needed
# Determine the directory where this script resides
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
MAIN_SCRIPT="$SCRIPT_DIR/main.py" # Path to main.py relative to this script

# --- Calculate Yesterday's Date ---
YESTERDAY=$(date -d "yesterday" '+%Y-%m-%d')
echo "--- Running pipeline for data up to: $YESTERDAY ---"

# --- Pipeline Steps ---

# 1. Fetch Data up to Yesterday
echo "[Step 1/5] Fetching data up to $YESTERDAY..."
$PYTHON_EXE $MAIN_SCRIPT fetch --log-level $LOG_LEVEL --train-end-date "$YESTERDAY"
if [ $? -ne 0 ]; then
  echo "ERROR: Data fetching failed. Aborting pipeline."
  exit 1
fi
echo "Data fetching finished."

# 2. Preprocess Data up to Yesterday
echo "[Step 2/5] Preprocessing data up to $YESTERDAY..."
$PYTHON_EXE $MAIN_SCRIPT preprocess --log-level $LOG_LEVEL --train-end-date "$YESTERDAY"
if [ $? -ne 0 ]; then
  echo "ERROR: Data preprocessing failed. Aborting pipeline."
  exit 1
fi
echo "Data preprocessing finished."

# 3. Tune Hyperparameters (Optional daily, but recommended periodically)
#    Uses the 'processed_data.csv' created in the previous step.
# echo "[Step 3/5] Tuning hyperparameters..."
# $PYTHON_EXE $MAIN_SCRIPT tune --log-level $LOG_LEVEL
# if [ $? -ne 0 ]; then
#   echo "ERROR: Hyperparameter tuning failed. Aborting pipeline."
#   exit 1
# fi
# echo "Hyperparameter tuning finished."

# 4. Train Agent
#    Uses 'processed_data.csv' and 'best_ppo_params.json' from previous steps.
echo "[Step 4/5] Training agent..."
$PYTHON_EXE $MAIN_SCRIPT train --log-level $LOG_LEVEL
if [ $? -ne 0 ]; then
  echo "ERROR: Agent training failed. Aborting pipeline."
  exit 1
fi
echo "Agent training finished."

# 5. Start Paper Trading (Optional - uncomment to run)
# echo "[Step 5/5] Starting paper trading..."
# $PYTHON_EXE $MAIN_SCRIPT papertrade --log-level $LOG_LEVEL
# if [ $? -ne 0 ]; then
#   echo "ERROR: Failed to start paper trading agent."
#   # Don't necessarily exit 1 here, as prior steps were successful
# fi
# echo "Paper trading agent initiated (if uncommented)."

echo "--- Daily pipeline finished successfully! ---"
exit 0
