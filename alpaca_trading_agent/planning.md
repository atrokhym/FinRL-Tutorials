# Project Improvement Plan

This file tracks the planned improvements for the `alpaca_trading_agent` project.

**Phase 1: Hyperparameter Tuning (Optuna) - Completed**

1.  Add Dependency: `optuna` added to `requirements.txt`. (Done)
2.  Identify Target Script: `src/training/train_agent.py`. (Done)
3.  Define Search Space: Key PPO hyperparameters identified. (Done in `tune_agent.py`)
4.  Prepare Validation Data: Split implemented in `tune_agent.py`. (Done)
5.  Create Objective Function: Implemented in `tune_agent.py`. (Done)
6.  Set Up Optuna Study: Implemented in `tune_agent.py`. (Done)
7.  Integrate into Workflow:
    *   `tune` mode added to `main.py`. (Done)
    *   `train_agent.py` updated to load best hyperparameters from `config/best_ppo_params.json`. (Done)

**Phase 2: Environment Improvements (Transaction Costs & Slippage) - Completed**

1.  Add Configuration: `TRANSACTION_COST_PERCENT` and `SLIPPAGE_PERCENT` added to `config/settings.py`. (Done)
2.  Modify Environment (`src/environment/trading_env.py`):
    *   `step` method updated to calculate and apply costs/slippage. (Done)
    *   Reward/portfolio value reflects costs/slippage. (Done)
3.  Re-run Tuning: `python main.py tune` executed for the cost-aware environment. (Done)
4.  Re-run Training: `python main.py train` executed using new parameters. (Done)
5.  Re-run Backtesting: `python main.py backtest` executed to evaluate performance with costs/slippage. (Done)

**Phase 3: Dynamic Adaptation - Walk-Forward & Retraining - In Progress**

*   **Part A: Walk-Forward Backtesting:** Implement a walk-forward validation framework to provide a more realistic performance assessment and test model robustness over time.
    1.  **Define Walk-Forward Parameters:** Determine window size (training length), step size (how much the window slides forward), and overall period in `config/settings.py`.
    2.  **Modify Orchestrator (`main.py`):** Add a new `walkforward` mode.
    3.  **Create Walk-Forward Script (`src/backtesting/walkforward_backtest.py`):**
        *   Loop through time periods defined by the parameters.
        *   In each loop iteration:
            *   Define training and testing date ranges.
            *   Slice the full dataset for the current training period.
            *   (Optional but Recommended) Run hyperparameter tuning (`tune_agent.py`) on the current training slice.
            *   Run training (`train_agent.py`) on the current training slice using tuned or fixed params.
            *   Slice the full dataset for the current testing period.
            *   Run backtesting (`backtest_agent.py` logic) on the current test slice using the model trained for that period.
            *   Store/aggregate performance metrics for each test period.
        *   Calculate overall performance across all test periods.
        *   Save aggregated results and potentially plots.
*   **Part B: Periodic Retraining (for Live/Paper Trading):** Implement a mechanism for periodically retraining the model using the latest data.
    1.  **Enhance Data Fetching/Preprocessing:** Ensure `fetch_data.py` and `preprocess_data.py` can handle fetching and processing only the *latest* incremental data efficiently.
    2.  **Modify Training Script (`train_agent.py`):** Add an option to load the *previous* model and continue training (fine-tuning) instead of always starting from scratch.
    3.  **Create Retraining Orchestration:** This could be a modification to `main.py` or a separate script/scheduler (e.g., using `cron` or a workflow tool) that:
        *   Runs data fetching/preprocessing periodically (e.g., daily, weekly).
        *   Runs `train_agent.py` in fine-tuning mode on the updated dataset.
        *   Updates the model used by the `papertrade` script.

**Phase 4: Further Enhancements (Potential)**

*   Training Monitoring (TensorBoard/MLflow Integration)
*   Enhanced Risk Management (Stop-loss, Position Sizing in Environment/Agent)
*   Order Execution Logic (Limit Orders, Market Impact Models)
*   Ensemble Methods (Combining multiple models/strategies)
*   Further Environment Improvements (e.g., more sophisticated state, reward shaping, market regime features)
