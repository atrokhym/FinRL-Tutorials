# Project Improvement Plan

This file tracks the planned improvements for the `alpaca_trading_agent` project.

**Phase 1: Hyperparameter Tuning (Optuna)**

1.  **Add Dependency:** Add `optuna` to `requirements.txt`.
2.  **Identify Target Script:** Examine `src/training/train_agent.py` to determine integration strategy. (Done - see next step)
3.  **Define Search Space:** Identify key PPO hyperparameters (e.g., `learning_rate`, `n_steps`, `batch_size`, `gamma`, `ent_coef`, `net_arch`) and define search ranges/choices.
4.  **Prepare Validation Data:** Split training data or define a validation period.
5.  **Create Objective Function:**
    *   Accept Optuna `trial` object.
    *   Configure/train PPO agent with suggested hyperparameters on training data.
    *   Evaluate agent on validation data (e.g., final portfolio value, Sharpe ratio).
    *   Return the performance metric for Optuna to optimize.
6.  **Set Up Optuna Study:**
    *   Create Optuna `study` object (specify direction: maximize/minimize).
    *   Run `study.optimize()` with the objective function and number of trials.
7.  **Integrate into Workflow:**
    *   (Optional) Add `tune` mode to `main.py`.
    *   Update `train_agent.py` or config with best found hyperparameters.

**Future Phases (Potential):**

*   Walk-Forward Backtesting
*   Training Monitoring (TensorBoard/MLflow)
*   Enhanced Risk Management (Stop-loss, Position Sizing)
*   Environment Improvements (State Representation, Reward Function, Costs/Slippage)
*   Order Execution Logic (Limit Orders)
*   Ensemble Methods
