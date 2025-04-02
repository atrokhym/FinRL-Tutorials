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

**Phase 2: Environment Improvements (Transaction Costs & Slippage) - In Progress**

1.  **Add Configuration:** Introduce parameters for `TRANSACTION_COST_PERCENT` and `SLIPPAGE_PERCENT` in `config/settings.py`.
2.  **Modify Environment (`src/environment/trading_env.py`):**
    *   Update the `step` method to calculate and subtract transaction costs based on trades.
    *   Implement a simple slippage model (e.g., percentage-based) and adjust execution prices.
    *   Ensure reward calculation/portfolio value reflects costs/slippage.
3.  **(Recommended) Re-run Tuning:** Execute `python main.py tune` to find optimal hyperparameters for the cost-aware environment.
4.  **Re-run Training:** Execute `python main.py train` using the new best parameters.
5.  **Re-run Backtesting:** Execute `python main.py backtest` to evaluate performance with costs/slippage.

**Future Phases (Potential):**

*   Walk-Forward Backtesting
*   Training Monitoring (TensorBoard/MLflow)
*   Enhanced Risk Management (Stop-loss, Position Sizing)
*   Order Execution Logic (Limit Orders)
*   Ensemble Methods
*   Further Environment Improvements (e.g., more sophisticated state, reward shaping)
