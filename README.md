# FinRL Trading Project

This project implements a trading framework using FinRL. It includes two distinct trading integrations:
 
1. **IBKR Integration** – The original implementation with Interactive Brokers.
2. **Alpaca Integration** – A parallel implementation using Alpaca as the trading provider.  
   (Note: The original Alpaca implementation in the `alpaca_trading_agent` directory remains unchanged. The Alpaca integration for this project is provided via new files in the `ibkr_trading_project` directory.)

---

## Project Structure

```
FinRL-Tutorials/
├── ibkr_trading_project/
│   ├── config.py               # IBKR configuration and settings
│   ├── config_alpaca.py        # Alpaca-specific configuration
│   ├── main.py                 # Main script for IBKR modes (training, testing, paper trading, etc.)
│   ├── main_alpaca.py          # Dedicated main script for Alpaca paper trading mode
│   ├── trading/
│   │   ├── ...                 # IBKR trading logic (unchanged)
│   │   └── alpaca_papertrader.py  # Alpaca paper trading implementation
│   ├── data/                   # Data storage directory
│   ├── training/               # Training logic and orchestration
│   └── ...                     # Other related modules/files (agent, env, pyfolio_patch, etc.)
├── alpaca_trading_agent/
│   └── ...                     # Original Alpaca project (should not be modified)
├── finrl_online_learning/      # Online Learning Demo
│   ├── main.py                 # Main script for the online learning demo
│   └── env_stocktrading_wrapper.py # Local wrapper for FinRL environment fix
└── README.md                   # This file
```

---

## Prerequisites

1. **Python 3.7+**  
2. Install required dependencies.

   - For IBKR-related components, install:
     ```
     pip install -r ibkr_trading_project/requirements.txt
     ```
   - For Alpaca-specific trading (if isolated testing is needed), you may also refer to:
     ```
     pip install -r ibkr_trading_project/requirements_alpaca.txt
     ```
   - Note: The `alpaca_trading_agent` folder has its own dependencies. Do not modify it.
   - For the `finrl_online_learning` demo, ensure `finrl`, `stable-baselines3`, `alpaca-trade-api`, `pandas`, `numpy`, `matplotlib`, `pytz`, `python-dotenv` are installed in your environment.

3. **API Credentials for Alpaca**  
   To run the Alpaca integration or the online learning demo, set the following environment variables:
   - `ALPACA_API_KEY`
   - `ALPACA_API_SECRET`
   - *Optional:* `ALPACA_API_BASE_URL` (defaults to `https://paper-api.alpaca.markets`)

   On Linux/Mac, you can set these in the terminal before running:
   ```
   export ALPACA_API_KEY="your_alpaca_api_key"
   export ALPACA_API_SECRET="your_alpaca_secret_key"
   export ALPACA_BASE_URL="https://paper-api.alpaca.markets"
   ```
   Alternatively, create a `.env` file in the project root (`FinRL-Tutorials/`) with the following content:
   ```
   ALPACA_API_KEY="your_alpaca_api_key"
   ALPACA_API_SECRET="your_alpaca_secret_key"
   # ALPACA_API_BASE_URL="https://paper-api.alpaca.markets" # Optional
   ```

---

## Running the Project

### IBKR Modes

The original IBKR integration remains unchanged. You can run the standard training, testing, retraining, or paper trading modes using the **IBKR main script**:

```bash
python ibkr_trading_project/main.py <mode> [options]
```

Where `<mode>` can be:
- `train` – to train your agent,
- `test` – to test the agent,
- `retrain` – to retrain the agent,
- `papertrade` – to run paper trading with IBKR.

Example (paper trading with IBKR):
```bash
python ibkr_trading_project/main.py papertrade --ib_host localhost --ib_port 7497
```

### Alpaca Integration

A separate Alpaca trading routine is provided. To run a paper trading session using Alpaca, use the dedicated Alpaca main script:

```bash
python ibkr_trading_project/main_alpaca.py papertrade [options]
```

Example:
```bash
python ibkr_trading_project/main_alpaca.py papertrade --tickers AAPL MSFT GOOG --model_dir ./papertrading_alpaca_erl
```

The `main_alpaca.py` script leverages:
- `config_alpaca.py` for Alpaca configuration,
- `trading/alpaca_papertrader.py` which implements the Alpaca paper trading session.

---

## Online Learning Demo (`finrl_online_learning`)

This directory contains a demonstration of an online/continual learning setup using FinRL and Alpaca.

### Workflow Explanation

1.  **Data Loading & Preparation (`main.py`)**:
    *   Loads initial historical daily data from Alpaca using `load_historical_data_from_alpaca`.
    *   Splits this data into initial training and validation sets.
2.  **Agent Initialization (`OnlineLearningAgent`)**:
    *   Initializes a `FeatureEngineer` to add technical indicators.
    *   Processes the raw training/validation data.
    *   Creates training and evaluation environments using a local `StockTradingEnvWrapper`. **This wrapper is crucial** as it fixes bugs in the original FinRL `StockTradingEnv` related to handling missing ticker data on specific days and ensures the state vector shape matches the defined observation space.
    *   Initializes an `AlpacaDataSource` to fetch new, near real-time data during the learning process.
    *   Initializes the `DRLAgent` and the underlying Stable Baselines3 model (e.g., PPO), correctly handling hyperparameters like `learning_rate`.
    *   Initializes an `OnlineLearningCallback`.
3.  **Initial Training (`initial_training`)**:
    *   Trains the model on the initial historical dataset (`train_env`).
4.  **Continual Learning (`continual_learning`)**:
    *   Continues the `model.learn()` process.
    *   The `OnlineLearningCallback` periodically triggers the `AlpacaDataSource` to fetch new minute-level data.
    *   If new data arrives, the callback processes it and calls `model.learn()` for a small number of steps to simulate adaptation based on recent market activity. (Note: This is a simplified simulation; a true online system might update the environment state more directly).
5.  **Evaluation (`evaluate_performance`)**:
    *   Evaluates the final trained model on the held-out validation set using the `StockTradingEnvWrapper`.

### Running the Demo

Ensure your Alpaca API keys are set (see Prerequisites). Then run:

```bash
python finrl_online_learning/main.py
```

### Fixes Applied

During development, several issues were identified and fixed:

*   **`ValueError: too many values to unpack`**: Corrected the usage of FinRL's `data_split` function, which acts as a filter, not a splitter. Manual splitting is now used.
*   **`TypeError: got an unexpected keyword argument 'transaction_cost_pct'`**: Fixed the `OnlineLearningAgent` to use the correct environment arguments (`buy_cost_pct`, `sell_cost_pct`) and filter invalid keys.
*   **`AttributeError: 'numpy.float64' object has no attribute 'values'` / `ValueError: could not broadcast input array...`**: Addressed by creating the `StockTradingEnvWrapper` which correctly handles state vector construction when data for some tickers is missing on a given day (by reindexing and filling NaNs).
*   **`ImportError: attempted relative import...` / `ModuleNotFoundError`**: Corrected the import of the wrapper class in `main.py` to use a direct import.
*   **`TypeError: DRLAgent.get_model() got an unexpected keyword argument 'learning_rate'`**: Fixed model initialization to apply hyperparameters like `learning_rate` correctly to the SB3 model instance, not the `DRLAgent` factory method.
*   **`RuntimeWarning: divide by zero encountered...`**: Fixed by overriding `_buy_stock` in the wrapper to check for non-positive prices before calculating purchase amounts.

---

## Data Fetching and Preprocessing

The project includes scripts (located in separate modules) for:
- **Data Fetching:** Uses Alpaca's API to download historical data (via `alpaca_trading_agent/src/data_fetcher/fetch_data.py`).
- **Data Preprocessing:** Processes raw data to calculate technical indicators (see `alpaca_trading_agent/src/preprocessing/preprocess_data.py`).

Before training or backtesting, ensure that the raw data is fetched and preprocessed correctly:
1. Run the data fetching script to download raw data.
2. Run the preprocessing script to generate the processed CSV files.

---

## Additional Notes

- **Alpaca Credentials and Environment Variables:**  
  Make sure your environment variables for Alpaca are set correctly before initiating any Alpaca paper trading sessions.

- **Model Loading:**  
  The current implementation of the Alpaca paper trader uses placeholder logic for model loading and action calculation. Integrate your trained RL model as needed.

- **Project Isolation:**  
  Do not modify the `alpaca_trading_agent` directory. All project-specific changes and integrations are contained within the `ibkr_trading_project` directory.

---

## Troubleshooting

- **API Connection Issues:**  
  If you encounter errors connecting to the Alpaca API, verify that your API keys are correct and that the base URL is set properly.

- **Data Issues:**  
  If preprocessing fails (e.g., missing columns), ensure that the fetched data meets the expected format and that you have the necessary indicator libraries installed (e.g., `stockstats`).

---

## Conclusion

This README provides the basic instructions to set up and run the FinRL Trading Project with both IBKR and Alpaca integrations. Customize the trading scripts and configurations further based on your specific requirements and trading strategy.

Happy Trading!
