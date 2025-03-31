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

3. **API Credentials for Alpaca**  
   To run the Alpaca integration, set the following environment variables:
   - `ALPACA_API_KEY`
   - `ALPACA_SECRET_KEY`
   - *Optional:* `ALPACA_BASE_URL` (defaults to `https://paper-api.alpaca.markets`)

   On Linux/Mac, you can set these in the terminal before running:
   ```
   export ALPACA_API_KEY="your_alpaca_api_key"
   export ALPACA_SECRET_KEY="your_alpaca_secret_key"
   export ALPACA_BASE_URL="https://paper-api.alpaca.markets"
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
