# alpaca_trading_agent/src/environment/trading_env.py

import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
import logging
# NOTE: We inherit from object now, as we are replacing __init__ entirely
# from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration Loading ---
import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_DIR = os.path.join(PROJECT_ROOT, 'config')
sys.path.insert(0, CONFIG_DIR)

try:
    import settings
except ImportError as e:
    logging.error(f"Error importing configuration: {e}. Ensure config/settings.py exists.")
    sys.exit(1)

# --- Define Feature List ---
INDICATORS = [
    'close', 'high', 'low', 'trade_count', 'open', 'volume', 'vwap',
    'macd', 'rsi_14', 'cci_14', 'boll_ub', 'boll_lb'
]
if settings.NUM_STOCK_FEATURES != len(INDICATORS):
    logging.warning(f"Mismatch between settings.NUM_STOCK_FEATURES ({settings.NUM_STOCK_FEATURES}) and actual indicators ({len(INDICATORS)}). Using {len(INDICATORS)}.")
    effective_num_features = len(INDICATORS)
    effective_state_space = 1 + settings.STOCK_DIM + settings.STOCK_DIM * effective_num_features
else:
    effective_num_features = settings.NUM_STOCK_FEATURES
    effective_state_space = settings.STATE_SPACE

# --- Custom Environment Definition (Replicating StockTradingEnv Logic) ---
# We inherit from gym.Env directly now
class AlpacaStockTradingEnv(gym.Env):
    """
    Custom Stock Trading Environment for Alpaca data using FinRL logic.

    NOTE: This implementation replicates and modifies the core logic of
    finrl.meta.env_stock_trading.env_stocktrading.StockTradingEnv (v0.3.8)
    to work around initialization issues, avoiding calling super().__init__().
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, df, **kwargs):
        logging.info("Initializing AlpacaStockTradingEnv (Manual Replication)...")

        self.day = 0
        self.df = df # Expects integer index, date column

        # --- Core Parameters (from settings and kwargs) ---
        self.stock_dim = settings.STOCK_DIM
        self.hmax = kwargs.get('hmax', 100) # Max shares to trade per step
        self.initial_amount = kwargs.get('initial_amount', settings.INITIAL_ACCOUNT_BALANCE)
        self.num_stock_shares = kwargs.get('num_stock_shares', [0] * self.stock_dim) # Initial holdings

        # --- Load Costs and Slippage from Settings ---
        self.transaction_cost_pct = kwargs.get('transaction_cost_pct', settings.TRANSACTION_COST_PERCENT)
        self.slippage_pct = kwargs.get('slippage_pct', settings.SLIPPAGE_PERCENT)
        logging.info(f"Env using Transaction Cost: {self.transaction_cost_pct*100:.3f}%, Slippage: {self.slippage_pct*100:.3f}%")
        # Remove old separate buy/sell cost lists if they exist
        # self.buy_cost_pct = kwargs.get('buy_cost_pct', [self.transaction_cost_pct] * self.stock_dim)
        # self.sell_cost_pct = kwargs.get('sell_cost_pct', [self.transaction_cost_pct] * self.stock_dim)
        # --- End Costs and Slippage ---

        self.reward_scaling = kwargs.get('reward_scaling', 1e-4)
        self.state_space_dim = effective_state_space # Use calculated state space
        self.action_space_dim = settings.ACTION_SPACE
        self.tech_indicator_list = INDICATORS # Use our defined list
        self.turbulence_threshold = kwargs.get('turbulence_threshold', None)
        self.risk_indicator_col = kwargs.get('risk_indicator_col', 'turbulence') # Not used if threshold is None
        self.print_verbosity = kwargs.get('print_verbosity', 10)

        # --- Space Setup ---
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space_dim,))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space_dim,)
        )

        # --- Data & State Setup ---
        self.data = self.df.loc[self.day, :] # Initial slice (might be just one row initially, fixed below)
        self.terminal = False
        self.state = None # Will be set by reset()
        self.reward = 0
        self.cost = 0
        self.asset_memory = [self.initial_amount]
        self.rewards_memory = []
        self.trades = 0
        self.episode = 0 # Not present in original, but useful for SB3 VecEnv

        # --- Find First Valid Start Date & Initialize State ---
        self.date_index = self.df.date.unique()
        first_valid_day_index = -1
        first_valid_date = None

        logging.info("Searching for the first date with complete data for all tickers...")
        for i, date in enumerate(self.date_index):
            daily_data = self.df[self.df['date'] == date]
            if len(daily_data['tic'].unique()) == self.stock_dim:
                first_valid_day_index = i
                first_valid_date = date
                logging.info(f"Found first valid date: {first_valid_date} at index {first_valid_day_index}")
                break

        if first_valid_day_index == -1:
            raise RuntimeError("Could not find any date with complete data for all tickers.")

        # Set the starting day index
        self.day = first_valid_day_index

        # Set initial data slice for the first valid day
        logging.info(f"Setting initial self.data for date: {first_valid_date}")
        self.data = self.df[self.df['date'] == first_valid_date].reset_index(drop=True)
        logging.info(f"Initial self.data shape: {self.data.shape}")

        # Initialize the state correctly for the first valid day
        self.initial = True # Flag for _initiate_state
        self.state = self._initiate_state()
        self.initial = False
        logging.info("Initial state calculated.")

        logging.info(f"AlpacaStockTradingEnv initialized successfully (Manual Replication, starting day {self.day}).")


    def _initiate_state(self):
        """Replicated from StockTradingEnv"""
        if self.initial:
            # For the first time, scan history (state_space steps)
            if len(self.df.date.unique()) > 1:
                # Use the data for the first valid day (self.day is already set)
                # Removed check: if self.day != 0: raise ValueError(...)
                # Ensure self.data contains all tickers for the first valid day
                if len(self.data['tic'].unique()) != self.stock_dim:
                     logging.error(f"Data for initial day {self.day} is incomplete. Tickers found: {self.data['tic'].unique()}. Expected: {self.stock_dim}")
                     raise ValueError("Incomplete data for initial state calculation.")

                state = (
                    [self.initial_amount] # Balance
                    + self.num_stock_shares # Shares held
                    + self.data[self.tech_indicator_list].values.flatten().tolist() # Features for all stocks on day 0
                )
                # print(f"Day {self.day} Initial State: len={len(state)}")
                # print(state)

            else:
                raise ValueError("Dataframe should have at least two dates to scan history")
        else:
            # After the first step
            # Ensure self.data contains all tickers for the current day
            if len(self.data['tic'].unique()) != self.stock_dim:
                 logging.error(f"Data for day {self.day} is incomplete. Tickers found: {self.data['tic'].unique()}. Expected: {self.stock_dim}")
                 raise ValueError(f"Incomplete data for state calculation on day {self.day}.")

            state = (
                [self.state[0]] # Previous balance (list)
                + self.state[1 : self.stock_dim + 1].tolist() # Previous shares held (convert slice to list)
                + self.data[self.tech_indicator_list].values.flatten().tolist() # Features for all stocks on current day (list)
            )
            # print(f"Day {self.day} Subsequent State: len={len(state)}")
            # print(state)

        # Verify state dimension
        expected_len = 1 + self.stock_dim + self.stock_dim * len(self.tech_indicator_list)
        if len(state) != expected_len:
             logging.error(f"State length mismatch! Expected {expected_len}, got {len(state)}")
             logging.error(f"Balance: 1, Shares: {self.stock_dim}, Features: {self.stock_dim * len(self.tech_indicator_list)}")
             raise RuntimeError("State dimension mismatch during calculation.")

        return np.array(state, dtype=np.float32)


    def _sell_stock(self, index, action):
        """Replicated from StockTradingEnv"""
        def _do_sell_normal():
            if self.state[index + 1] > 0: # Check if shares > 0
                # Apply slippage to sell price (get slightly less)
                sell_price_after_slippage = self.data.loc[index, "close"] * (1 - self.slippage_pct)

                # Sell a proportion
                sell_num_shares = min(abs(action), self.state[index + 1])

                # Calculate amount before transaction cost
                gross_sell_amount = sell_price_after_slippage * sell_num_shares

                # Calculate transaction cost
                transaction_cost = gross_sell_amount * self.transaction_cost_pct

                # Calculate final amount received
                net_sell_amount = gross_sell_amount - transaction_cost

                # Update balance
                self.state[0] += net_sell_amount
                # Update shares
                self.state[index + 1] -= sell_num_shares
                # Update total cost tracker
                self.cost += transaction_cost
                self.trades += 1
            else:
                sell_num_shares = 0
            return sell_num_shares

        # Perform sell action based on the sign of the action
        if self.turbulence_threshold is not None:
             # Simplified turbulence check (original uses self.turbulence, which might not be set)
             current_turbulence = self.df.loc[self.day * self.stock_dim, 'turbulence'] if 'turbulence' in self.df.columns else 0
             if current_turbulence >= self.turbulence_threshold:
                 # Sell all shares if turbulence is high
                 if self.state[index + 1] > 0:
                     # Apply slippage to sell price
                     sell_price_after_slippage = self.data.loc[index, "close"] * (1 - self.slippage_pct)
                     shares_to_sell = self.state[index + 1] # Store before zeroing

                     # Calculate amount before transaction cost
                     gross_sell_amount = sell_price_after_slippage * shares_to_sell

                     # Calculate transaction cost
                     transaction_cost = gross_sell_amount * self.transaction_cost_pct

                     # Calculate final amount received
                     net_sell_amount = gross_sell_amount - transaction_cost

                     # Update balance
                     self.state[0] += net_sell_amount
                     # Update shares
                     self.state[index + 1] = 0 # Zero out shares
                     # Update total cost tracker
                     self.cost += transaction_cost
                     self.trades += 1
                 else:
                     shares_to_sell = 0 # No shares were held to sell
                     sell_num_shares = 0 # Should this be shares_to_sell = 0? No, sell_num_shares is fine here.
                 return shares_to_sell # Return the stored number of shares sold
             else:
                 # Sell normally if turbulence is low
                 sell_num_shares = _do_sell_normal()
                 return sell_num_shares
        else:
             # Sell normally if turbulence is disabled
             sell_num_shares = _do_sell_normal()
             return sell_num_shares

    def _buy_stock(self, index, action):
        """Replicated from StockTradingEnv, modified for costs/slippage"""
        def _do_buy():
            # Apply slippage to buy price (pay slightly more)
            buy_price_after_slippage = self.data.loc[index, "close"] * (1 + self.slippage_pct)

            # Calculate available shares we *could* buy based on price *after* slippage but *before* cost
            # (Cost is deducted separately later)
            if buy_price_after_slippage <= 0: # Avoid division by zero
                return 0
            available_shares = self.state[0] // buy_price_after_slippage

            # Determine actual shares to buy (limited by action and available funds)
            buy_num_shares = min(available_shares, action)

            # Calculate cost of shares before transaction cost
            gross_buy_amount = buy_price_after_slippage * buy_num_shares

            # Calculate transaction cost
            transaction_cost = gross_buy_amount * self.transaction_cost_pct

            # Calculate total amount deducted from balance
            total_buy_deduction = gross_buy_amount + transaction_cost

            # Check if we actually have enough cash for the purchase + cost
            if self.state[0] < total_buy_deduction:
                 # Not enough cash, recalculate shares based on total cost
                 # Max affordable cost = Balance / (price * (1+slip) * (1+cost))
                 # Simplified: Reduce shares slightly if needed, though the initial check should be mostly sufficient
                 # For simplicity here, if the exact calculation fails, we might buy slightly fewer shares or none.
                 # Let's try buying one less share if the check fails, as a simple heuristic
                 buy_num_shares = max(0, buy_num_shares -1) # Reduce shares by 1 if initial check failed
                 gross_buy_amount = buy_price_after_slippage * buy_num_shares
                 transaction_cost = gross_buy_amount * self.transaction_cost_pct
                 total_buy_deduction = gross_buy_amount + transaction_cost
                 # If still not enough, we buy zero
                 if self.state[0] < total_buy_deduction:
                     buy_num_shares = 0
                     total_buy_deduction = 0
                     transaction_cost = 0

            # Update balance only if shares > 0
            if buy_num_shares > 0:
                self.state[0] -= total_buy_deduction
                # Update shares
                self.state[index + 1] += buy_num_shares # Corrected index
                # Update total cost tracker
                self.cost += transaction_cost
                self.trades += 1

            return buy_num_shares

        # Perform buy action
        if self.turbulence_threshold is None:
            buy_num_shares = _do_buy()
        else:
            # Simplified turbulence check
            current_turbulence = self.df.loc[self.day * self.stock_dim, 'turbulence'] if 'turbulence' in self.df.columns else 0
            if current_turbulence < self.turbulence_threshold:
                buy_num_shares = _do_buy()
            else:
                buy_num_shares = 0 # Don't buy if turbulence is high
        return buy_num_shares

    def step(self, actions):
        """Replicated and modified from StockTradingEnv"""
        self.terminal = self.day >= len(self.date_index) - 1
        if self.terminal:
            # End of episode
            if self.print_verbosity > 0:
                print(f"Episode end. Day: {self.day}")
                print(f"Total Asset: {self.state[0] + sum(np.array(self.state[1 : (self.stock_dim + 1)]) * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])) :0.2f}") # Balance + sum(shares * price) - price needs to be fetched correctly
                print(f"Total Reward: {self.state[0] + sum(np.array(self.state[1 : (self.stock_dim + 1)]) * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])) - self.initial_amount :0.2f}")
                print(f"Total Cost: {self.cost :0.2f}")
                print(f"Total Trades: {self.trades}")
            # Calculate final portfolio value for info dict
            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)]) # Shares
                * self.data["close"].values # Use current day's close prices
            )
            df_total_value = pd.DataFrame(self.asset_memory)
            df_total_value.columns = ["account_value"]
            df_total_value["date"] = self.date_index[:len(self.asset_memory)] # Add dates
            df_total_value["daily_return"] = df_total_value["account_value"].pct_change(1)

            sharpe = (
                (252**0.5)
                * df_total_value["daily_return"].mean()
                / df_total_value["daily_return"].std()
            ) if df_total_value["daily_return"].std() != 0 else 0 # Handle zero std dev

            info = {
                'episode': {
                    'r': self.reward, # Use accumulated reward
                    'l': len(self.rewards_memory), # Episode length
                    't': self.day # Current time step
                },
                'total_asset': end_total_asset,
                'total_reward': end_total_asset - self.initial_amount,
                'total_cost': self.cost,
                'total_trades': self.trades,
                'sharpe': sharpe,
                # 'asset_memory': self.asset_memory, # Can be large
                # 'rewards_memory': self.rewards_memory, # Can be large
            }
            # Reset state to initial state? Or let VecEnv handle reset?
            # For SB3, returning terminated=True is enough.
            # self.state = self._initiate_state() # Re-initiate state? No, reset does this.
            return self.state, self.reward, self.terminal, False, info # Return False for truncated

        else:
            # Process actions
            actions = actions * self.hmax  # Scale actions to represent shares
            actions = actions.astype(int) # Convert to integer shares

            # Ensure data for the current day is loaded correctly
            current_date = self.date_index[self.day]
            self.data = self.df[self.df['date'] == current_date].reset_index(drop=True)
            if len(self.data) != self.stock_dim:
                 logging.error(f"Data error on day {self.day} ({current_date}). Expected {self.stock_dim} rows, got {len(self.data)}")
                 raise RuntimeError(f"Inconsistent data for date {current_date}")

            begin_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)]) # Shares
                * self.data["close"].values # Use current day's close prices
            )

            # Trading logic
            argsort_actions = np.argsort(actions)
            sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                actions[index] = self._sell_stock(index, actions[index]) * (-1)

            for index in buy_index:
                actions[index] = self._buy_stock(index, actions[index])

            self.trades += len(sell_index) + len(buy_index)

            # Move to next day
            self.day += 1
            next_date = self.date_index[self.day]
            self.data = self.df[self.df['date'] == next_date].reset_index(drop=True) # Load next day's data
            if len(self.data) != self.stock_dim:
                 logging.error(f"Data error on day {self.day} ({next_date}). Expected {self.stock_dim} rows, got {len(self.data)}")
                 # Handle potential end-of-data issue if day == len(date_index) - 1
                 if self.day >= len(self.date_index) - 1:
                      self.terminal = True # Mark as terminal if data is missing on last expected day
                      # Use previous day's state? Or calculate final value now?
                      # Let's calculate final value based on previous state and prices
                      end_total_asset = self.state[0] + sum(
                          np.array(self.state[1 : (self.stock_dim + 1)]) # Shares
                          * self.df[self.df['date'] == current_date]["close"].values # Use previous day's close
                      )
                      info = {'total_asset': end_total_asset} # Minimal info
                      return self.state, self.reward, self.terminal, False, info
                 else:
                      raise RuntimeError(f"Inconsistent data for date {next_date}")


            # Update state with new day's data
            self.state = self._initiate_state() # Calculate state for the new day

            # Calculate reward (change in portfolio value)
            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)]) # Shares
                * self.data["close"].values # Use current day's close prices
            )
            self.reward = (end_total_asset - begin_total_asset) * self.reward_scaling
            self.rewards_memory.append(self.reward)
            self.asset_memory.append(end_total_asset)

            if self.print_verbosity > 10: # Print step info only if very verbose
                print(f"Day: {self.day}, Action: {actions}")
                print(f"Total Asset: {end_total_asset:0.2f}")
                print(f"Reward: {self.reward:0.2f}")
                print(f"Cost: {self.cost:0.2f}")

        # info dict is typically empty during steps unless debugging
        info = {}
        # Return state, reward, terminated, truncated, info
        return self.state, self.reward, self.terminal, False, info

    def reset(self, *, seed=None, options=None):
        """Replicated from StockTradingEnv, adjusted for potentially non-zero start day"""
        super().reset(seed=seed) # Set seed if using gym > 0.26

        # Find the first valid start day index again (could be stored, but safer to re-find)
        first_valid_day_index = -1
        for i, date in enumerate(self.date_index):
            daily_data = self.df[self.df['date'] == date]
            if len(daily_data['tic'].unique()) == self.stock_dim:
                first_valid_day_index = i
                break
        if first_valid_day_index == -1:
             # This shouldn't happen if init succeeded, but handle defensively
             raise RuntimeError("Failed to find valid start date during reset.")

        # Reset internal state to the first valid day
        self.day = first_valid_day_index
        self.cost = 0
        self.trades = 0
        self.terminal = False
        self.rewards_memory = []
        self.asset_memory = [self.initial_amount]

        # Reload data for day 0 and initiate state
        self.data = self.df[self.df['date'] == self.date_index[self.day]].reset_index(drop=True)
        self.initial = True # Flag for _initiate_state
        self.state = self._initiate_state()
        self.initial = False

        # Standard reset return: observation, info
        info = {} # Can add initial info if needed
        return self.state, info

    def render(self, mode="human"):
        """Replicated from StockTradingEnv - Basic print"""
        print(f"Day: {self.day}")
        print(f"Balance: {self.state[0]:0.2f}")
        print(f"Shares: {self.state[1 : self.stock_dim + 1]}")
        # Calculate current portfolio value
        current_value = self.state[0] + sum(
            np.array(self.state[1 : (self.stock_dim + 1)]) * self.data["close"].values
        )
        print(f"Portfolio Value: {current_value:0.2f}")
        print(f"Total Reward (Cumulative): {sum(self.rewards_memory):0.2f}")
        print(f"Total Cost: {self.cost:0.2f}")
        print(f"Total Trades: {self.trades}")

    def close(self):
        # Optional cleanup
        pass

# --- Helper Function to Create Environment ---
# Remains the same, but now instantiates our manually replicated class
def create_env(df, env_config={}):
    env = AlpacaStockTradingEnv(df=df, **env_config)
    return env

# --- Example Usage (for testing) ---
# Remains the same
if __name__ == "__main__":
    logging.info("--- Testing Environment Creation ---")
    processed_train_filename = f"train_processed_{settings.TRAIN_START_DATE}_{settings.TRAIN_END_DATE}.csv"
    processed_train_filepath = os.path.join(PROJECT_ROOT, 'data', processed_train_filename)
    try:
        train_df = pd.read_csv(processed_train_filepath)
        train_df['date'] = pd.to_datetime(train_df['date'])
        logging.info(f"Loaded processed training data: {processed_train_filepath}")
    except FileNotFoundError:
        logging.error(f"Processed training data file not found: {processed_train_filepath}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading processed training data: {e}")
        sys.exit(1)

    try:
        env_instance = create_env(train_df)
        logging.info("Environment created successfully.")
        obs, info = env_instance.reset()
        logging.info(f"Reset successful. Initial observation shape: {obs.shape}")
        logging.info(f"Observation space: {env_instance.observation_space}")
        logging.info(f"Action space: {env_instance.action_space}")
        random_action = env_instance.action_space.sample()
        obs, reward, terminated, truncated, info = env_instance.step(random_action)
        logging.info(f"Step successful. New observation shape: {obs.shape}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
    except Exception as e:
        logging.error(f"Error during environment testing: {e}", exc_info=True)
    logging.info("--- Environment Test Finished ---")
