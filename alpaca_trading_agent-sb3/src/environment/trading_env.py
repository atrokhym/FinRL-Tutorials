# alpaca_trading_agent/src/environment/trading_env.py

import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
import logging

# NOTE: We inherit from object now, as we are replacing __init__ entirely
# from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Configuration Loading ---
import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
sys.path.insert(0, CONFIG_DIR)

try:
    import settings
except ImportError as e:
    logging.error(
        f"Error importing configuration: {e}. Ensure config/settings.py exists."
    )
    sys.exit(1)

# --- Define Feature List ---
INDICATORS = [
    "close",
    "high",
    "low",
    "trade_count",
    "open",
    "volume",
    "vwap",
    "macd",
    "rsi_14",
    "cci_14",
    "boll_ub",
    "boll_lb",
    # Turbulence is calculated separately
]
# Use the STATE_SPACE defined in settings, which should be correctly calculated
# using INDICATORS_WITH_TURBULENCE length
effective_state_space = settings.STATE_SPACE
# We still need the list of indicators *excluding* turbulence for state construction later
tech_indicator_list_for_state = settings.INDICATORS_WITH_TURBULENCE

logging.info(f"Using State Space size from settings: {effective_state_space}")
logging.info(f"Indicator list for state construction: {tech_indicator_list_for_state}")


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
        self.df = df  # Expects integer index, date column

        # --- Core Parameters (from settings and kwargs) ---
        self.stock_dim = settings.STOCK_DIM
        self.hmax = kwargs.get("hmax", 100)  # Max shares to trade per step
        self.initial_amount = kwargs.get(
            "initial_amount", settings.INITIAL_ACCOUNT_BALANCE
        )
        self.num_stock_shares = kwargs.get(
            "num_stock_shares", [0] * self.stock_dim
        )  # Initial holdings

        # --- Load Costs and Slippage from Settings ---
        self.transaction_cost_pct = kwargs.get(
            "transaction_cost_pct", settings.TRANSACTION_COST_PERCENT
        )
        self.slippage_pct = kwargs.get("slippage_pct", settings.SLIPPAGE_PERCENT)
        logging.info(
            f"Env using Transaction Cost: {self.transaction_cost_pct*100:.3f}%, Slippage: {self.slippage_pct*100:.3f}%"
        )
        # Remove old separate buy/sell cost lists if they exist
        # self.buy_cost_pct = kwargs.get('buy_cost_pct', [self.transaction_cost_pct] * self.stock_dim)
        # self.sell_cost_pct = kwargs.get('sell_cost_pct', [self.transaction_cost_pct] * self.stock_dim)
        # --- End Costs and Slippage ---

        self.reward_scaling = kwargs.get("reward_scaling", 1e-4)
        self.state_space_dim = (
            effective_state_space  # Use state space size from settings
        )
        self.action_space_dim = settings.ACTION_SPACE
        self.tech_indicator_list = (
            tech_indicator_list_for_state  # Use the list including turbulence
        )
        self.turbulence_threshold = kwargs.get(
            "turbulence_threshold",
            (
                settings.TURBULENCE_THRESHOLD
                if hasattr(settings, "TURBULENCE_THRESHOLD")
                else None
            ),
        )  # Load from settings
        self.risk_indicator_col = kwargs.get(
            "risk_indicator_col", "turbulence"
        )  # Default column name
        self.print_verbosity = kwargs.get("print_verbosity", 10)

        # --- Space Setup ---
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space_dim,))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space_dim,)
        )

        # --- Data & State Setup ---
        self.data = self.df.loc[
            self.day, :
        ]  # Initial slice (might be just one row initially, fixed below)
        self.terminal = False
        self.state = None  # Will be set by reset()
        self.reward = 0
        self.cost = 0
        self.asset_memory = [self.initial_amount]
        self.rewards_memory = []
        self.trades = 0
        self.episode = 0  # Not present in original, but useful for SB3 VecEnv

        # --- Find First Valid Start Date & Initialize State ---
        self.date_index = self.df.date.unique()
        first_valid_day_index = -1
        first_valid_date = None

        logging.info(
            "Searching for the first date with complete data for all tickers..."
        )
        for i, date in enumerate(self.date_index):
            daily_data = self.df[self.df["date"] == date]
            if len(daily_data["tic"].unique()) == self.stock_dim:
                first_valid_day_index = i
                first_valid_date = date
                logging.info(
                    f"Found first valid date: {first_valid_date} at index {first_valid_day_index}"
                )
                break

        if first_valid_day_index == -1:
            raise RuntimeError(
                "Could not find any date with complete data for all tickers."
            )

        # Set the starting day index
        self.day = first_valid_day_index

        # Set initial data slice for the first valid day
        logging.info(f"Setting initial self.data for date: {first_valid_date}")
        self.data = self.df[self.df["date"] == first_valid_date].reset_index(drop=True)
        logging.info(f"Initial self.data shape: {self.data.shape}")

        # Initialize the state correctly for the first valid day
        self.initial = True  # Flag for _initiate_state
        self.state = self._initiate_state()
        self.initial = False
        logging.info("Initial state calculated.")

        logging.info(
            f"AlpacaStockTradingEnv initialized successfully (Manual Replication, starting day {self.day})."
        )

    def _initiate_state(self):
        """Replicated from StockTradingEnv"""
        if self.initial:
            # For the first time, scan history (state_space steps)
            if len(self.df.date.unique()) > 1:
                # Use the data for the first valid day (self.day is already set)
                # Removed check: if self.day != 0: raise ValueError(...)
                # Ensure self.data contains all tickers for the first valid day
                if len(self.data["tic"].unique()) != self.stock_dim:
                    logging.error(
                        f"Data for initial day {self.day} is incomplete. Tickers found: {self.data['tic'].unique()}. Expected: {self.stock_dim}"
                    )
                    raise ValueError(
                        "Incomplete data for initial state calculation."
                    )  # Corrected indentation

                # Use the list including turbulence for state construction
                # Corrected: Use INDICATORS_WITH_TURBULENCE from settings
                features = (
                    self.data[settings.INDICATORS_WITH_TURBULENCE]
                    .values.flatten()
                    .tolist()
                )
                state = (
                    [self.initial_amount]  # Balance
                    + self.num_stock_shares  # Shares held
                    + features  # Features for all stocks on day 0
                )
                # print(f"Day {self.day} Initial State: len={len(state)}")
            # print(state)

            else:
                raise ValueError(
                    "Dataframe should have at least two dates to scan history"
                )
        else:
            # After the first step
            # Ensure self.data contains all tickers for the current day
            if len(self.data["tic"].unique()) != self.stock_dim:
                logging.error(
                    f"Data for day {self.day} is incomplete. Tickers found: {self.data['tic'].unique()}. Expected: {self.stock_dim}"
                )
                raise ValueError(
                    f"Incomplete data for state calculation on day {self.day}."
                )

            # Use the list including turbulence for state construction
            # Corrected: Use INDICATORS_WITH_TURBULENCE from settings
            features = (
                self.data[settings.INDICATORS_WITH_TURBULENCE].values.flatten().tolist()
            )
            state = (
                [self.state[0]]  # Previous balance (list)
                + self.state[
                    1 : self.stock_dim + 1
                ].tolist()  # Previous shares held (convert slice to list)
                + features  # Features for all stocks on current day (list)
            )
            # print(f"Day {self.day} Subsequent State: len={len(state)}")
            # print(state)

        # Verify state dimension using the correct list length from settings
        # Corrected: Use settings.INDICATORS_WITH_TURBULENCE directly
        expected_len = (
            1
            + self.stock_dim
            + self.stock_dim * len(settings.INDICATORS_WITH_TURBULENCE)
        )
        if len(state) != expected_len:
            logging.error(
                f"State length mismatch! Expected {expected_len}, got {len(state)}"
            )
            logging.error(
                f"Balance: 1, Shares: {self.stock_dim}, Features: {self.stock_dim * len(settings.INDICATORS_WITH_TURBULENCE)}"
            )
            raise RuntimeError("State dimension mismatch during calculation.")

        return np.array(state, dtype=np.float32)

    def _sell_stock(self, index, action):
        """Replicated from StockTradingEnv"""

        def _do_sell_normal():
            if self.state[index + 1] > 0:  # Check if shares > 0
                # Apply slippage to sell price (get slightly less)
                sell_price_after_slippage = self.data.loc[index, "close"] * (
                    1 - self.slippage_pct
                )

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
            # Fetch turbulence for the specific stock on the current day
            try:
                current_turbulence = self.data.loc[index, self.risk_indicator_col]
            except KeyError:
                logging.warning(
                    f"Turbulence column '{self.risk_indicator_col}' not found for index {index} on day {self.day}. Defaulting to 0."
                )
                current_turbulence = 0

            if current_turbulence >= self.turbulence_threshold:
                # Sell all shares if turbulence is high
                if self.state[index + 1] > 0:
                    logging.debug(
                        f"Turbulence high ({current_turbulence:.2f} >= {self.turbulence_threshold}). Forcing sell of {self.state[index + 1]} shares for index {index}."
                    )
                    # Apply slippage to sell price
                    sell_price_after_slippage = self.data.loc[index, "close"] * (
                        1 - self.slippage_pct
                    )
                    shares_to_sell = self.state[index + 1]  # Store before zeroing

                    # Calculate amount before transaction cost
                    gross_sell_amount = sell_price_after_slippage * shares_to_sell

                    # Calculate transaction cost
                    transaction_cost = gross_sell_amount * self.transaction_cost_pct

                    # Calculate final amount received
                    net_sell_amount = gross_sell_amount - transaction_cost

                    # Update balance
                    self.state[0] += net_sell_amount
                    # Update shares
                    self.state[index + 1] = 0  # Zero out shares
                    # Update total cost tracker
                    self.cost += transaction_cost
                    self.trades += 1
                else:
                    shares_to_sell = 0  # No shares were held to sell
                    sell_num_shares = 0  # Should this be shares_to_sell = 0? No, sell_num_shares is fine here.
                return shares_to_sell  # Return the stored number of shares sold
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
            buy_price_after_slippage = self.data.loc[index, "close"] * (
                1 + self.slippage_pct
            )

            # Calculate available shares we *could* buy based on price *after* slippage but *before* cost
            # (Cost is deducted separately later)
            if buy_price_after_slippage <= 0:  # Avoid division by zero
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
                buy_num_shares = max(
                    0, buy_num_shares - 1
                )  # Reduce shares by 1 if initial check failed
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
                self.state[index + 1] += buy_num_shares  # Corrected index
                # Update total cost tracker
                self.cost += transaction_cost
                self.trades += 1

            return buy_num_shares

        # Perform buy action
        if self.turbulence_threshold is None:
            buy_num_shares = _do_buy()
        else:
            # Fetch turbulence for the specific stock on the current day
            try:
                current_turbulence = self.data.loc[index, self.risk_indicator_col]
            except KeyError:
                logging.warning(
                    f"Turbulence column '{self.risk_indicator_col}' not found for index {index} on day {self.day}. Defaulting to 0."
                )
                current_turbulence = 0

            if current_turbulence < self.turbulence_threshold:
                buy_num_shares = _do_buy()
            else:
                logging.debug(
                    f"Turbulence high ({current_turbulence:.2f} >= {self.turbulence_threshold}). Preventing buy for index {index}."
                )
                buy_num_shares = 0  # Don't buy if turbulence is high
        return buy_num_shares

    def step(self, actions):
        """Replicated and modified from StockTradingEnv"""
        self.terminal = self.day >= len(self.date_index) - 1
        if self.terminal:
            # Calculate final portfolio value using current day's prices (self.data holds the last day's data here)
            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])  # Shares
                * self.data["close"].values  # Use current day's close prices
            )

            # End of episode printout (Corrected calculations)
            if self.print_verbosity > 0:
                print(f"Episode end. Day: {self.day}")
                # Use the correctly calculated end_total_asset
                print(f"Final Portfolio Value: {end_total_asset :0.2f}") # Corrected Label
                # Calculate reward based on the correct end_total_asset
                print(f"Overall PnL: {end_total_asset - self.initial_amount :0.2f}") # Corrected Label
                print(f"Total Cost: {self.cost :0.2f}")
                print(f"Total Trades: {self.trades}")

            # Calculate Sharpe Ratio for info dict
            df_total_value = pd.DataFrame(self.asset_memory)
            df_total_value.columns = ["account_value"]
            # Ensure index matches length of asset memory before assigning dates
            if len(self.date_index) >= len(self.asset_memory):
                df_total_value["date"] = self.date_index[
                    : len(self.asset_memory)
                ]  # Add dates
            else:
                # Handle case where date_index might be shorter (e.g., if episode ended prematurely)
                logging.warning(
                    "Length mismatch between date_index and asset_memory. Dates might be incorrect in final info."
                )
                df_total_value["date"] = pd.to_datetime(
                    np.arange(len(self.asset_memory))
                )  # Fallback

            df_total_value["daily_return"] = df_total_value["account_value"].pct_change(
                1
            )

            sharpe = (
                (
                    (252**0.5)
                    * df_total_value["daily_return"].mean()
                    / (
                        df_total_value["daily_return"].std() + 1e-9
                    )  # Add epsilon to prevent division by zero
                )
                if len(df_total_value["daily_return"].dropna()) > 1
                else 0
            )  # Need at least 2 returns

            # Use accumulated reward (sum of self.rewards_memory) for info dict reward
            # Note: self.reward only holds the reward for the *last* step
            cumulative_reward = sum(self.rewards_memory) if self.rewards_memory else 0

            info = {
                "episode": {
                    "r": cumulative_reward,  # Use accumulated reward
                    "l": len(self.rewards_memory),  # Episode length
                    "t": self.day,  # Current time step (final day index)
                },
                "total_asset": end_total_asset,
                "total_reward": end_total_asset - self.initial_amount,  # Overall PnL
                "total_cost": self.cost,
                "total_trades": self.trades,
                "sharpe": sharpe,
                # 'asset_memory': self.asset_memory, # Can be large, omit by default
                # 'rewards_memory': self.rewards_memory, # Can be large, omit by default
            }
            # For SB3 VecEnv compatibility, reset internal state when terminal
            # self.reset() # Avoid calling reset here, VecEnv handles it.
            return (
                self.state,
                self.reward,
                self.terminal,
                False,
                info,
            )  # Return False for truncated

        else:
            # Process actions
            actions = actions * self.hmax  # Scale actions to represent shares
            actions = actions.astype(int)  # Convert to integer shares

            # Ensure data for the current day is loaded correctly
            current_date = self.date_index[self.day]
            self.data = self.df[self.df["date"] == current_date].reset_index(drop=True)
            if len(self.data) != self.stock_dim:
                logging.error(
                    f"Data error on day {self.day} ({current_date}). Expected {self.stock_dim} rows, got {len(self.data)}"
                )
                raise RuntimeError(f"Inconsistent data for date {current_date}")

            begin_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])  # Shares
                * self.data["close"].values  # Use current day's close prices
            )

            # Trading logic
            argsort_actions = np.argsort(actions)
            sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                # Pass action directly, sell_stock handles absolute value and sign
                self._sell_stock(index, actions[index])

            for index in buy_index:
                # Pass action directly, buy_stock expects positive value
                self._buy_stock(index, actions[index])

            # self.trades count is updated within _buy_stock and _sell_stock

            # Move to next day
            self.day += 1
            # Ensure we don't go past the last date index
            if self.day >= len(self.date_index):
                self.terminal = True
                # Calculate final values based on the state *after* last trade but before incrementing day
                end_total_asset = self.state[0] + sum(
                    np.array(self.state[1 : (self.stock_dim + 1)])
                    * self.data["close"].values
                )
                self.reward = (
                    end_total_asset - begin_total_asset
                ) * self.reward_scaling  # Reward for the final step
                self.rewards_memory.append(self.reward)
                self.asset_memory.append(end_total_asset)
                # Construct final info dict
                cumulative_reward = (
                    sum(self.rewards_memory) if self.rewards_memory else 0
                )
                info = {
                    "episode": {
                        "r": cumulative_reward,
                        "l": len(self.rewards_memory),
                        "t": self.day - 1,
                    },  # Day before increment
                    "total_asset": end_total_asset,
                    "total_reward": end_total_asset - self.initial_amount,
                    "total_cost": self.cost,
                    "total_trades": self.trades,
                }
                return self.state, self.reward, self.terminal, False, info

            # Load next day's data
            next_date = self.date_index[self.day]
            next_day_data = self.df[self.df["date"] == next_date].reset_index(drop=True)
            if len(next_day_data) != self.stock_dim:
                logging.error(
                    f"Data error loading next day {self.day} ({next_date}). Expected {self.stock_dim} rows, got {len(next_day_data)}"
                )
                # Handle potentially ending the episode if data is inconsistent
                self.terminal = True
                end_total_asset = self.state[0] + sum(
                    np.array(self.state[1 : (self.stock_dim + 1)])
                    * self.data["close"].values
                )  # Value based on current data
                self.reward = (
                    end_total_asset - begin_total_asset
                ) * self.reward_scaling
                self.rewards_memory.append(self.reward)
                self.asset_memory.append(end_total_asset)
                cumulative_reward = (
                    sum(self.rewards_memory) if self.rewards_memory else 0
                )
                info = {
                    "episode": {
                        "r": cumulative_reward,
                        "l": len(self.rewards_memory),
                        "t": self.day - 1,
                    },  # Day before increment
                    "total_asset": end_total_asset,
                    "total_reward": end_total_asset - self.initial_amount,
                    "total_cost": self.cost,
                    "total_trades": self.trades,
                }
                return self.state, self.reward, self.terminal, False, info

            self.data = next_day_data  # Assign loaded data

            # Update state vector with new day's data features
            # self.state = self._initiate_state() # Incorrect usage, _initiate_state is for day 0
            # Correct state update:
            features = (
                self.data[tech_indicator_list_for_state].values.flatten().tolist()
            )
            self.state = np.array(
                [self.state[0]]  # Current balance
                + self.state[1 : self.stock_dim + 1].tolist()  # Current shares
                + features,  # New day's features
                dtype=np.float32,
            )

            # Calculate reward (change in portfolio value) based on NEW day's prices
            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])  # Shares
                * self.data["close"].values  # Use NEW day's close prices
            )
            self.reward = (end_total_asset - begin_total_asset) * self.reward_scaling
            self.rewards_memory.append(self.reward)
            self.asset_memory.append(end_total_asset)

            if self.print_verbosity > 10:  # Print step info only if very verbose
                print(f"Day: {self.day}, Action: {actions}")
                print(f"Total Asset: {end_total_asset:0.2f}")
                print(f"Reward: {self.reward:0.2f}")
                print(f"Cost: {self.cost:0.2f}")

        # info dict for non-terminal steps (can be minimal or include step-specific info)
        info = {}
        # Return state, reward, terminated, truncated, info
        return self.state, self.reward, self.terminal, False, info

    def reset(self, *, seed=None, options=None):
        """Replicated from StockTradingEnv, adjusted for potentially non-zero start day"""
        super().reset(seed=seed)  # Set seed if using gym > 0.26

        # Find the first valid start day index again (could be stored, but safer to re-find)
        first_valid_day_index = -1
        for i, date in enumerate(self.date_index):
            daily_data = self.df[self.df["date"] == date]
            if len(daily_data["tic"].unique()) == self.stock_dim:
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
        # --- Explicitly reset portfolio shares to initial state ---
        self.num_stock_shares = [0] * self.stock_dim
        # --- End share reset ---

        # Reload data for day 0 and initiate state
        self.data = self.df[self.df["date"] == self.date_index[self.day]].reset_index(
            drop=True
        )
        # self.initial = True # Flag for _initiate_state - No longer needed with direct call
        self.state = self._initiate_state()  # Recalculate state for day 0
        # self.initial = False

        # Standard reset return: observation, info
        info = {}  # Can add initial info if needed
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
    processed_train_filename = (
        f"train_processed_{settings.TRAIN_START_DATE}_{settings.TRAIN_END_DATE}.csv"
    )
    processed_train_filepath = os.path.join(
        PROJECT_ROOT, "data", processed_train_filename
    )
    try:
        train_df = pd.read_csv(processed_train_filepath)
        train_df["date"] = pd.to_datetime(train_df["date"])
        logging.info(f"Loaded processed training data: {processed_train_filepath}")
    except FileNotFoundError:
        logging.error(
            f"Processed training data file not found: {processed_train_filepath}"
        )
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
        logging.info(
            f"Step successful. New observation shape: {obs.shape}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}"
        )
    except Exception as e:
        logging.error(f"Error during environment testing: {e}", exc_info=True)
    logging.info("--- Environment Test Finished ---")
