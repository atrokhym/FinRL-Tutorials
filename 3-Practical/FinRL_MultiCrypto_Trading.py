# %% [markdown]
# <a target="_blank" href="https://colab.research.google.com/github/AI4Finance-Foundation/FinRL-Tutorials/blob/master/3-Practical/FinRL_MultiCrypto_Trading.ipynb">
#   <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
# </a>

import time
import numpy as np
import warnings
import math

# Configure matplotlib for non-interactive backend
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates

warnings.filterwarnings("ignore", category=DeprecationWarning)
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import sys
sys.path.insert(0, "/var/tmp/FinRL-Meta")

import importlib
from meta.env_crypto_trading.env_multiple_crypto import MyCryptoEnv

# from agents import elegantrl_models
# importlib.reload(elegantrl_models)

from agents.elegantrl_models import DRLAgent as DRLAgent_erl
from agents.rllib_models import DRLAgent as DRLAgent_rllib
from agents.stablebaselines3_models import DRLAgent as DRLAgent_sb3
from meta.data_processor import DataProcessor


def train(start_date, end_date, ticker_list, data_source, time_interval, 
          technical_indicator_list, drl_lib, env, model_name, if_vix=True,
          **kwargs):
    
    #process data using unified data processor
    DP = DataProcessor(data_source, start_date, end_date, time_interval, **kwargs)
    price_array, tech_array, turbulence_array = DP.run(ticker_list,
                                                        technical_indicator_list, 
                                                        if_vix, cache=True)

    data_config = {'price_array': price_array,
                   'tech_array': tech_array,
                   'turbulence_array': turbulence_array}

    #build environment using processed data
    env_instance = env(config=data_config)

    #read parameters and load agents
    current_working_dir = kwargs.get('current_working_dir','./'+str(model_name))

    if drl_lib == 'elegantrl':
        break_step = max(kwargs.get('break_step', 1e6), MIN_TRAINING_STEPS)  # Ensure minimum steps
        erl_params = kwargs.get('erl_params')
        
        print("\nStarting ElegantRL training with params:")
        print(f"Learning rate: {erl_params['learning_rate']}")
        print(f"Batch size: {erl_params['batch_size']}")
        print(f"Net dimension: {erl_params['net_dimension']}")
        print(f"Target steps: {erl_params['target_step']}")
        print(f"Training steps: {break_step}")

        agent = DRLAgent_erl(env = env,
                           price_array = price_array,
                           tech_array=tech_array,
                           turbulence_array=turbulence_array)

        print("\nInitializing model...")
        model = agent.get_model(model_name, model_kwargs = erl_params)

        print("\nStarting training...")
        trained_model = agent.train_model(model=model,
                                        cwd=current_working_dir,
                                        total_timesteps=break_step)

        # Validate training metrics
        if hasattr(trained_model, 'eval_time'):  # Check if training completed
            print("\nValidating training results:")
            avg_reward = trained_model.eval_time  # Get average reward
            print(f"Average reward: {avg_reward}")
            
            if avg_reward < MIN_REWARD_THRESHOLD:
                print(f"WARNING: Low average reward ({avg_reward} < {MIN_REWARD_THRESHOLD})")
                print("Model may not have learned an effective strategy")
        
      
    elif drl_lib == 'rllib':
        total_episodes = kwargs.get('total_episodes', 100)
        rllib_params = kwargs.get('rllib_params')

        agent_rllib = DRLAgent_rllib(env = env,
                       price_array=price_array,
                       tech_array=tech_array,
                       turbulence_array=turbulence_array)

        model,model_config = agent_rllib.get_model(model_name)

        model_config['lr'] = rllib_params['lr']
        model_config['train_batch_size'] = rllib_params['train_batch_size']
        model_config['gamma'] = rllib_params['gamma']

        trained_model = agent_rllib.train_model(model=model, 
                                          model_name=model_name,
                                          model_config=model_config,
                                          total_episodes=total_episodes)
        trained_model.save(current_working_dir)
        
            
    elif drl_lib == 'stable_baselines3':
        total_timesteps = kwargs.get('total_timesteps', 1e6)
        agent_params = kwargs.get('agent_params')

        agent = DRLAgent_sb3(env = env_instance)

        model = agent.get_model(model_name, model_kwargs = agent_params)
        trained_model = agent.train_model(model=model, 
                                tb_log_name=model_name,
                                total_timesteps=total_timesteps)
        print('Training finished!')
        trained_model.save(current_working_dir)
        print('Trained model saved in ' + str(current_working_dir))
    else:
        raise ValueError('DRL library input is NOT supported. Please check.')


def test(start_date, end_date, ticker_list, data_source, time_interval,
          technical_indicator_list, drl_lib, env, model_name, if_vix=True,
          **kwargs):
    
    print("\nStarting test phase...")
    print(f"Test period: {start_date} to {end_date}")
    print(f"Model: {model_name} using {drl_lib}")
    
    # Process data using unified data processor
    DP = DataProcessor(data_source, start_date, end_date, time_interval, **kwargs)
    print("\nProcessing data with parameters:")
    print(f"Time interval: {time_interval}")
    print(f"Technical indicators: {technical_indicator_list}")
    print(f"Tickers: {ticker_list}")
    
    price_array, tech_array, turbulence_array = DP.run(ticker_list,
                                                       technical_indicator_list,
                                                       if_vix, cache=True)
    
    print("\nData Processing Results:")
    print(f"Price array shape: {price_array.shape}, dtype: {price_array.dtype}")
    print(f"Price array sample:\n{price_array[:5,:5]}")  # Show first 5 rows, 5 cols
    print(f"\nTech array shape: {tech_array.shape}, dtype: {tech_array.dtype}")
    print(f"Tech array sample:\n{tech_array[:5,:5]}")  # Show first 5 rows, 5 cols
    print(f"Turbulence array shape: {turbulence_array.shape if turbulence_array is not None else 'None'}")
    
    if np.any(np.isnan(price_array)):
        print("WARNING: Price array contains NaN values")
    if np.any(np.isnan(tech_array)):
        print("WARNING: Tech array contains NaN values")
    
    np.save('./price_array.npy', price_array)
    data_config = {'price_array': price_array,
                  'tech_array': tech_array,
                  'turbulence_array': turbulence_array}
    
    # Build environment using processed data
    env_config = {
        "price_array": price_array,
        "tech_array": tech_array,
        "turbulence_array": turbulence_array,
        "if_train": False,
    }
    env_instance = env(config=env_config)
    
    # Load parameters
    net_dimension = kwargs.get("net_dimension", 2 ** 7)
    current_working_dir = kwargs.get("current_working_dir", "./" + str(model_name))
    
    print(f"\nModel Configuration:")
    print(f"Net dimension: {net_dimension}")
    print(f"Working directory: {current_working_dir}")
    
    if drl_lib == "elegantrl":
        print("\nStarting ElegantRL prediction...")
        episode_total_assets = DRLAgent_erl.DRL_prediction(
            model_name=model_name,
            cwd=current_working_dir,
            net_dimension=net_dimension,
            environment=env_instance,
            env_args=env_config)
        
        # Validate prediction results
        if isinstance(episode_total_assets, (list, np.ndarray)):
            print("\nPrediction Results:")
            print(f"Total assets shape: {np.array(episode_total_assets).shape}")
            print(f"First few values: {episode_total_assets[:5]}")
            print(f"Last few values: {episode_total_assets[-5:]}")
            print(f"Min value: {np.min(episode_total_assets)}")
            print(f"Max value: {np.max(episode_total_assets)}")
            
            # Check for constant values
            unique_values = np.unique(episode_total_assets)
            if len(unique_values) == 1:
                print("\nWARNING: Prediction returned constant values!")
                print(f"Constant value: {unique_values[0]}")
        else:
            print(f"\nWARNING: Unexpected prediction result type: {type(episode_total_assets)}")
        
        return episode_total_assets

    elif drl_lib == "rllib":
        # load agent
        episode_total_assets = DRLAgent_rllib.DRL_prediction(
            model_name=model_name,
            env=env,
            price_array=price_array,
            tech_array=tech_array,
            turbulence_array=turbulence_array,
            agent_path=current_working_dir,
        )

        return episode_total_assets

    elif drl_lib == "stable_baselines3":
        episode_total_assets = DRLAgent_sb3.DRL_prediction_load_from_file(
            model_name=model_name, environment=env_instance, cwd=current_working_dir
        )

        return episode_total_assets
    else:
        raise ValueError("DRL library input is NOT supported. Please check.")


# Set parameters
TICKER_LIST = ['BTCUSDT','ETHUSDT','ADAUSDT','BNBUSDT','XRPUSDT',
                'SOLUSDT','DOTUSDT', 'DOGEUSDT','AVAXUSDT','UNIUSDT']
env = MyCryptoEnv
# Training period: 1 month for better learning
TRAIN_START_DATE = '2021-08-01'
TRAIN_END_DATE = '2021-09-01'

# Test on same period as before
TEST_START_DATE = '2021-09-21'
TEST_END_DATE = '2021-09-30'

INDICATORS = ['macd', 'rsi', 'cci', 'dx']

# PPO hyperparameters optimized for crypto volatility
ERL_PARAMS = {
    "learning_rate": 2**-13.5,      # Increased LR slightly (was 2**-14)
    "batch_size": 2**11,          
    "gamma": 0.99,               
    "seed": 312,
    "net_dimension": 2**10,       
    "target_step": 10000,         
    "eval_gap": 200,              
    "eval_times": 5,              
    "if_allow_break": False,      
    
    # PPO-specific parameters for crypto
    "eps_clip": 0.2,             # Reduced eps_clip slightly (was 0.25)
    "lambda_entropy": 0.02,       # Kept the same
    "lambda_gae": 0.97,           
    "repeat_times": 16,           
    "advantage_norm": True,        
    "recompute_advantage": True,   
    "ratio_clip": 0.25,            
    "value_clip": 0.4             
}

# Validation thresholds
MIN_TRAINING_STEPS = 1e6         # Increased minimum training (was 5e5)
#MIN_TRAINING_STEPS = 1e5
MIN_REWARD_THRESHOLD = -2.0      
MAX_LOSS_THRESHOLD = 50.0        


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    train(start_date=TRAIN_START_DATE, 
        end_date=TRAIN_END_DATE,
        ticker_list=TICKER_LIST, 
        data_source='binance',
        time_interval='5m',
        #time_interval='1h', # Reduced time interval         
        technical_indicator_list=INDICATORS,
        drl_lib='elegantrl', 
        env=env, 
        model_name='ppo', 
        current_working_dir='./test_ppo',
        erl_params=ERL_PARAMS,
        break_step=1e6,          # Increased training steps to match MIN_TRAINING_STEPS
        #break_step=1e5,          # Reduced training steps to match MIN_TRAINING_STEPS        
        if_vix=False,
        reward_scaling=80,      # Changed reward scaling
        )
    
    account_value_erl = test(start_date = TEST_START_DATE, 
                        end_date = TEST_END_DATE,
                        ticker_list = TICKER_LIST, 
                        data_source = 'binance',
                        time_interval= '5m',  # Match training interval
                        technical_indicator_list= INDICATORS,
                        drl_lib='elegantrl', 
                        env=env, 
                        model_name='ppo', 
                        current_working_dir='./test_ppo', 
                        net_dimension = 2**9, 
                        if_vix=False
                        )

    # duration_test = round((time.time() - start_time), 2)

    print("\nProcessing returns calculations...")
    
    # Validate and calculate agent returns
    account_value_erl = np.array(account_value_erl)
    print("\nAgent performance statistics:")
    print(f"Account values shape: {account_value_erl.shape}")
    print(f"Initial value: {account_value_erl[0]:.2f}")
    print(f"Final value: {account_value_erl[-1]:.2f}")
    
    if account_value_erl[0] == 0:
        raise ValueError("Initial account value is 0, cannot calculate returns")
        
    agent_returns = account_value_erl/account_value_erl[0]
    print(f"Agent returns range: {agent_returns.min():.4f} to {agent_returns.max():.4f}")
    
    # Calculate benchmark returns
    price_array = np.load('./price_array.npy')
    print(f"\nPrice array shape: {price_array.shape}")
    print("Available crypto columns:", price_array.shape[1])
    print("Expected tickers:", TICKER_LIST)
    
    # Ensure we only use available tickers
    available_tickers = TICKER_LIST[:price_array.shape[1]]
    print("Using tickers:", available_tickers)
    
    # BTC buy and hold returns
    btc_prices = price_array[:,0]
    if btc_prices[0] == 0:
        raise ValueError("Initial BTC price is 0, cannot calculate returns")
    buy_hold_btc_returns = btc_prices/btc_prices[0]
    print(f"BTC returns range: {buy_hold_btc_returns.min():.4f} to {buy_hold_btc_returns.max():.4f}")
    
    # Equal weight portfolio returns
    initial_prices = price_array[0,:]
    if np.any(initial_prices == 0):
        raise ValueError("Some initial prices are 0, cannot calculate equal weight returns")
        
    num_tickers = price_array.shape[1]
    equal_weight = np.array([1e5/initial_prices[i] for i in range(num_tickers)])
    equal_weight_values = []
    for i in range(0, price_array.shape[0]):
        equal_weight_values.append(np.sum(equal_weight * price_array[i]))
    equal_weight_values = np.array(equal_weight_values)
    equal_returns = equal_weight_values/equal_weight_values[0]

    # Plot results with error handling
    print("\nPlotting results...")
    try:
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), dpi=200)
        
        # Top subplot: Returns Comparison
        ax1.grid(True)
        ax1.grid(which='minor', axis='y')
        ax1.set_title('Strategy Returns Comparison', fontsize=16)
        
        # Plot returns
        if len(np.unique(agent_returns)) == 1:
            print(f"WARNING: Agent returns are constant at value: {agent_returns[0]}")
        elif np.any(np.isnan(agent_returns)):
            print("WARNING: Agent returns contain NaN values")
        else:
            ax1.plot(agent_returns, label='ElegantRL Agent', color='red')
            
        if not np.any(np.isnan(buy_hold_btc_returns)):
            ax1.plot(buy_hold_btc_returns, label='Buy-and-Hold BTC', color='blue')
        else:
            print("WARNING: BTC returns contain invalid values")
            
        if not np.any(np.isnan(equal_returns)):
            ax1.plot(equal_returns, label='Equal Weight Portfolio', color='green')
        else:
            print("WARNING: Equal weight returns contain invalid values")
            
        ax1.set_ylabel('Return', fontsize=14)
        ax1.set_xlabel('Time Steps (5min)', fontsize=14)
        ax1.legend(fontsize=10)
        
        # Bottom subplot: Crypto Prices
        ax2.grid(True)
        ax2.set_title('Individual Cryptocurrency Prices', fontsize=16)
        
        # Plot normalized prices for available cryptos
        available_cryptos = min(len(available_tickers), price_array.shape[1])
        print(f"\nPlotting prices for {available_cryptos} cryptocurrencies")
        
        for i in range(available_cryptos):
            prices = price_array[:, i]
            ticker = available_tickers[i]
            normalized_prices = prices / prices[0]  # Normalize to starting price
            ax2.plot(normalized_prices, label=ticker, alpha=0.7, linewidth=1.5)
        
        ax2.set_ylabel('Normalized Price', fontsize=14)
        ax2.set_xlabel('Time Steps (5min)', fontsize=14)
        ax2.legend(fontsize=8, ncol=2)
        
        plt.tight_layout()
        save_path = 'crypto_trading_results.png'
        plt.savefig(save_path, bbox_inches='tight')
        print(f"\nPlots saved to: {save_path}")
        
    except Exception as e:
        print(f"\nError during plotting: {str(e)}")
        print("Plotting failed, but analysis completed")
    finally:
        plt.close('all')  # Ensure all figures are closed
    