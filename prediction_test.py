import sys
import numpy as np
import os
sys.path.insert(0, '/tmp/FinRL-Meta')

try:
    # Import the necessary classes
    from agents.elegantrl_models_patched import DRLAgent, MODELS
    from elegantrl.train.config import Config
    
    # Create a dummy environment and data
    class DummyEnv:
        env_num = 1
        state_dim = 10
        action_dim = 2
        if_discrete = False
        initial_total_asset = 1000.0
        max_step = 10
        price_ary = np.ones((10, 2))  # 10 time steps, 2 stocks
        day = 0
        amount = 500.0
        stocks = np.ones(2)  # 2 stocks
        
        def reset(self):
            return np.zeros(self.state_dim)
            
        def step(self, action):
            self.day += 1
            return np.zeros(self.state_dim), 0.0, self.day >= self.max_step, {}
            
    env = DummyEnv()
    
    # Create dummy arrays
    price_array = np.ones((100, 2))  # 100 time steps, 2 stocks
    tech_array = np.ones((100, 2, 3))  # 100 time steps, 2 stocks, 3 technical indicators
    
    # Create a temporary directory for testing
    cwd = "/tmp/test_prediction"
    os.makedirs(cwd, exist_ok=True)
    
    # Create a dummy model file
    import torch
    import torch.nn as nn
    
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 2)
            
        def forward(self, x):
            return torch.ones(1, 2)  # Always return [1, 1]
    
    # Save the dummy model
    model = DummyModel()
    torch.save(model, f"{cwd}/act.pth")
    
    # Test DRL_prediction method
    print("Testing DRL_prediction method...")
    
    # Add price_array and tech_array to the environment
    env.price_array = price_array
    env.tech_array = tech_array
    
    # Test without env_args
    print("\nTesting without env_args...")
    try:
        # Add more debugging
        print("Creating a simplified version for debugging...")
        
        # Create a simplified version of DRL_prediction for debugging
        def simplified_prediction():
            # Create env_args
            env_args = {
                "price_array": price_array,
                "tech_array": tech_array,
            }
            
            # Extract dimensions
            stock_dim = env_args["price_array"].shape[1]
            state_dim = 1 + 2 + 3 * stock_dim + env_args["tech_array"].shape[1]
            action_dim = stock_dim
            
            # Create env_args_config
            env_args_config = {
                "env_num": 1,
                "env_name": "StockEnv",
                "state_dim": state_dim,
                "action_dim": action_dim,
                "if_discrete": False,
                "max_step": env_args["price_array"].shape[0] - 1,
                "config": env_args,
            }
            
            print(f"env_args_config: {env_args_config}")
            
            # Create Config
            args = Config(agent_class=MODELS["ppo"], env_class=env, env_args=env_args_config)
            print(f"args created: {args}")
            
            # Create a dummy episode_total_assets
            return [1000.0, 1100.0, 1200.0]
        
        # Run the simplified version
        episode_total_assets = simplified_prediction()
        print(f"Simplified version success! episode_total_assets: {episode_total_assets}")
        
        # Now try the actual DRL_prediction
        print("Now trying the actual DRL_prediction...")
        episode_total_assets = DRLAgent.DRL_prediction(
            model_name="ppo",
            cwd=cwd,
            net_dimension=2**9,
            environment=env
        )
        print(f"Success! episode_total_assets: {episode_total_assets}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error: {e}")
    
except Exception as e:
    print(f"Test failed with error: {e}")
