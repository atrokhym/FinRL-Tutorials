import sys
import numpy as np
sys.path.insert(0, '/tmp/FinRL-Meta')

try:
    # Import the DRLAgent class
    from agents.elegantrl_models_patched import DRLAgent
    
    # Create a dummy environment and data
    class DummyEnv:
        env_num = 1
        
    env = DummyEnv()
    
    # Create dummy arrays
    price_array = np.ones((100, 2))  # 100 time steps, 2 stocks
    tech_array = np.ones((100, 2, 3))  # 100 time steps, 2 stocks, 3 technical indicators
    turbulence_array = np.ones(100)  # 100 time steps
    
    # Create the agent
    agent = DRLAgent(env, price_array, tech_array, turbulence_array)
    
    # Test get_model method
    model_kwargs = {
        "learning_rate": 1e-4,
        "batch_size": 64,
        "gamma": 0.99
    }
    
    print("Testing get_model method...")
    model = agent.get_model("ppo", model_kwargs=model_kwargs)
    
    print("Model created successfully!")
    print(f"Model agent_class: {model.agent_class}")
    print(f"Model if_off_policy: {model.if_off_policy}")
    print(f"Model learning_rate: {model.learning_rate}")
    
except Exception as e:
    print(f"Test failed with error: {e}")
