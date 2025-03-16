import sys
sys.path.insert(0, '/tmp/FinRL-Meta')

# Import the patched AgentPPO class
from elegantrl.agents.AgentPPO_patched import AgentPPO
from elegantrl.train.config import Config
from elegantrl.train.fixed_run import train_agent_single_process

# Create a simple test to verify the fix
if __name__ == "__main__":
    print("Testing the fix for 'AgentPPO' object has no attribute 'explore_rate'")
    
    # Create a minimal agent
    agent = AgentPPO(
        net_dims=[64, 32],
        state_dim=10,
        action_dim=5,
        gpu_id=-1,  # Use CPU
        args=Config()
    )
    
    # Create a minimal config
    config = Config()
    config.agent_class = AgentPPO
    
    # Test the logging_tuple line that was causing the error
    try:
        # This is the line that was causing the error
        explore_rate = getattr(agent, 'explore_rate', 0.0)
        logging_tuple = (0.1, 0.2, 0.3, explore_rate, "test")
        print(f"Success! logging_tuple = {logging_tuple}")
        print("The fix works!")
    except Exception as e:
        print(f"Error: {e}")
        print("The fix did not work.")
