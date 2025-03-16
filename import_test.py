import sys
sys.path.insert(0, '/tmp/FinRL-Meta')

try:
    # This is the import that was failing
    from agents.elegantrl_models_patched import DRLAgent as DRLAgent_erl
    print("Import successful!")
    
    # Create a simple test instance
    class DummyEnv:
        pass
    
    env = DummyEnv()
    price_array = []
    tech_array = []
    turbulence_array = []
    
    agent = DRLAgent_erl(env, price_array, tech_array, turbulence_array)
    print("DRLAgent instance created successfully!")
    
except Exception as e:
    print(f"Import failed with error: {e}")
