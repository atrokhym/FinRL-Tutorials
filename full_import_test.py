import sys
sys.path.insert(0, '/tmp/FinRL-Meta')

try:
    # These are the imports that were failing
    print("Testing imports...")
    from agents.elegantrl_models_patched import DRLAgent as DRLAgent_erl
    print("✓ elegantrl_models_patched import successful")
    
    # Try the other imports mentioned in the error
    try:
        from agents.rllib_models import DRLAgent as DRLAgent_rllib
        print("✓ rllib_models import successful")
    except Exception as e:
        print(f"✗ rllib_models import failed: {e}")
    
    try:
        from agents.stablebaselines3_models import DRLAgent as DRLAgent_sb3
        print("✓ stablebaselines3_models import successful")
    except Exception as e:
        print(f"✗ stablebaselines3_models import failed: {e}")
    
    # Create a simple test instance for the one that works
    class DummyEnv:
        pass
    
    env = DummyEnv()
    price_array = []
    tech_array = []
    turbulence_array = []
    
    agent = DRLAgent_erl(env, price_array, tech_array, turbulence_array)
    print("DRLAgent_erl instance created successfully!")
    
except Exception as e:
    print(f"Import failed with error: {e}")
