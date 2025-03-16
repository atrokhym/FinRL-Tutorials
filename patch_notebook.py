import json
import os
import sys

def patch_notebook(notebook_path):
    """
    Patch the notebook to fix the equal weight calculation issue
    
    Args:
        notebook_path: path to the notebook file
    """
    print(f"Patching notebook: {notebook_path}")
    
    # Read the notebook file
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Find the cell with the equal weight calculation
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'equal_weight = np.array([1e5/initial_prices[i] for i in range(len(TICKER_LIST))])' in source:
                print(f"Found equal weight calculation in cell {i}")
                
                # Replace the problematic code
                new_source = []
                for line in cell['source']:
                    if 'equal_weight = np.array([1e5/initial_prices[i] for i in range(len(TICKER_LIST))])' in line:
                        # Replace with the fixed code
                        new_source.append("# Make sure we only use as many tickers as we have price data for\n")
                        new_source.append("num_assets = min(price_array.shape[1], len(TICKER_LIST))\n")
                        new_source.append("print(f\"Using {num_assets} assets out of {len(TICKER_LIST)} tickers\")\n")
                        new_source.append("\n")
                        new_source.append("# Calculate equal weight (100k initial investment)\n")
                        new_source.append("initial_investment = 1e5\n")
                        new_source.append("# Equal weight means we invest initial_investment / num_assets in each asset\n")
                        new_source.append("investment_per_asset = initial_investment / num_assets\n")
                        new_source.append("# Calculate how many units of each asset we buy\n")
                        new_source.append("equal_weight = np.array([investment_per_asset / initial_prices[i] for i in range(num_assets)])\n")
                    else:
                        new_source.append(line)
                
                # Update the cell source
                notebook['cells'][i]['source'] = new_source
                print("Updated equal weight calculation")
                
                # Also fix any subsequent code that uses equal_weight
                for j in range(i+1, len(notebook['cells'])):
                    if notebook['cells'][j]['cell_type'] == 'code':
                        source = ''.join(notebook['cells'][j]['source'])
                        if 'for i in range(0, price_array.shape[0]):' in source and 'equal_weight_values.append' in source:
                            print(f"Found portfolio value calculation in cell {j}")
                            
                            # Replace the portfolio value calculation
                            new_source = []
                            for line in notebook['cells'][j]['source']:
                                if 'portfolio_value = sum(equal_weight * price_array[i])' in line:
                                    # Replace with the fixed code
                                    new_source.append("    # Get prices at time i\n")
                                    new_source.append("    prices = price_array[i, :num_assets]\n")
                                    new_source.append("    # Calculate portfolio value\n")
                                    new_source.append("    portfolio_value = sum(equal_weight * prices)\n")
                                else:
                                    new_source.append(line)
                            
                            # Update the cell source
                            notebook['cells'][j]['source'] = new_source
                            print("Updated portfolio value calculation")
                
                # Save the patched notebook
                patched_path = notebook_path.replace('.ipynb', '_patched.ipynb')
                with open(patched_path, 'w', encoding='utf-8') as f:
                    json.dump(notebook, f, indent=1)
                
                print(f"Saved patched notebook to {patched_path}")
                return True
    
    print("Could not find equal weight calculation in notebook")
    return False

if __name__ == '__main__':
    # Check if a notebook path was provided
    if len(sys.argv) > 1:
        notebook_path = sys.argv[1]
    else:
        # Try to find the notebook in the current directory
        notebooks = [f for f in os.listdir('.') if f.endswith('.ipynb') and 'MultiCrypto_Trading' in f]
        if notebooks:
            notebook_path = notebooks[0]
            print(f"Found notebook: {notebook_path}")
        else:
            # Try to find the notebook in the 2-Advance and 3-Practical directories
            for directory in ['2-Advance', '3-Practical']:
                if os.path.exists(directory):
                    notebooks = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.ipynb') and 'MultiCrypto_Trading' in f]
                    if notebooks:
                        notebook_path = notebooks[0]
                        print(f"Found notebook: {notebook_path}")
                        break
            else:
                print("Could not find a MultiCrypto_Trading notebook")
                sys.exit(1)
    
    # Patch the notebook
    patch_notebook(notebook_path)
