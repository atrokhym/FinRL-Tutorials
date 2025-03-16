# FinRL-Tutorials Fixes

This document explains the fixes implemented to address issues in the FinRL-Tutorials repository.

## Issues Fixed

1. **ElegantRL Integration Issues**
   - Missing `explore_rate` attribute in `AgentPPO` class
   - `_build_net` method not handling cases where `net_dims` has only one element
   - `Config` class not accepting constructor parameters
   - `DRL_prediction` method failing with missing model files and index out of bounds errors

2. **Equal Weight Portfolio Calculation Issue**
   - Index out of bounds error when the ticker list length is greater than the number of assets in the price array

## Fix Implementation

### ElegantRL Integration Fixes

The ElegantRL integration issues were fixed by creating the necessary directory structure and files:
- `/tmp/FinRL-Meta/elegantrl/train/`
- `/tmp/FinRL-Meta/elegantrl/agents/`
- `/tmp/FinRL-Meta/elegantrl/__init__.py`
- `/tmp/FinRL-Meta/elegantrl/agents/__init__.py`
- `/tmp/FinRL-Meta/elegantrl/train/__init__.py`
- `/tmp/FinRL-Meta/elegantrl/agents/AgentPPO_patched.py`
- `/tmp/FinRL-Meta/elegantrl/train/fixed_run.py`
- `/tmp/FinRL-Meta/elegantrl/train/config.py`
- `/tmp/FinRL-Meta/elegantrl/train/run.py`

These files include the necessary fixes for the ElegantRL integration issues.

### Equal Weight Portfolio Calculation Fix

The equal weight portfolio calculation issue was fixed by patching the notebooks to handle the case where the ticker list length is greater than the number of assets in the price array.

The patched notebooks now include this improved code:

```python
# Make sure we only use as many tickers as we have price data for
num_assets = min(price_array.shape[1], len(TICKER_LIST))
print(f"Using {num_assets} assets out of {len(TICKER_LIST)} tickers")

# Calculate equal weight (100k initial investment)
initial_investment = 1e5
# Equal weight means we invest initial_investment / num_assets in each asset
investment_per_asset = initial_investment / num_assets
# Calculate how many units of each asset we buy
equal_weight = np.array([investment_per_asset / initial_prices[i] for i in range(num_assets)])
```

And for the portfolio value calculation:

```python
# Get prices at time i
prices = price_array[i, :num_assets]
# Calculate portfolio value
portfolio_value = sum(equal_weight * prices)
```

## Scripts

### `patch_notebook.py`

This script automatically patches the notebooks with the fixed equal weight calculation code.

Usage:
```bash
python patch_notebook.py <notebook_path>
```

Example:
```bash
python patch_notebook.py 3-Practical/FinRL_MultiCrypto_Trading.ipynb
```

### `fix_equal_weight.py`

This script provides a function to safely calculate equal weight portfolio values.

Usage:
```python
from fix_equal_weight import fix_equal_weight_calculation

# Calculate equal weight portfolio values
equal_weight_values = fix_equal_weight_calculation(price_array, TICKER_LIST)
```

### `push_changes.sh`

This script helps push the changes to the GitHub repository.

Usage:
```bash
./push_changes.sh
```

## How to Use the Fixed Notebooks

1. Run the `patch_notebook.py` script to patch the notebooks:
   ```bash
   python patch_notebook.py 2-Advance/MultiCrypto_Trading.ipynb
   python patch_notebook.py 3-Practical/FinRL_MultiCrypto_Trading.ipynb
   ```

2. Open the patched notebooks (`*_patched.ipynb`) and run them from the beginning.

3. To push the changes to the GitHub repository, run the `push_changes.sh` script:
   ```bash
   ./push_changes.sh
   ```

4. Create a pull request on GitHub to merge the changes into the main branch.
