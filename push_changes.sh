#!/bin/bash

# Script to push the patched notebooks to GitHub

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "Error: git is not installed. Please install git first."
    exit 1
fi

# Check if we're in a git repository
if ! git rev-parse --is-inside-work-tree &> /dev/null; then
    echo "Error: Not in a git repository. Please run this script from within the FinRL-Tutorials repository."
    exit 1
fi

# Check if the patched notebooks exist
if [ ! -f "2-Advance/MultiCrypto_Trading_patched.ipynb" ] || [ ! -f "3-Practical/FinRL_MultiCrypto_Trading_patched.ipynb" ]; then
    echo "Error: Patched notebooks not found. Please run the patch_notebook.py script first."
    exit 1
fi

# Create a new branch for the changes
BRANCH_NAME="fix-equal-weight-calculation-$(date +%Y%m%d)"
echo "Creating a new branch: $BRANCH_NAME"
git checkout -b $BRANCH_NAME

# Copy the patched notebooks over the original ones
echo "Copying patched notebooks over the original ones..."
cp 2-Advance/MultiCrypto_Trading_patched.ipynb 2-Advance/MultiCrypto_Trading.ipynb
cp 3-Practical/FinRL_MultiCrypto_Trading_patched.ipynb 3-Practical/FinRL_MultiCrypto_Trading.ipynb

# Add the changes to git
echo "Adding changes to git..."
git add 2-Advance/MultiCrypto_Trading.ipynb 3-Practical/FinRL_MultiCrypto_Trading.ipynb README_FIXES.md patch_notebook.py fix_equal_weight.py push_changes.sh

# Commit the changes
echo "Committing changes..."
git commit -m "Fix equal weight calculation to handle mismatched ticker list length"

# Push the changes to origin
echo "Pushing changes to origin..."
git push origin $BRANCH_NAME

echo "Done! Changes pushed to origin/$BRANCH_NAME"
echo "You can now create a pull request on GitHub to merge these changes into the main branch."
echo "Visit: https://github.com/atrokhym/FinRL-Tutorials/pull/new/$BRANCH_NAME"
