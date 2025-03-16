#!/bin/bash

# Script to merge changes to master and push to origin

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

# Get the current branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "Current branch: $CURRENT_BRANCH"

# Check if we're on the fix-equal-weight-calculation branch
if [[ $CURRENT_BRANCH == fix-equal-weight-calculation* ]]; then
    # Copy the patched notebooks over the original ones if not already done
    echo "Copying patched notebooks over the original ones..."
    cp 2-Advance/MultiCrypto_Trading_patched.ipynb 2-Advance/MultiCrypto_Trading.ipynb
    cp 3-Practical/FinRL_MultiCrypto_Trading_patched.ipynb 3-Practical/FinRL_MultiCrypto_Trading.ipynb

    # Add the changes to git if not already done
    echo "Adding changes to git..."
    git add 2-Advance/MultiCrypto_Trading.ipynb 3-Practical/FinRL_MultiCrypto_Trading.ipynb README_FIXES.md patch_notebook.py fix_equal_weight.py push_changes.sh merge_to_master.sh

    # Commit the changes if not already done
    echo "Committing changes..."
    git commit -m "Fix equal weight calculation to handle mismatched ticker list length" || true

    # Switch to master branch
    echo "Switching to master branch..."
    git checkout master

    # Merge the fix branch into master
    echo "Merging $CURRENT_BRANCH into master..."
    git merge $CURRENT_BRANCH

    # Push the changes to origin
    echo "Pushing changes to origin/master..."
    git push origin master

    echo "Done! Changes pushed to origin/master"
else
    echo "Not on a fix-equal-weight-calculation branch. Current branch: $CURRENT_BRANCH"
    
    # Check if we're on master
    if [ "$CURRENT_BRANCH" = "master" ]; then
        # Copy the patched notebooks over the original ones
        echo "Copying patched notebooks over the original ones..."
        cp 2-Advance/MultiCrypto_Trading_patched.ipynb 2-Advance/MultiCrypto_Trading.ipynb
        cp 3-Practical/FinRL_MultiCrypto_Trading_patched.ipynb 3-Practical/FinRL_MultiCrypto_Trading.ipynb

        # Add the changes to git
        echo "Adding changes to git..."
        git add 2-Advance/MultiCrypto_Trading.ipynb 3-Practical/FinRL_MultiCrypto_Trading.ipynb README_FIXES.md patch_notebook.py fix_equal_weight.py push_changes.sh merge_to_master.sh

        # Commit the changes
        echo "Committing changes..."
        git commit -m "Fix equal weight calculation to handle mismatched ticker list length"

        # Push the changes to origin
        echo "Pushing changes to origin/master..."
        git push origin master

        echo "Done! Changes pushed to origin/master"
    else
        echo "Please switch to the master branch first:"
        echo "git checkout master"
        exit 1
    fi
fi
