#!/bin/bash

# Check if a commit message was provided
if [ -z "$1" ]; then
  echo "Usage: ./push_script.sh \"Your commit message\""
  exit 1
fi

# Add all changes
git add .

# Commit changes
git commit -m "$1"

# Push to the main branch
git push origin main

echo "Code pushed successfully!"
