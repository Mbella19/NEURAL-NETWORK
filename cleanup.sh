#!/bin/bash

echo "Cleaning up training artifacts..."

# Directories to remove
rm -rf logs/
rm -rf checkpoints/
rm -rf models/
rm -rf wandb/
rm -rf data/processed/*
rm -rf data/features/*

# Recursive file/dir removal
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

echo "Cleanup complete!"
