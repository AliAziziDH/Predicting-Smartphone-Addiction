#!/bin/bash
# scripts/provision_gpu.sh
# Remote GPU provisioning workflow using google-colab-cli
# Strictly honors our "Zero Local Footprint" doctrine.

set -e

echo "Allocating Nvidia Tesla T4 GPU cloud VM..."
colab new --gpu T4 -s s6e8

echo "Bootstrapping environment..."
colab install -s s6e8 lightgbm xgboost catboost pydantic optuna scipy scikit-learn kaggle

echo "Remote GPU environment provisioned successfully on session s6e8."
