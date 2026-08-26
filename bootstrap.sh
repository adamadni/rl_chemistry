#!/usr/bin/env bash
# Rebuild the Python env on a fresh RunPod pod.
#
# /workspace is a persistent network volume, so code+data survive pod
# termination -- but the container disk (and the system python) do not.
# This script recreates the venv against whatever torch the new image ships.
set -e
ROOT=/workspace/rl_chemistry

python -m venv --system-site-packages $ROOT/venv     # reuse image's CUDA torch
source $ROOT/venv/bin/activate
pip install --upgrade pip
pip install rdkit scikit-learn pandas tqdm tensorboard networkx pyyaml matplotlib joblib

# OpenChem ships NO __init__.py files -- find_packages() finds nothing and
# `pip install -e .` is a no-op. Put the repo root on sys.path instead and let
# PEP 420 namespace packages resolve openchem.*
SP=$(python -c "import site; print(site.getsitepackages()[0])")
echo "$ROOT/refs/OpenChem" > "$SP/openchem_repo.pth"

python -c "import torch, rdkit, openchem.models.GenerativeRNN; \
print('env OK | torch', torch.__version__, '| cuda', torch.cuda.is_available())"
