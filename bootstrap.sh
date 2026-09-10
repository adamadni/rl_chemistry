#!/usr/bin/env bash
# Build the Python environment for this project.
#
# Root is derived from this script's own location, so it works in a clone and
# on a machine where the checkout sits elsewhere. Override with RL_CHEM_ROOT --
# the same variable src/paths.py reads -- if data and checkpoints live off the
# checkout.
#
# The venv is created with --system-site-packages so an image that already
# ships a CUDA-matched torch is reused rather than downloading another one.
set -e
ROOT="${RL_CHEM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
echo "project root: $ROOT"

python -m venv --system-site-packages "$ROOT/venv"
source "$ROOT/venv/bin/activate"
pip install --upgrade pip

# Core runtime. torch is deliberately not installed here: use the build that
# matches the host CUDA, or let --system-site-packages supply the image's.
pip install rdkit scikit-learn pandas tqdm joblib numpy requests

python - <<'PY'
import rdkit, sklearn
try:
    import torch
    print(f"env OK | torch {torch.__version__} | cuda {torch.cuda.is_available()} | rdkit {rdkit.__version__}")
except ImportError:
    print(f"env OK | rdkit {rdkit.__version__} | NOTE: torch not found, install a CUDA-matched build")
PY

cat <<'EOF'

Optional extras, none of them required for the core pipeline:

  Exact environment   pip install -r requirements-lock.txt
  Docking             put a `smina` or `gnina` binary in ./docking/
                      plus: pip install biopython dimorphite-dl
  Free energy         a conda env with openfe (see src/run_rbfe.py)

  src/verify_pipeline.py compares this project's SMILES handling against
  OpenChem's. OpenChem is not vendored here:
      git clone https://github.com/Mariewelt/OpenChem refs/OpenChem
  It ships no __init__.py files, so `pip install -e .` is a no-op -- put the
  repo root on sys.path instead and let PEP 420 resolve openchem.*:
      SP=$(python -c "import site; print(site.getsitepackages()[0])")
      echo "$ROOT/refs/OpenChem" > "$SP/openchem_repo.pth"
EOF
