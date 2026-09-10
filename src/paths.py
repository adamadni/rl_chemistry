"""Project paths, resolved relative to the repository rather than hardcoded.

Every path in this project used to be an absolute `/workspace/rl_chemistry/...`
literal, which is fine on the machine the work ran on and useless anywhere
else -- a clone could not execute a single script. This module resolves the
project root once, and everything else derives from it.

Resolution order:
  1. `$RL_CHEM_ROOT` if set, for the case where data and checkpoints live
     somewhere other than the checkout (a mounted volume, a scratch disk).
  2. Otherwise the parent of the directory containing this file.

The fallback is what makes a fresh clone work unconfigured: `src/paths.py`
sits one level under the root, so `Path(__file__).parent.parent` is the
repository. That also happens to be correct on the original compute node,
where the checkout lived at `/workspace/rl_chemistry`, so neither environment
needs the variable set.

Paths are exported as `str`, not `Path`, because the call sites build filenames
with f-strings (`f"{DATA}/chembl_train.smi"`) and `os.path.join`. Both accept
either, but strings keep this a drop-in replacement with no behaviour change.

Directories that scripts write into are created on import; directories that
must already contain input data are not, so a missing corpus fails loudly at
the point of use instead of silently producing an empty result.
"""
import os
from pathlib import Path

_env = os.environ.get("RL_CHEM_ROOT")
ROOT = str(Path(_env).expanduser().resolve()) if _env else str(Path(__file__).resolve().parent.parent)

SRC = os.path.join(ROOT, "src")
DATA_DIR = os.path.join(ROOT, "data")
DATA = os.path.join(DATA_DIR, "processed")     # preprocessed corpus + ABL1 set
RAW = os.path.join(DATA_DIR, "raw")            # downloaded ChEMBL archives
CKPT = os.path.join(ROOT, "checkpoints")       # model weights (not in git)
RESULTS = os.path.join(ROOT, "results")        # metrics, committed
LOGS = os.path.join(ROOT, "logs")
DOCKING = os.path.join(ROOT, "docking")        # receptors, poses, docking output
RBFE = os.path.join(ROOT, "rbfe")              # free-energy inputs

for _d in (CKPT, RESULTS, LOGS, DOCKING):
    os.makedirs(_d, exist_ok=True)


def rel(*parts):
    """Path under the project root, e.g. rel('results', 'candidates_v5d.json')."""
    return os.path.join(ROOT, *parts)
