"""ABL1 reward: RF P(active) minus an ensemble-uncertainty penalty.

Reward hacking risk: RL will happily exploit whatever region of chemical
space the RF is wrong about, and a 500-tree forest is not calibrated off the
training distribution. Rather than a hard applicability-domain cutoff
(similarity to train set < threshold -> reward 0), which would also block
genuine scaffold-hopping, this penalizes reward by the RF's own per-tree
disagreement: high variance across trees means the forest itself doesn't
trust the prediction, and reward should reflect that regardless of how
structurally similar/dissimilar the molecule is to anything in training.
This still allows moderately novel scaffolds to score well as long as the
trees agree; it discourages the narrow regions where they don't.

  reward(smiles) = P(active) - LAMBDA_UNC * std_over_trees(P(active))   [NOT floor-clipped]
  reward(invalid SMILES) = INVALID_REWARD (default -1.0)

LAMBDA_UNC is a free coefficient (default 1.0, i.e. one-sigma penalty);
tune via `--lambda-unc` in train_rl.py once we see the actual variance scale
on generator output.

Deliberately NOT clipped to [0, 1] (changed 2026-08-24, see baseline-run
postmortem). Pretrained-prior samples typically score p_active~0.08,
uncertainty~0.25, so p_active - uncertainty < 0 for nearly every early
sample; clipping that to a floor of 0.0 made almost the entire early
reward landscape a literal tie, so the first sample with real signal
looked like an infinite outlier and REINFORCE collapsed onto it in one
gradient step. Leaving the negative tail intact gives real (if small)
contrast between "bad" and "less bad" from step one, and gives the
optimizer a real "push away from this" signal instead of "no signal
either way". Invalid SMILES get a fixed negative reward clearly below
anything a valid-but-bad molecule can score, so it never accidentally
ties with (or beats) real chemistry.
"""
import os, sys
import joblib
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import CKPT

CKPT_PATH = os.path.join(CKPT, "abl1_rf.joblib")
class RewardModel:
    def __init__(self, ckpt_path=CKPT_PATH, lambda_unc=1.0, invalid_reward=-1.0):
        ck = joblib.load(ckpt_path)
        self.model = ck["model"]
        self.radius = ck["radius"]
        self.nbits = ck["nbits"]
        self.threshold = ck["threshold"]
        self.scheme = ck.get("scheme", "?")
        self.lambda_unc = lambda_unc
        self.invalid_reward = invalid_reward
        self._gen = rdFingerprintGenerator.GetMorganGenerator(radius=self.radius, fpSize=self.nbits)
        self._trees = self.model.estimators_
    def _featurize(self, smiles_list):
        """Returns (X, valid_mask). Invalid entries get an all-zero row (masked out downstream)."""
        X = np.zeros((len(smiles_list), self.nbits), dtype=np.uint8)
        valid = np.zeros(len(smiles_list), dtype=bool)
        for i, s in enumerate(smiles_list):
            m = Chem.MolFromSmiles(s) if s else None
            if m is None:
                continue
            X[i] = self._gen.GetFingerprintAsNumPy(m)
            valid[i] = True
        return X, valid
    def score(self, smiles_list):
        """Batch score. Returns dict of float32 arrays, one entry per input SMILES:
          valid     - RDKit-parseable (1.0/0.0)
          p_active  - forest mean P(active); 0.0 for invalid
          uncertainty - std of P(active) across the 500 trees; 0.0 for invalid
          reward    - p_active - lambda_unc * uncertainty (unclipped); invalid_reward for invalid
        """
        n = len(smiles_list)
        X, valid = self._featurize(smiles_list)
        p_active = np.zeros(n, dtype=np.float32)
        uncertainty = np.zeros(n, dtype=np.float32)
        if valid.any():
            Xv = X[valid]
            # per-tree P(active) for the valid subset -> (n_trees, n_valid)
            tree_p = np.stack([t.predict_proba(Xv)[:, 1] for t in self._trees], axis=0)
            p_active[valid] = tree_p.mean(axis=0)
            uncertainty[valid] = tree_p.std(axis=0)
        reward = (p_active - self.lambda_unc * uncertainty).astype(np.float32)
        reward[~valid] = self.invalid_reward
        return {"valid": valid.astype(np.float32), "p_active": p_active,
                "uncertainty": uncertainty, "reward": reward}
if __name__ == "__main__":
    rm = RewardModel()
    print(f"loaded {CKPT_PATH} | scheme={rm.scheme} | threshold={rm.threshold} | lambda_unc={rm.lambda_unc}")
    known = {
        "imatinib  (ABL1 inhibitor)":  "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
        "dasatinib (ABL1 inhibitor)":  "Cc1nc(Nc2ncc(C(=O)Nc3c(C)cccc3Cl)s2)cc(N2CCN(CCO)CC2)n1",
        "nilotinib (ABL1 inhibitor)":  "Cc1cn(-c2cc(NC(=O)c3ccc(C)c(Nc4nccc(-c5cccnc5)n4)c3)cc(C(F)(F)F)c2)cn1",
        "aspirin   (negative control)": "CC(=O)Oc1ccccc1C(=O)O",
        "glucose   (negative control)": "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
        "garbage SMILES (invalid)":     "not_a_smiles(((",
        "wild macrocycle (novel-ish)":  "C1CCCCCCCCCCCCCCCCCCCCCCCCCCCCC1",
    }
    names, smis = list(known.keys()), list(known.values())
    out = rm.score(smis)
    print(f"\n{'molecule':<32}{'valid':>7}{'P(active)':>12}{'uncertainty':>13}{'reward':>9}")
    for i, name in enumerate(names):
        print(f"{name:<32}{out['valid'][i]:>7.0f}{out['p_active'][i]:>12.3f}"
              f"{out['uncertainty'][i]:>13.3f}{out['reward'][i]:>9.3f}")
