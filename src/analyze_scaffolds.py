"""Scaffold-level analysis of a policy's HIGH-REWARD output.

`eval_policy.py` reports diversity over the whole sample; this asks the
sharper question the project actually cares about: among the molecules the
reward model calls active, *how many genuinely distinct chemotypes* are
there, and are they new?

The distinction matters. A policy can look diverse globally (thousands of
scaffolds overall) while its high-reward tail is one core structure with
different substituents hung off it -- that is R-group enumeration, not
scaffold hopping, and only the latter is what "propose new scaffolds"
means. Reporting novelty without the scaffold-frequency breakdown below
would overstate the result: nearly every high-reward molecule can be
"novel" (absent from both training sets) while still sharing one core.

Usage:
    python src/analyze_scaffolds.py [--policy PATH] [--n 5000] [--threshold 0.5]
"""
import argparse, collections, json, sys
import numpy as np
sys.path.insert(0, "/workspace/rl_chemistry/src")
from eval_policy import load, sample
from reward_model import RewardModel
from train_rl import scaffold_of, ADReference, DATA, CKPT
from smiles_utils import canonicalize
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=f"{CKPT}/rl_baseline_v3/policy_latest.pt")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--threshold", type=float, default=0.5, help="P(active) cutoff for 'high reward'")
    ap.add_argument("--top", type=int, default=15, help="how many scaffolds to list")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rm = RewardModel()
    ad = ADReference()
    policy, vocab, step = load(args.policy)
    print(f"policy: {args.policy} (step {step}), sampling {args.n}...")
    smiles = sample(policy, vocab, args.n, 100, 1.0)
    canon = [canonicalize(x) if x else None for x in smiles]
    valid = [c for c in canon if c]
    sc = rm.score(valid)
    train_set = set(open(f"{DATA}/chembl_train.smi").read().split())
    abl_set = {d["smiles"] for d in json.load(open(f"{DATA}/abl1_qsar.json"))}
    hi = sorted({valid[i] for i in range(len(valid)) if sc["p_active"][i] >= args.threshold})
    scaf = collections.Counter(scaffold_of(x) for x in hi)
    novel = [x for x in hi if x not in abl_set and x not in train_set]
    mt = ad.max_tanimoto(hi, [True] * len(hi)) if hi else np.zeros(0)
    print(f"\n=== high-reward set: P(active) >= {args.threshold} ===")
    print(f"  unique molecules                  {len(hi)}")
    print(f"  unique Bemis-Murcko scaffolds     {len(scaf)}")
    print(f"  already in ABL1 activity set      {sum(1 for x in hi if x in abl_set)}/{len(hi)}")
    print(f"  already in ChEMBL pretraining set {sum(1 for x in hi if x in train_set)}/{len(hi)}")
    print(f"  NOVEL (in neither)                {len(novel)}/{len(hi)}")
    if len(mt):
        print(f"  max-Tanimoto to ChEMBL train      mean {mt.mean():.3f} "
              f"min {mt.min():.3f} max {mt.max():.3f}")
    top = scaf.most_common(args.top)
    dominant = top[0][1] / max(1, len(hi)) if top else 0
    print(f"\n  scaffold concentration: top scaffold covers {100*dominant:.0f}% of the high-reward set")
    print(f"\n  {'count':>6}  scaffold")
    for sm, c in top:
        print(f"  {c:>6}  {sm}")
    if args.out:
        json.dump({"threshold": args.threshold, "n_sampled": args.n,
                   "n_unique_high_reward": len(hi), "n_scaffolds": len(scaf),
                   "n_novel": len(novel),
                   "top_scaffold_share": float(dominant),
                   "scaffolds": [{"scaffold": s, "count": c} for s, c in scaf.most_common()],
                   "molecules": hi}, open(args.out, "w"), indent=1)
        print(f"\nwrote {args.out}")
