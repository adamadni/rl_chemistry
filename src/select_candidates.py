"""Pick N generated candidates and N measured negatives for docking.

Produces the two sets that go into structure-based evaluation. Neither set is
docked here -- this only chooses molecules and records why each was chosen.

## Candidates

Sampled from a trained policy, then filtered and diversified:
  1. valid, and NOVEL -- absent from both the ChEMBL pretraining corpus and
     the ABL1 activity set. A "candidate" that is already a known ABL1
     compound proves nothing about the generator.
  2. P(active) >= --min-p under the QSAR reward model.
  3. inside the applicability domain (max-Tanimoto to train >= --min-ad),
     so the RF prediction is interpolation. This is enforced explicitly here
     rather than trusted to the RF-uncertainty penalty, which was shown on
     this project (v4a) to stay low while the policy drifted out of domain --
     ensemble agreement is not evidence of validity where the forest never
     looked.
  4. drug-like enough to dock meaningfully: MW/logP/HBD/HBA/rotatable bonds
     within Lipinski-ish bounds. A 900-Da floppy molecule will score well on
     most docking functions for reasons that have nothing to do with binding.
  5. mutually dissimilar: one per generic framework AND pairwise ECFP4
     Tanimoto below --max-sim against every candidate already chosen.

     The framework constraint alone is not sufficient, which is worth
     recording because the first run of this script proved it. Its five
     "distinct framework" picks were all one chemotype -- a tolyl-quinoline
     cyclopropanecarboxamide -- differing only in an amide substituent that
     happened to be cyclic (cyclopropane / THF / azetidine / thiazolidine
     dioxide). Murcko pulls those substituent rings into the scaffold, so
     each got its own framework while the actual binding core was identical.
     A framework count is a good aggregate diversity statistic and a bad
     selection criterion: at n=5 it is trivially gamed by decorating one
     core with different small rings. Whole-molecule fingerprint distance is
     what "these are different molecules to dock" actually requires.

## Negatives -- and why they are size-matched

Drawn from the ABL1 activity set itself: compounds with a MEASURED pChEMBL
well below the activity threshold, i.e. genuine experimental negatives
against this target rather than random decoys or unmeasured molecules.

They are then matched to the candidates on heavy-atom count. This matters
more than it looks: essentially every docking scoring function is extensive
in molecular size, so larger ligands accumulate more favourable contact terms
and score better almost regardless of whether they bind. Comparing candidates
against systematically smaller negatives would manufacture a positive result
from arithmetic alone. Matching heavy atoms removes the dominant confound
before any docking is run.

Usage:
    python src/select_candidates.py --policy <ckpt> [--n-sample 20000] [--k 5]
"""
import argparse, collections, json, sys
import numpy as np
sys.path.insert(0, "/workspace/rl_chemistry/src")
from eval_policy import load, sample
from reward_model import RewardModel
from train_rl import ADReference, DATA, CKPT
from diversity_filter import scaffold_key
from smiles_utils import canonicalize
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, rdFingerprintGenerator
RDLogger.DisableLog("rdApp.*")


def props(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return {"mw": Descriptors.MolWt(m), "logp": Crippen.MolLogP(m),
            "hbd": rdMolDescriptors.CalcNumHBD(m), "hba": rdMolDescriptors.CalcNumHBA(m),
            "rotb": rdMolDescriptors.CalcNumRotatableBonds(m),
            "heavy": m.GetNumHeavyAtoms(), "tpsa": rdMolDescriptors.CalcTPSA(m)}


def druglike(p, args):
    return (args.mw_min <= p["mw"] <= args.mw_max and p["logp"] <= args.logp_max
            and p["hbd"] <= 5 and p["hba"] <= 10 and p["rotb"] <= args.rotb_max)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=f"{CKPT}/rl_v5d/policy_latest.pt")
    ap.add_argument("--n-sample", type=int, default=20000)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--min-p", type=float, default=0.5)
    ap.add_argument("--min-ad", type=float, default=0.30)
    ap.add_argument("--max-sim", type=float, default=0.45,
                    help="reject a candidate whose ECFP4 Tanimoto to any already-"
                         "selected candidate reaches this")
    ap.add_argument("--mw-min", type=float, default=250.0)
    ap.add_argument("--mw-max", type=float, default=550.0)
    ap.add_argument("--logp-max", type=float, default=5.0)
    ap.add_argument("--rotb-max", type=int, default=10)
    ap.add_argument("--neg-max-pchembl", type=float, default=5.0,
                    help="measured pChEMBL ceiling for a compound to count as a negative")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rm = RewardModel()
    ad = ADReference()
    policy, vocab, step = load(args.policy)
    print(f"policy {args.policy} (step {step}); sampling {args.n_sample}...", flush=True)

    raw = sample(policy, vocab, args.n_sample, 100, 1.0)
    canon = sorted({c for c in (canonicalize(s) if s else None for s in raw) if c})
    train_set = set(open(f"{DATA}/chembl_train.smi").read().split())
    abl = json.load(open(f"{DATA}/abl1_qsar.json"))
    abl_set = {d["smiles"] for d in abl}
    novel = [c for c in canon if c not in train_set and c not in abl_set]
    sc = rm.score(novel)
    tan = ad.max_tanimoto(novel, [True] * len(novel))
    print(f"  {len(canon)} unique valid -> {len(novel)} novel", flush=True)

    pool = []
    for i, smi in enumerate(novel):
        if sc["p_active"][i] < args.min_p or tan[i] < args.min_ad:
            continue
        p = props(smi)
        if p is None or not druglike(p, args):
            continue
        pool.append({"smiles": smi, "p_active": float(sc["p_active"][i]),
                     "uncertainty": float(sc["uncertainty"][i]),
                     "reward": float(sc["reward"][i]), "max_tanimoto_train": float(tan[i]),
                     "framework": scaffold_key(smi, "generic"), **p})
    pool.sort(key=lambda d: -d["p_active"])
    print(f"  {len(pool)} pass P>={args.min_p}, AD>={args.min_ad}, drug-likeness", flush=True)

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    def fp(s):
        return gen.GetFingerprint(Chem.MolFromSmiles(s))
    picked, seen_fw, picked_fps = [], set(), []
    for d in pool:
        if d["framework"] in seen_fw:
            continue
        f = fp(d["smiles"])
        sims = [DataStructs.TanimotoSimilarity(f, g) for g in picked_fps]
        if sims and max(sims) >= args.max_sim:
            continue
        d["max_sim_to_other_candidates"] = float(max(sims)) if sims else 0.0
        seen_fw.add(d["framework"])
        picked.append(d)
        picked_fps.append(f)
        if len(picked) >= args.k:
            break

    # ---- negatives: measured-inactive, then size-matched to the candidates
    tgt = float(np.mean([d["heavy"] for d in picked])) if picked else 30.0
    negs = []
    for d in abl:
        if d["label"] != 0 or d["pchembl"] > args.neg_max_pchembl:
            continue
        p = props(d["smiles"])
        if p is None or not druglike(p, args):
            continue
        negs.append({"smiles": d["smiles"], "chembl_id": d["chembl_id"],
                     "pchembl": d["pchembl"], "n_measurements": d["n_measurements"], **p})
    ns = rm.score([d["smiles"] for d in negs])
    for i, d in enumerate(negs):
        d["p_active_rf"] = float(ns["p_active"][i])
        d["heavy_delta"] = abs(d["heavy"] - tgt)
    negs.sort(key=lambda d: (d["heavy_delta"], d["pchembl"]))
    picked_negs = negs[:args.k]

    def show(title, rows, cols):
        print(f"\n=== {title} ===")
        print("  " + "".join(f"{c:>{w}}" for c, w in cols))
        for r in rows:
            print("  " + "".join(f"{r[k]:>{w}.3f}" if isinstance(r[k], float) else f"{str(r[k]):>{w}}"
                                 for k, w in cols))
            print(f"      {r['smiles']}")

    show(f"{len(picked)} CANDIDATES (distinct generic frameworks)", picked,
         [("p_active", 10), ("uncertainty", 12), ("max_tanimoto_train", 20), ("mw", 8), ("logp", 7), ("heavy", 7)])
    show(f"{len(picked_negs)} MEASURED NEGATIVES (size-matched, mean heavy {tgt:.1f})", picked_negs,
         [("chembl_id", 16), ("pchembl", 9), ("p_active_rf", 12), ("mw", 8), ("logp", 7), ("heavy", 7)])

    if args.out:
        json.dump({"policy": args.policy, "step": step, "n_sample": args.n_sample,
                   "filters": {"min_p": args.min_p, "min_ad": args.min_ad,
                               "mw": [args.mw_min, args.mw_max], "logp_max": args.logp_max,
                               "rotb_max": args.rotb_max, "neg_max_pchembl": args.neg_max_pchembl},
                   "n_pool": len(pool), "target_heavy_atoms": tgt,
                   "candidates": picked, "negatives": picked_negs},
                  open(args.out, "w"), indent=1)
        print(f"\nwrote {args.out}")
