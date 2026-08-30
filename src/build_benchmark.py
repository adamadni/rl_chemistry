"""Build a property-matched actives/inactives benchmark for docking validation.

The first control run used 3 known drugs against 5 measured inactives -- 15
Mann-Whitney pairs, on which the observed AUC of 0.733 carried a 95% interval
spanning roughly 0.40-0.95. That is compatible with "useless" and with "good"
simultaneously, so it could not decide whether the docking protocol was fit to
score candidates. This builds a set large enough to answer that.

Two biases have to be designed out, or the AUC measures the wrong thing:

**Size bias.** Empirical docking scores are extensive in molecular size --
more heavy atoms means more favourable contact terms almost regardless of
binding. If actives are systematically larger than inactives, enrichment is
manufactured by arithmetic. Inactives are therefore greedily matched to
actives on heavy-atom count (and secondarily logP), one-to-one, and the
residual imbalance is reported so it can be checked rather than assumed.

**Analogue bias.** ChEMBL actives for a well-worked target cluster into a few
heavily-elaborated series. Sampling 40 actives at random would likely draw
many near-identical imatinib analogues, and a protocol that recognises one
recognises all of them -- inflating AUC while saying nothing about general
performance. Actives are therefore drawn one per Bemis-Murcko scaffold,
highest-potency first, so the set spans chemotypes rather than a few series.

Usage:
    python src/build_benchmark.py --n 40 --out docking/benchmark.json
"""
import argparse, json, sys
import numpy as np
sys.path.insert(0, "/workspace/rl_chemistry/src")
from train_rl import scaffold_of, DATA
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
RDLogger.DisableLog("rdApp.*")


def props(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return {"heavy": m.GetNumHeavyAtoms(), "mw": round(Descriptors.MolWt(m), 1),
            "logp": round(Crippen.MolLogP(m), 2),
            "rotb": rdMolDescriptors.CalcNumRotatableBonds(m)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--active-min-pchembl", type=float, default=8.0)
    ap.add_argument("--inactive-max-pchembl", type=float, default=5.5)
    ap.add_argument("--heavy-min", type=int, default=20)
    ap.add_argument("--heavy-max", type=int, default=45)
    ap.add_argument("--out", default="/workspace/rl_chemistry/docking/benchmark.json")
    args = ap.parse_args()

    data = json.load(open(f"{DATA}/abl1_qsar.json"))
    for d in data:
        d["props"] = props(d["smiles"])
    data = [d for d in data if d["props"]
            and args.heavy_min <= d["props"]["heavy"] <= args.heavy_max]

    # ---- actives: one per scaffold, most potent first (analogue-bias control)
    acts = sorted([d for d in data if d["pchembl"] >= args.active_min_pchembl],
                  key=lambda d: -d["pchembl"])
    picked_a, seen = [], set()
    for d in acts:
        s = scaffold_of(d["smiles"])
        if s is None or s in seen:
            continue
        seen.add(s)
        picked_a.append(d)
        if len(picked_a) >= args.n:
            break

    # ---- inactives: greedy 1:1 match on heavy atoms, tie-broken on logP
    pool = [d for d in data if d["pchembl"] <= args.inactive_max_pchembl and d["label"] == 0]
    used, picked_i = set(), []
    for a in picked_a:
        ha, la = a["props"]["heavy"], a["props"]["logp"]
        best, bd = None, None
        for j, d in enumerate(pool):
            if j in used:
                continue
            dist = (abs(d["props"]["heavy"] - ha), abs(d["props"]["logp"] - la))
            if bd is None or dist < bd:
                best, bd = j, dist
        if best is not None:
            used.add(best)
            picked_i.append(pool[best])

    ha = np.array([d["props"]["heavy"] for d in picked_a], dtype=float)
    hi = np.array([d["props"]["heavy"] for d in picked_i], dtype=float)
    la = np.array([d["props"]["logp"] for d in picked_a])
    li = np.array([d["props"]["logp"] for d in picked_i])
    print(f"actives   {len(picked_a):3d}  pChEMBL {min(d['pchembl'] for d in picked_a):.2f}"
          f"-{max(d['pchembl'] for d in picked_a):.2f}  "
          f"heavy {ha.mean():.1f}+-{ha.std():.1f}  logP {la.mean():.2f}")
    print(f"inactives {len(picked_i):3d}  pChEMBL {min(d['pchembl'] for d in picked_i):.2f}"
          f"-{max(d['pchembl'] for d in picked_i):.2f}  "
          f"heavy {hi.mean():.1f}+-{hi.std():.1f}  logP {li.mean():.2f}")
    print(f"residual imbalance: heavy {abs(ha.mean()-hi.mean()):.2f} atoms, "
          f"logP {abs(la.mean()-li.mean()):.2f}   ({len(seen)} distinct active scaffolds)")

    out = {"n_actives": len(picked_a), "n_inactives": len(picked_i),
           "match": {"heavy_mean_active": float(ha.mean()), "heavy_mean_inactive": float(hi.mean()),
                     "logp_mean_active": float(la.mean()), "logp_mean_inactive": float(li.mean())},
           "ligands": [{"name": d["chembl_id"], "smiles": d["smiles"], "pchembl": d["pchembl"],
                        "class": "benchmark_active", **d["props"]} for d in picked_a]
                    + [{"name": d["chembl_id"], "smiles": d["smiles"], "pchembl": d["pchembl"],
                        "class": "benchmark_inactive", **d["props"]} for d in picked_i]}
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
