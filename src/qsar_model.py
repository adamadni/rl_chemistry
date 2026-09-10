"""ABL1 QSAR reward model: ECFP4 + random forest.

Activity threshold is pChEMBL >= 8 (the screening definition). Two labelling
schemes are evaluated:
  A (primary) active >= 8, inactive < 8            -- uses every compound
  B (margin)  active >= 8, inactive < 7, 7-8 dropped -- cleaner class boundary

Validated on a Bemis-Murcko SCAFFOLD split, not a random split: a random split
leaks close analogues across the boundary and overstates what the model will do
on generator output, which is the only regime the RL loop cares about.
"""
import json, os, sys
from collections import defaultdict

import numpy as np
import joblib
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score, confusion_matrix,
                             balanced_accuracy_score, matthews_corrcoef, precision_score,
                             recall_score, brier_score_loss)

RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA, CKPT
os.makedirs(CKPT, exist_ok=True)
SEED, RADIUS, NBITS = 42, 2, 2048          # ECFP4

_gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=NBITS)


def featurize(smiles_list):
    X, keep = [], []
    for i, s in enumerate(smiles_list):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        X.append(_gen.GetFingerprintAsNumPy(m)); keep.append(i)
    return np.array(X, dtype=np.uint8), keep


def scaffold_split(smiles, frac_train=0.8):
    """Group by Bemis-Murcko scaffold; largest groups to train. No scaffold spans both sides."""
    groups = defaultdict(list)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        sc = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False) if m else ""
        groups[sc].append(i)
    sets = sorted(groups.values(), key=lambda g: (-len(g), g[0]))
    n_train = int(frac_train * len(smiles))
    train, test = [], []
    for g in sets:
        (train if len(train) + len(g) <= n_train else test).extend(g)
    return np.array(train), np.array(test), len(groups)


def rf():
    return RandomForestClassifier(n_estimators=500, max_features="sqrt", n_jobs=-1,
                                  random_state=SEED, class_weight="balanced_subsample")


def report(name, y, p, thresh=0.5):
    pred = (p >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
           "mcc": matthews_corrcoef(y, pred), "balanced_acc": balanced_accuracy_score(y, pred),
           "precision": precision_score(y, pred, zero_division=0),
           "recall": recall_score(y, pred, zero_division=0),
           "brier": brier_score_loss(y, p)}
    print(f"\n--- {name} ---")
    print(f"  n = {len(y)}   actives = {int(y.sum())} ({100*y.mean():.1f}%)")
    print(f"  ROC-AUC      {out['roc_auc']:.4f}      PR-AUC (AP)  {out['pr_auc']:.4f}")
    print(f"  balanced acc {out['balanced_acc']:.4f}      MCC          {out['mcc']:.4f}")
    print(f"  precision    {out['precision']:.4f}      recall       {out['recall']:.4f}")
    print(f"  Brier        {out['brier']:.4f}  (calibration, lower=better)")
    print(f"  confusion    TN={tn} FP={fp} FN={fn} TP={tp}")
    return out


raw = json.load(open(f"{DATA}/abl1_qsar.json"))
print(f"loaded {len(raw)} ABL1 compounds "
      f"(pChEMBL {min(d['pchembl'] for d in raw):.1f} - {max(d['pchembl'] for d in raw):.1f})")

SCHEMES = {
    "A_primary": [d for d in raw],
    "B_margin":  [d for d in raw if d["pchembl"] >= 8 or d["pchembl"] < 7],
}

all_results = {}
for scheme, subset in SCHEMES.items():
    smi = [d["smiles"] for d in subset]
    y_all = np.array([int(d["pchembl"] >= 8) for d in subset])
    X, keep = featurize(smi)
    y = y_all[keep]
    smi = [smi[i] for i in keep]

    print("\n" + "=" * 70)
    print(f"SCHEME {scheme}: {len(y)} compounds, {y.sum()} active ({100*y.mean():.1f}%)")
    if scheme == "B_margin":
        print(f"  (dropped {len(raw)-len(subset)} compounds in the 7 <= pChEMBL < 8 grey band)")
    print("=" * 70)

    tr, te, n_scaf = scaffold_split(smi)
    print(f"scaffold split: {n_scaf} unique scaffolds -> train {len(tr)} / test {len(te)}")
    m = rf().fit(X[tr], y[tr])
    res = {"scaffold": report(f"{scheme} SCAFFOLD SPLIT (held-out)", y[te],
                              m.predict_proba(X[te])[:, 1])}

    if scheme == "A_primary":
        rng = np.random.RandomState(SEED)
        idx = rng.permutation(len(y)); cut = int(0.8 * len(y))
        m_r = rf().fit(X[idx[:cut]], y[idx[:cut]])
        res["random"] = report("A_primary RANDOM SPLIT (optimistic, for contrast)",
                               y[idx[cut:]], m_r.predict_proba(X[idx[cut:]])[:, 1])

        print("\n--- 5-FOLD STRATIFIED CV ---")
        aucs, aps = [], []
        for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y), 1):
            pk = rf().fit(X[a], y[a]).predict_proba(X[b])[:, 1]
            aucs.append(roc_auc_score(y[b], pk)); aps.append(average_precision_score(y[b], pk))
            print(f"  fold {k}: ROC-AUC {aucs[-1]:.4f}  PR-AUC {aps[-1]:.4f}")
        print(f"  mean ROC-AUC {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
        print(f"  mean PR-AUC  {np.mean(aps):.4f} +/- {np.std(aps):.4f}")
        res["cv"] = {"roc_auc_mean": float(np.mean(aucs)), "roc_auc_std": float(np.std(aucs)),
                     "pr_auc_mean": float(np.mean(aps)), "pr_auc_std": float(np.std(aps))}

        final = rf().fit(X, y)
        joblib.dump({"model": final, "radius": RADIUS, "nbits": NBITS, "threshold": 8.0,
                     "scheme": scheme, "n_train": len(y), "metrics": res},
                    f"{CKPT}/abl1_rf.joblib")
        print(f"\nsaved reward model -> {CKPT}/abl1_rf.joblib (all {len(y)} compounds)")

        print("\n--- sanity check: known ABL1 inhibitors vs negative control ---")
        known = {
            "imatinib  (ABL1 inhibitor)":  "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
            "dasatinib (ABL1 inhibitor)":  "Cc1nc(Nc2ncc(C(=O)Nc3c(C)cccc3Cl)s2)cc(N2CCN(CCO)CC2)n1",
            "nilotinib (ABL1 inhibitor)":  "Cc1cn(-c2cc(NC(=O)c3ccc(C)c(Nc4nccc(-c5cccnc5)n4)c3)cc(C(F)(F)F)c2)cn1",
            "aspirin   (negative control)": "CC(=O)Oc1ccccc1C(=O)O",
            "glucose   (negative control)": "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
        }
        for name, s in known.items():
            fp = _gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)).reshape(1, -1)
            print(f"  {name:<30} P(active) = {final.predict_proba(fp)[0,1]:.3f}")

    all_results[scheme] = res

json.dump(all_results, open(f"{CKPT}/qsar_metrics.json", "w"), indent=1)
print("\nQSAR_DONE")
