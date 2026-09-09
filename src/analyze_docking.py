"""Enrichment analysis of the docking benchmark: which score, if any, works.

Reports AUC with a bootstrap 95% interval and enrichment at 10% for every
available scoring scheme, per receptor and for the best-of-ensemble. The
interval is the point of the exercise: the first control run's AUC of 0.733
came from 15 pairs and its interval spanned 0.40-0.95, so it could not
distinguish a working protocol from a broken one. A number without an
interval cannot answer the question this script exists to answer.

Score directions differ and getting one backwards silently inverts the
result, so they are declared explicitly:
    affinity      kcal/mol, LOWER is better
    cnn_score     pose plausibility [0,1], HIGHER is better
    cnn_affinity  predicted pK, HIGHER is better

Usage:
    python src/analyze_docking.py --receptors 1IEP 2GQG
"""
import argparse, json, os
import numpy as np

FIELDS = [("affinity", False), ("cnn_score", True), ("cnn_affinity", True)]


def auc(pos, neg):
    """Mann-Whitney AUC; `pos`/`neg` already oriented so higher = better."""
    if not len(pos) or not len(neg):
        return float("nan")
    p = np.asarray(pos)[:, None]
    n = np.asarray(neg)[None, :]
    return float(((p > n).sum() + 0.5 * (p == n).sum()) / (p.size * n.size))


def boot_ci(pos, neg, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    pos, neg = np.asarray(pos), np.asarray(neg)
    vals = [auc(rng.choice(pos, len(pos), replace=True),
                rng.choice(neg, len(neg), replace=True)) for _ in range(n_boot)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def ef(pos, neg, frac=0.10):
    """Enrichment factor: actives in the top `frac` vs the base rate."""
    lab = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    sc = np.r_[pos, neg]
    order = np.argsort(-sc)
    k = max(1, int(round(frac * len(sc))))
    hit = lab[order][:k].mean()
    base = lab.mean()
    return float(hit / base) if base > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/workspace/rl_chemistry/docking")
    ap.add_argument("--receptors", nargs="+", default=["1IEP", "2GQG"])
    ap.add_argument("--pattern", default="bench_{r}.json",
                    help="filename template; use e.g. 'rd_{r}_s42.json' for the "
                         "re-dock with corrected pH 7.4 protonation")
    args = ap.parse_args()

    per = {}
    for r in args.receptors:
        path = os.path.join(args.dir, args.pattern.format(r=r))
        if not os.path.exists(path):
            print(f"missing {path}"); continue
        per[r] = {L["name"]: L for L in json.load(open(path))["ligands"]}

    names = sorted(set.intersection(*[set(v) for v in per.values()]))
    any_r = per[args.receptors[0]]
    cls = {n: any_r[n]["class"] for n in names}
    acts = [n for n in names if cls[n] == "benchmark_active"]
    inas = [n for n in names if cls[n] == "benchmark_inactive"]
    ok = [n for n in names if all(per[r][n].get("dock") for r in per)]
    print(f"{len(names)} ligands ({len(acts)} active / {len(inas)} inactive); "
          f"{len(ok)} docked successfully on all {len(per)} receptors\n")
    acts = [n for n in acts if n in ok]
    inas = [n for n in inas if n in ok]

    print("%-28s %7s %-16s %7s" % ("scheme", "AUC", "95% CI", "EF10%"))
    rows = []
    for field, higher in FIELDS:
        sign = 1.0 if higher else -1.0
        for r in per:
            g = lambda n: sign * per[r][n]["dock"][field]
            rows.append((f"{field} @ {r}", [g(n) for n in acts], [g(n) for n in inas]))
        # ensemble: best value across receptors, in the correct direction
        gb = lambda n: max(sign * per[r][n]["dock"][field] for r in per)
        rows.append((f"{field} @ ensemble", [gb(n) for n in acts], [gb(n) for n in inas]))
    for label, p, n in rows:
        a = auc(p, n)
        lo, hi = boot_ci(p, n)
        print("%-28s %7.3f  [%.3f, %.3f] %7.2f" % (label, a, lo, hi, ef(p, n)))

    print("\ntop 10 by best CNNaffinity across receptors:")
    scored = sorted(ok, key=lambda n: -max(per[r][n]["dock"]["cnn_affinity"] for r in per))
    for n in scored[:10]:
        best_r = max(per, key=lambda r: per[r][n]["dock"]["cnn_affinity"])
        d = per[best_r][n]["dock"]
        tag = "ACTIVE" if cls[n] == "benchmark_active" else "inactive"
        print(f"  {n:<16} {tag:<9} CNNaff {d['cnn_affinity']:6.3f}  "
              f"aff {d['affinity']:7.2f}  ({best_r}, pChEMBL {any_r[n]['pchembl']})")


if __name__ == "__main__":
    main()
