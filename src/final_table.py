"""Final results table: RL-generated candidates vs imatinib and the benchmark.

Anchors every generated molecule against imatinib -- the marketed ABL1 drug --
scored under an identical protocol, and against the distribution of 40 known
actives and 40 property-matched inactives docked in the same run. The
percentile columns are the load-bearing ones: with benchmark AUC ~0.78 the
protocol supports a distributional claim ("this molecule scores where actives
score") and not a per-molecule one ("this molecule binds"), so a raw kcal/mol
figure quoted on its own would overstate what was measured.

Scores are best-over-crystals (1IEP/3CS9 DFG-out, 2GQG DFG-in). The AlphaFold
column is reported separately rather than folded into the ensemble, because
that model is DFG-in and cannot represent type-II binding at all -- including
it in a best-of would quietly hide the conformational failure this project
measured.

Usage:  python src/final_table.py [--exh16] [--md]
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DOCKING, rel

CRYSTALS = ["1IEP", "3CS9", "2GQG"]
ALL_RECS = ["1IEP", "3CS9", "2GQG", "AF"]


def load(dirpath, prefix, recs):
    out = {}
    for r in recs:
        p = os.path.join(dirpath, f"{prefix}_{r}.json")
        if os.path.exists(p):
            out[r] = {L["name"]: L for L in json.load(open(p))["ligands"]}
    return out


def best(store, name, field, higher, recs):
    v = [store[r][name]["dock"][field] for r in recs
         if r in store and name in store[r] and store[r][name].get("dock")]
    if not v:
        return None
    return (max if higher else min)(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DOCKING)
    ap.add_argument("--prefix", default="all16")
    ap.add_argument("--cand-json", default=rel("results", "candidates_v5d.json"))
    ap.add_argument("--out", default=os.path.join(DOCKING, "final_table.json"))
    args = ap.parse_args()

    S = load(args.dir, args.prefix, ALL_RECS)
    if not S:
        print(f"no {args.prefix}_*.json in {args.dir}"); return
    ref = S[CRYSTALS[0]]
    names = list(ref)
    cls = {n: ref[n]["class"] for n in names}
    cand_meta = {f"cand{i}": d for i, d in
                 enumerate(json.load(open(args.cand_json))["candidates"], 1)}

    acts = [n for n in names if cls[n] == "benchmark_active"]
    inas = [n for n in names if cls[n] == "benchmark_inactive"]
    A = np.array([v for v in (best(S, n, "affinity", False, CRYSTALS) for n in acts) if v is not None])
    I = np.array([v for v in (best(S, n, "affinity", False, CRYSTALS) for n in inas) if v is not None])
    Ac = np.array([v for v in (best(S, n, "cnn_affinity", True, CRYSTALS) for n in acts) if v is not None])

    def pct_vs_actives(v, higher=False):
        return 100.0 * ((A > v).mean() if not higher else (Ac < v).mean())

    rows = []
    order = ["imatinib", "dasatinib", "nilotinib"] + [f"cand{i}" for i in range(1, 6)]
    for n in order:
        if n not in ref:
            continue
        aff = best(S, n, "affinity", False, CRYSTALS)
        cnn = best(S, n, "cnn_affinity", True, CRYSTALS)
        af = S["AF"][n]["dock"]["affinity"] if "AF" in S and S["AF"][n].get("dock") else None
        per = {r: (S[r][n]["dock"]["affinity"] if S[r][n].get("dock") else None) for r in ALL_RECS if r in S}
        rows.append({"name": n, "class": cls[n], "affinity_best_crystal": aff,
                     "cnn_affinity_best": cnn, "affinity_AF": af, "per_receptor": per,
                     "pct_vs_actives_affinity": pct_vs_actives(aff) if aff is not None else None,
                     "pct_vs_actives_cnnaff": pct_vs_actives(cnn, True) if cnn is not None else None,
                     "af_penalty": (af - aff) if (af is not None and aff is not None) else None,
                     "rf_p_active": cand_meta.get(n, {}).get("p_active"),
                     "mw": cand_meta.get(n, {}).get("mw"),
                     "max_tanimoto_train": cand_meta.get(n, {}).get("max_tanimoto_train")})

    ima = next((r for r in rows if r["name"] == "imatinib"), None)
    print(f"\nBenchmark reference (same protocol, best over {'/'.join(CRYSTALS)}):")
    print(f"  40 known actives    affinity mean {A.mean():6.2f}  median {np.median(A):6.2f}  "
          f"[{A.min():.2f}, {A.max():.2f}]")
    print(f"  40 matched inactive affinity mean {I.mean():6.2f}  median {np.median(I):6.2f}  "
          f"[{I.min():.2f}, {I.max():.2f}]")

    hdr = ("molecule", "source", "dock", "vs act", "CNNaff", "AF", "AFpen", "RF P", "MW")
    print("\n%-12s %-14s %7s %7s %8s %7s %7s %7s %7s" % hdr)
    print("-" * 90)
    for r in rows:
        src = {"reference_drug": "marketed drug", "generated_candidate": "RL v5d"}.get(r["class"], r["class"])
        f = lambda v, w, p=2: (f"%{w}.{p}f" % v) if v is not None else " " * (w - 3) + "n/a"
        print("%-12s %-14s %7s %6s%% %8s %7s %7s %7s %7s" % (
            r["name"], src, f(r["affinity_best_crystal"], 7),
            ("%5.0f" % r["pct_vs_actives_affinity"]) if r["pct_vs_actives_affinity"] is not None else "  n/a",
            f(r["cnn_affinity_best"], 8, 3), f(r["affinity_AF"], 7),
            ("%+7.2f" % r["af_penalty"]) if r["af_penalty"] is not None else "    n/a",
            f(r["rf_p_active"], 7, 3), f(r["mw"], 7, 1)))
    if ima:
        print(f"\nimatinib anchor: {ima['affinity_best_crystal']:.2f} kcal/mol "
              f"({ima['pct_vs_actives_affinity']:.0f}th percentile of known actives)")
        better = [r["name"] for r in rows if r["class"] == "generated_candidate"
                  and r["affinity_best_crystal"] is not None
                  and r["affinity_best_crystal"] < ima["affinity_best_crystal"]]
        print(f"candidates scoring better than imatinib: {len(better)}/5  {better}")

    json.dump({"benchmark_actives": {"mean": float(A.mean()), "median": float(np.median(A)),
                                     "min": float(A.min()), "max": float(A.max()), "n": len(A)},
               "benchmark_inactives": {"mean": float(I.mean()), "median": float(np.median(I)),
                                       "min": float(I.min()), "max": float(I.max()), "n": len(I)},
               "rows": rows}, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
