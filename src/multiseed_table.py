"""Multi-seed docking: per-ligand error bars, and whether score gaps survive them.

Vina's search is stochastic. A single seed gives a point estimate with no
indication of its own reliability, which on this project already produced one
badly misleading number (imatinib at -4.42 against its own crystal structure).
This reports, per ligand, the median and spread across seeds, and then asks the
only question that matters for the final table: is the candidate-vs-imatinib
gap larger than the noise on either measurement?

Per ligand the score is median-over-seeds of the best-over-crystals value, so
each entry is a robust summary rather than a lucky draw. The spread reported
is over the ensemble-best across seeds -- that is the quantity the table
actually uses, so it is the quantity whose uncertainty matters.
"""
import argparse, glob, json, os, re
import numpy as np

CRYSTALS = ["1IEP", "3CS9", "2GQG"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/workspace/rl_chemistry/docking")
    ap.add_argument("--cand-json", default="/workspace/rl_chemistry/results_candidates_v5d.json")
    ap.add_argument("--out", default="/workspace/rl_chemistry/docking/multiseed_table.json")
    args = ap.parse_args()

    runs = {}
    for p in sorted(glob.glob(os.path.join(args.dir, "ms_*_s*.json"))):
        m = re.match(r"ms_(\w+)_s(\d+)\.json", os.path.basename(p))
        if not m:
            continue
        rec, seed = m.group(1), int(m.group(2))
        runs.setdefault(seed, {})[rec] = {L["name"]: L for L in json.load(open(p))["ligands"]}
    seeds = sorted(runs)
    if not seeds:
        print("no ms_*.json found"); return
    print(f"seeds: {seeds}   receptors: {sorted(runs[seeds[0]])}")

    ref = runs[seeds[0]][CRYSTALS[0]]
    names = list(ref)
    cls = {n: ref[n]["class"] for n in names}

    def ens(seed, n):
        v = [runs[seed][r][n]["dock"]["affinity"] for r in CRYSTALS
             if r in runs[seed] and runs[seed][r][n].get("dock")]
        return min(v) if v else None

    stats = {}
    for n in names:
        v = [ens(s, n) for s in seeds]
        v = [x for x in v if x is not None]
        if v:
            stats[n] = {"median": float(np.median(v)), "min": float(np.min(v)),
                        "max": float(np.max(v)), "spread": float(np.max(v) - np.min(v)),
                        "std": float(np.std(v)), "n": len(v), "values": v}

    A = np.array([stats[n]["median"] for n in names if cls[n] == "benchmark_active" and n in stats])
    I = np.array([stats[n]["median"] for n in names if cls[n] == "benchmark_inactive" and n in stats])
    sp = np.array([stats[n]["spread"] for n in stats])
    print(f"\nseed-to-seed spread across all {len(sp)} ligands: "
          f"median {np.median(sp):.2f}  mean {sp.mean():.2f}  90th pct {np.percentile(sp,90):.2f}  "
          f"max {sp.max():.2f} kcal/mol")
    print(f"benchmark medians: {len(A)} actives mean {A.mean():.2f} | "
          f"{len(I)} inactives mean {I.mean():.2f}")

    cand_meta = {f"cand{i}": d for i, d in
                 enumerate(json.load(open(args.cand_json))["candidates"], 1)}
    order = ["nilotinib", "imatinib", "dasatinib"] + [f"cand{i}" for i in range(1, 6)]
    order = [n for n in order if n in stats]
    order.sort(key=lambda n: stats[n]["median"])

    print("\n%-11s %-14s %8s %7s %14s %8s %7s" % (
        "molecule", "source", "median", "spread", "range", "%ile", "RF P"))
    print("-" * 82)
    for n in order:
        s = stats[n]
        pct = 100.0 * (A > s["median"]).mean()
        src = {"reference_drug": "marketed drug", "generated_candidate": "RL v5d"}.get(cls[n], cls[n])
        rf = cand_meta.get(n, {}).get("p_active")
        print("%-11s %-14s %8.2f %7.2f  [%6.2f,%6.2f] %7.0f%% %7s" % (
            n, src, s["median"], s["spread"], s["min"], s["max"], pct,
            ("%.3f" % rf) if rf is not None else "-"))

    ima = stats.get("imatinib")
    if ima:
        print(f"\nimatinib median {ima['median']:.2f} (spread {ima['spread']:.2f})")
        print("candidate vs imatinib -- does the gap clear the combined seed noise?")
        for i in range(1, 6):
            n = f"cand{i}"
            if n not in stats:
                continue
            s = stats[n]
            gap = s["median"] - ima["median"]
            noise = s["spread"] + ima["spread"]
            verdict = "SEPARATED" if abs(gap) > noise else "within noise"
            print(f"  {n}: gap {gap:+.2f}  vs combined spread {noise:.2f}  -> {verdict}")

    json.dump({"seeds": seeds, "stats": stats,
               "benchmark_active_mean": float(A.mean()), "benchmark_inactive_mean": float(I.mean())},
              open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
