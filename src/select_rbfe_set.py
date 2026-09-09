"""Curate an RBFE-ready subset and build its perturbation map.

`build_series.py` produces every core-matching molecule it can find; that is the
wrong thing to hand to a free-energy campaign. RBFE cost scales with the number
of edges actually simulated, and each edge's error grows with the size of the
perturbation, so the deliverable is a SMALL set wired into a cheap graph rather
than a large pile of analogues.

Selection criteria, in the order they bind:
  * drug-like size -- MW <= --max-mw. The raw cand4 series runs to MW 648 and
    563/1170 members exceed 500; unconstrained R-group growth reliably inflates
    both predicted affinity and molecular weight, and a series that drifts out
    of drug-like space is not a lead-optimisation series.
  * small perturbations -- heavy-atom delta from the seed within
    --max-heavy-delta, since that delta is what an alchemical edge has to morph.
  * ACTIVITY SPREAD. A series where every member is equipotent teaches a
    free-energy calculation nothing: the whole point is to test whether the
    method reproduces a known rank order. Members are therefore taken across
    predicted-activity strata rather than by taking the top N, which would
    return a flat, uninformative set.
  * mutual dissimilarity within a stratum, so the set does not collapse onto
    one substitution pattern.

The perturbation map is a STAR graph centred on the seed: every member is
morphed to/from the pose-validated reference. A star is the right topology when
one member is the anchor -- it needs n-1 edges rather than a cycle basis, and
every edge inherits the seed's validated binding mode, so a failed edge is
diagnostic rather than ambiguous. Cycle closures can be added later to estimate
statistical error, and the script reports which extra edges would be cheapest.

Usage:
    python src/select_rbfe_set.py --series series_cand4.json --k 16
"""
import argparse, json, itertools, sys
import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFMCS, rdFingerprintGenerator
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
RDLogger.DisableLog("rdApp.*")

_p = FilterCatalogParams()
_p.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
_p.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
CATALOG = FilterCatalog(_p)


def protonated_charge(smi, ph=7.4):
    """Formal charge of the dominant microstate at `ph`, not of the drawn form.

    This distinction is the whole point. A carboxylic acid is drawn neutral and
    passes any naive formal-charge check, but at pH 7.4 it is the carboxylate,
    so morphing it from a neutral seed is a CHARGE-CHANGING perturbation --
    which needs co-alchemical ions and finite-size corrections, and is exactly
    what a first RBFE campaign should not contain. Two members of the initial
    selection (an aryl acid and an omega-acid chain) were caught only by this.
    """
    try:
        from dimorphite_dl import protonate_smiles
        # precision=0.0 returns the SINGLE dominant microstate. With the
        # default 1.0 Dimorphite enumerates every state within +/-1 pH unit --
        # 4 to 8 per molecule here, with charges spanning -1/0/+1 -- and
        # out[0] is then an arbitrary member of that list, not the dominant
        # form. That bug silently passed 6 basic amines through this filter.
        out = protonate_smiles(smi, ph_min=ph, ph_max=ph, precision=0.0)
        if out:
            m = Chem.MolFromSmiles(out[0])
            if m is not None:
                return Chem.GetFormalCharge(m)
    except Exception as e:
        sys.stderr.write(f"protonation check failed for {smi}: {e}\n")
    m = Chem.MolFromSmiles(smi)
    return Chem.GetFormalCharge(m) if m else 0


def bond_stereo(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return set()
    return {str(b.GetStereo()) for b in m.GetBonds()
            if str(b.GetStereo()) not in ("STEREONONE",)}


def liabilities(smi, seed_charge, seed_stereo):
    """Reasons this molecule should not enter a free-energy campaign."""
    bad = []
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return ["unparseable"]
    ent = CATALOG.GetFirstMatch(m)
    if ent is not None:
        bad.append(f"alert:{ent.GetDescription()[:28]}")
    if protonated_charge(smi) != seed_charge:
        bad.append("charge-change@pH7.4")
    if bond_stereo(smi) != seed_stereo:
        bad.append("stereo-change")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default="/workspace/rl_chemistry/series_cand4.json")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--max-mw", type=float, default=500.0)
    ap.add_argument("--max-heavy-delta", type=int, default=8)
    ap.add_argument("--strata", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = json.load(open(args.series))
    seed_smi = Chem.CanonSmiles(d["seed_smiles"])
    seed_charge = protonated_charge(d["seed_smiles"])
    seed_stereo = bond_stereo(d["seed_smiles"])
    size_ok = [r for r in d["members"]
               if r["mw"] <= args.max_mw and r["heavy_delta_from_seed"] <= args.max_heavy_delta]
    pool, rejected = [], {}
    for r in size_ok:
        bad = liabilities(r["smiles"], seed_charge, seed_stereo)
        if bad:
            rejected[bad[0].split(":")[0] + (":" + bad[0].split(":")[1] if ":" in bad[0] else "")] = \
                rejected.get(bad[0], 0) + 1
            continue
        pool.append(r)
    print(f"{d['seed']}: {len(d['members'])} members -> {len(size_ok)} pass "
          f"MW<={args.max_mw:.0f} and dHeavy<={args.max_heavy_delta}")
    print(f"  -> {len(pool)} survive liability screening "
          f"(PAINS/BRENK alerts, charge change at pH 7.4, stereo change)")
    if rejected:
        top = sorted(rejected.items(), key=lambda kv: -kv[1])[:6]
        print("  most common rejections: " + ", ".join(f"{k} x{v}" for k, v in top))

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp = {r["smiles"]: gen.GetFingerprint(Chem.MolFromSmiles(r["smiles"])) for r in pool}

    seed_rec = next((r for r in pool if Chem.CanonSmiles(r["smiles"]) == seed_smi), None)
    if seed_rec is None:
        seed_rec = {"smiles": d["seed_smiles"], "p_active": None, "origin": "seed",
                    "r_group": None, "mw": None, "heavy_delta_from_seed": 0}
        fp[seed_rec["smiles"]] = gen.GetFingerprint(Chem.MolFromSmiles(seed_rec["smiles"]))

    # ---- stratified pick across predicted activity
    pa = np.array([r["p_active"] for r in pool])
    edges = np.quantile(pa, np.linspace(0, 1, args.strata + 1))
    chosen, per = [seed_rec], max(1, (args.k - 1) // args.strata)
    for i in range(args.strata):
        lo, hi = edges[i], edges[i + 1]
        band = [r for r in pool if lo <= r["p_active"] <= hi
                and Chem.CanonSmiles(r["smiles"]) != seed_smi]
        band.sort(key=lambda r: -r["p_active"])
        taken = 0
        for r in band:
            if any(DataStructs.TanimotoSimilarity(fp[r["smiles"]], fp[c["smiles"]]) > 0.92
                   for c in chosen):
                continue                      # near-duplicate of something already in
            chosen.append(r)
            taken += 1
            if taken >= per or len(chosen) >= args.k:
                break
        if len(chosen) >= args.k:
            break

    mols = [Chem.MolFromSmiles(r["smiles"]) for r in chosen]
    mcs = rdFMCS.FindMCS(mols, timeout=300, completeRingsOnly=True, ringMatchesRingOnly=True)
    avg = np.mean([m.GetNumHeavyAtoms() for m in mols])
    pa_c = np.array([r["p_active"] for r in chosen if r["p_active"] is not None])
    print(f"\nselected {len(chosen)} (incl. seed)")
    print(f"  MCS {mcs.numAtoms} atoms / {mcs.numBonds} bonds = {100*mcs.numAtoms/avg:.0f}% of avg member")
    print(f"  predicted activity spans P {pa_c.min():.3f}-{pa_c.max():.3f} "
          f"(delta {pa_c.max()-pa_c.min():.3f})")
    print(f"  MW {min(r['mw'] for r in chosen if r['mw']):.0f}-"
          f"{max(r['mw'] for r in chosen if r['mw']):.0f}")

    print(f"\n{'#':>3} {'origin':<11} {'P(act)':>7} {'dHeavy':>7} {'MW':>7} {'simSeed':>8}  smiles")
    rows = []
    for i, r in enumerate(chosen):
        sim = DataStructs.TanimotoSimilarity(fp[r["smiles"]], fp[seed_rec["smiles"]])
        tag = "SEED" if i == 0 else f"L{i:02d}"
        print(f"{tag:>3} {r['origin']:<11} "
              f"{(('%7.3f' % r['p_active']) if r['p_active'] is not None else '      -')} "
              f"{r['heavy_delta_from_seed']:>7d} {(('%7.1f' % r['mw']) if r['mw'] else '      -')} "
              f"{sim:>8.3f}  {r['smiles']}")
        rows.append({"id": tag, **r, "tanimoto_to_seed": float(sim)})

    # ---- star perturbation map, plus the cheapest cycle closures
    star = [{"from": "SEED", "to": r["id"],
             "heavy_delta": abs(r["heavy_delta_from_seed"]),
             "tanimoto": r["tanimoto_to_seed"]} for r in rows[1:]]
    pairs = []
    for a, b in itertools.combinations(rows[1:], 2):
        s = DataStructs.TanimotoSimilarity(fp[a["smiles"]], fp[b["smiles"]])
        pairs.append({"from": a["id"], "to": b["id"], "tanimoto": float(s)})
    pairs.sort(key=lambda e: -e["tanimoto"])
    print(f"\nperturbation map: star on SEED, {len(star)} edges "
          f"(mean Tanimoto to seed {np.mean([e['tanimoto'] for e in star]):.3f})")
    print("cheapest 5 cycle-closure edges to add for error estimation:")
    for e in pairs[:5]:
        print(f"  {e['from']} <-> {e['to']}   Tanimoto {e['tanimoto']:.3f}")

    out = args.out or f"/workspace/rl_chemistry/rbfe_{d['seed']}.json"
    json.dump({"seed": d["seed"], "seed_smiles": d["seed_smiles"], "core": d["core"],
               "mcs_smarts": mcs.smartsString, "mcs_atoms": mcs.numAtoms,
               "ligands": rows, "star_edges": star, "suggested_closures": pairs[:5]},
              open(out, "w"), indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
