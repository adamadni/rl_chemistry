"""Dock a ligand set against the ABL1 receptor ensemble with smina.

Ligands are 3D-embedded with ETKDG and MMFF-minimised, then docked against
every receptor in `receptors.json`; each ligand's reported score is its best
(most negative) affinity across the ensemble, because the ensemble exists
precisely so that a ligand can be scored in the conformation it actually
needs. Per-receptor scores are kept too -- for ABL1 the *pattern* across
receptors is more informative than the best value, since a type-II ligand
scoring well on DFG-out and badly on DFG-in is the expected signature rather
than an inconsistency.

Sets:
  control    -- 3 known ABL1 drugs + the 5 measured-inactive compounds.
                RUN THIS FIRST. If the protocol cannot separate known actives
                from known inactives on this target, candidate scores mean
                nothing and the setup needs fixing before anything is read
                into them.
  candidates -- the 5 generated molecules.

Interpretation caveat, deliberately recorded next to the code that produces
the numbers: docking affinity correlates only weakly with measured binding
(r ~ 0.3-0.5 across benchmarks). These scores support *discrimination and
enrichment* claims, not predicted potencies, and no result here should be
quoted as an affinity.

Usage:
    python src/run_docking.py --set control --exhaustiveness 16
"""
import argparse, json, os, re, subprocess, sys
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DOCKING, rel
RDLogger.DisableLog("rdApp.*")

KNOWN_ACTIVES = {
    "imatinib":  "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
    "dasatinib": "Cc1nc(Nc2ncc(C(=O)Nc3c(C)cccc3Cl)s2)cc(N2CCN(CCO)CC2)n1",
    "nilotinib": "Cc1cn(-c2cc(NC(=O)c3ccc(C)c(Nc4nccc(-c5cccnc5)n4)c3)cc(C(F)(F)F)c2)cn1",
}


def embed(smiles, seed=42):
    """SMILES -> single MMFF-minimised 3D conformer, hydrogens explicit."""
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    m = Chem.AddHs(m)
    ps = AllChem.ETKDGv3()
    ps.randomSeed = seed
    if AllChem.EmbedMolecule(m, ps) != 0:
        ps.useRandomCoords = True
        if AllChem.EmbedMolecule(m, ps) != 0:
            return None
    try:
        AllChem.MMFFOptimizeMolecule(m, maxIters=500)
    except Exception:
        pass
    return m


def dock(smina, rec, lig_sdf, box, out_sdf, exhaustiveness, cpu, seed=42):
    cmd = [smina, "-r", rec, "-l", lig_sdf, "-o", out_sdf,
           "--center_x", str(box["center"][0]), "--center_y", str(box["center"][1]),
           "--center_z", str(box["center"][2]),
           "--size_x", str(box["size"][0]), "--size_y", str(box["size"][1]),
           "--size_z", str(box["size"][2]),
           "--exhaustiveness", str(exhaustiveness), "--num_modes", "9",
           "--seed", str(seed), "--cpu", str(cpu)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    best = None
    for line in r.stdout.splitlines():
        m = re.match(r"^\s*1\s+(-?\d+\.\d+)\s", line)
        if m:
            best = float(m.group(1))
            break
    if best is None:
        sys.stderr.write(r.stdout[-600:] + r.stderr[-600:])
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DOCKING)
    ap.add_argument("--candidates", default=rel("results", "candidates_v5d.json"))
    ap.add_argument("--set", choices=["control", "candidates", "all"], default="control")
    ap.add_argument("--exhaustiveness", type=int, default=16)
    ap.add_argument("--cpu", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    recs = json.load(open(os.path.join(args.dir, "receptors.json")))
    cand = json.load(open(args.candidates))
    smina = os.path.join(args.dir, "smina")

    ligands = []
    if args.set in ("control", "all"):
        for n, s in KNOWN_ACTIVES.items():
            ligands.append({"name": n, "smiles": s, "class": "known_active"})
        for d in cand["negatives"]:
            ligands.append({"name": d["chembl_id"], "smiles": d["smiles"],
                            "class": "measured_negative", "pchembl": d["pchembl"]})
    if args.set in ("candidates", "all"):
        for i, d in enumerate(cand["candidates"], 1):
            ligands.append({"name": f"cand{i}", "smiles": d["smiles"],
                            "class": "generated_candidate", "p_active": d["p_active"]})

    ldir = os.path.join(args.dir, "lig")
    os.makedirs(ldir, exist_ok=True)
    rec_names = [r for r in ("1IEP", "3CS9", "2GQG", "AF") if r in recs]
    print(f"{len(ligands)} ligands x {len(rec_names)} receptors "
          f"({', '.join(rec_names)}), exhaustiveness {args.exhaustiveness}\n", flush=True)

    for L in ligands:
        m = embed(L["smiles"])
        if m is None:
            print(f"  !! embed failed: {L['name']}"); L["scores"] = {}; continue
        sdf = os.path.join(ldir, f"{L['name']}.sdf")
        w = Chem.SDWriter(sdf); w.write(m); w.close()
        L["scores"] = {}
        for rn in rec_names:
            s = dock(smina, os.path.join(args.dir, "struct", f"{rn}_rec.pdb"), sdf,
                     recs[rn], os.path.join(args.dir, "out", f"{L['name']}_{rn}.sdf"),
                     args.exhaustiveness, args.cpu)
            L["scores"][rn] = s
        vals = [v for v in L["scores"].values() if v is not None]
        L["best"] = min(vals) if vals else None
        L["best_receptor"] = min(L["scores"], key=lambda k: (L["scores"][k] is None, L["scores"][k])) if vals else None
        print(f"  {L['name']:<16} {L['class']:<19} "
              + "  ".join(f"{rn} {L['scores'][rn] if L['scores'][rn] is not None else float('nan'):6.2f}"
                          for rn in rec_names)
              + f"   | best {L['best']:6.2f} ({L['best_receptor']})", flush=True)

    out = args.out or os.path.join(args.dir, f"results_{args.set}.json")
    json.dump({"receptors": recs, "exhaustiveness": args.exhaustiveness,
               "ligands": ligands}, open(out, "w"), indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
