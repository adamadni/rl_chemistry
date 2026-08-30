"""Dock the matched benchmark with gnina: Vina score + CNN pose/affinity.

Three changes from the first control run, each targeting a specific weakness
that run exposed:

1. **Sample size.** 40 actives vs 40 matched inactives is 1600 Mann-Whitney
   pairs, against the first control's 15. The AUC from 15 pairs had a 95%
   interval of roughly 0.40-0.95 and could not decide anything.

2. **Protonation at pH 7.4** (Dimorphite-DL). The first run docked neutral
   species. Several ABL1 ligands -- dasatinib's hydroxyethyl-piperazine, the
   piperazines in our own candidates -- carry a basic amine that is charged at
   physiological pH, which changes both the H-bond pattern and the desolvation
   term. Only the dominant microstate is taken; full microstate enumeration
   would multiply the run and is not warranted for an enrichment estimate.

3. **CNN rescoring** (gnina). Vina-class empirical scoring gave AUC 0.733 with
   the best inactive outranking two of three known drugs. gnina reports three
   numbers per pose and all are recorded, because they measure different
   things and it is worth seeing which -- if any -- actually separates the set:
       affinity     Vina-like, kcal/mol, LOWER is better
       CNNscore     pose plausibility in [0,1], HIGHER is better
       CNNaffinity  predicted pK, HIGHER is better

One receptor per process so the two conformations run concurrently on one GPU.

Usage:
    python src/run_benchmark_docking.py --receptor 1IEP
"""
import argparse, json, os, re, subprocess, sys
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")


def protonate(smiles, ph=7.4):
    """Dominant microstate at `ph`; falls back to the input on failure."""
    try:
        from dimorphite_dl import protonate_smiles
        out = protonate_smiles(smiles, ph_min=ph, ph_max=ph, precision=1.0)
        if out:
            return out[0]
    except Exception as e:
        sys.stderr.write(f"protonate failed for {smiles}: {e}\n")
    return smiles


def embed(smiles, seed=42):
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


ROW = re.compile(r"^\s*1\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)")


def dock(gnina, rec, lig, box, out, exh, seed):
    cmd = [gnina, "-r", rec, "-l", lig, "-o", out,
           "--center_x", str(box["center"][0]), "--center_y", str(box["center"][1]),
           "--center_z", str(box["center"][2]),
           "--size_x", str(box["size"][0]), "--size_y", str(box["size"][1]),
           "--size_z", str(box["size"][2]),
           "--exhaustiveness", str(exh), "--num_modes", "9",
           "--cnn_scoring", "rescore", "--seed", str(seed)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=2400)
    except subprocess.TimeoutExpired:
        return None
    for line in r.stdout.splitlines():
        m = ROW.match(line)
        if m:
            aff, intra, cnns, cnna = (float(x) for x in m.groups())
            return {"affinity": aff, "cnn_score": cnns, "cnn_affinity": cnna}
    sys.stderr.write(r.stdout[-800:] + r.stderr[-400:])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/workspace/rl_chemistry/docking")
    ap.add_argument("--receptor", required=True)
    ap.add_argument("--ligands", default=None, help="defaults to docking/benchmark.json")
    ap.add_argument("--exhaustiveness", type=int, default=8)
    ap.add_argument("--ph", type=float, default=7.4)
    ap.add_argument("--seed", type=int, default=42,
                    help="Vina search seed. The search is stochastic and its spread is NOT "
                         "negligible: at exhaustiveness 8 imatinib scored -4.42 against its "
                         "own crystal structure and -12.79 at 16. Run several seeds and take "
                         "the median rather than trusting one draw.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    recs = json.load(open(os.path.join(args.dir, "receptors.json")))
    box = recs[args.receptor]
    src = args.ligands or os.path.join(args.dir, "benchmark.json")
    ligs = json.load(open(src))["ligands"]
    gnina = os.path.join(args.dir, "gnina")
    ldir = os.path.join(args.dir, f"lig_{args.receptor}")
    os.makedirs(ldir, exist_ok=True)
    os.makedirs(os.path.join(args.dir, "out"), exist_ok=True)

    print(f"{args.receptor}: {len(ligs)} ligands, exhaustiveness {args.exhaustiveness}, pH {args.ph}",
          flush=True)
    for i, L in enumerate(ligs, 1):
        smi = protonate(L["smiles"], args.ph)
        L["smiles_protonated"] = smi
        m = embed(smi)
        if m is None:
            L["dock"] = None
            print(f"  [{i}/{len(ligs)}] {L['name']}: embed FAILED", flush=True)
            continue
        sdf = os.path.join(ldir, f"{L['name']}.sdf")
        w = Chem.SDWriter(sdf); w.write(m); w.close()
        L["dock"] = dock(gnina, os.path.join(args.dir, "struct", f"{args.receptor}_rec.pdb"),
                         sdf, box,
                         os.path.join(args.dir, "out", f"{L['name']}_{args.receptor}_s{args.seed}.sdf"),
                         args.exhaustiveness, args.seed)
        d = L["dock"]
        print(f"  [{i}/{len(ligs)}] {L['name']:<16} {L['class']:<20} "
              + (f"aff {d['affinity']:7.2f}  CNNscore {d['cnn_score']:.4f}  "
                 f"CNNaff {d['cnn_affinity']:6.3f}" if d else "FAILED"), flush=True)

    out = args.out or os.path.join(args.dir, f"bench_{args.receptor}_s{args.seed}.json")
    json.dump({"receptor": args.receptor, "box": box, "ph": args.ph,
               "exhaustiveness": args.exhaustiveness, "seed": args.seed, "ligands": ligs},
              open(out, "w"), indent=1)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
