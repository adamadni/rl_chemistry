"""Does each docked pose make the interactions an ABL1 binder has to make?

A docking score is a number attached to a pose, and the number is only worth
as much as the pose. This checks the poses directly against ABL1's known
pharmacophore, which is the cheap way to tell a plausible hit from a scoring
artefact -- and the cheap way to find out why a real drug scored badly.

The residues checked, and why each one:
  Met318  HINGE. Essentially every ATP-competitive kinase inhibitor donates
          and/or accepts a hydrogen bond to the hinge backbone. A pose that
          misses the hinge entirely is almost certainly wrong, however good
          its score.
  Thr315  GATEKEEPER. Controls access to the hydrophobic back pocket; the
          T315I mutation is the classic imatinib-resistance mechanism, which
          is precisely because this contact matters.
  Glu286  alphaC-helix glutamate. With Asp381 it forms the salt-bridge network
          that TYPE II inhibitors exploit in the DFG-out pocket.
  Asp381  DFG aspartate. Type-II inhibitors hydrogen bond to its backbone NH.
  Lys271  Catalytic lysine.
Contacting Glu286/Asp381 in a DFG-out receptor is the type-II signature;
hinge-only binding in a DFG-in receptor is the type-I signature. Which
signature a generated molecule shows is chemically interesting in its own
right, independent of the score.

**Deliberate approximation, stated rather than hidden:** hydrogen bonds are
detected by heavy-atom distance (N/O to N/O within `--polar-cut`), with no
angular criterion, because poses come from gnina without reliable explicit
hydrogens. This over-counts: some contacts flagged polar are geometrically
incapable of being real hydrogen bonds. It is therefore a screen for
*absence* -- "this pose never approaches the hinge" is trustworthy, "this pose
hydrogen bonds to the hinge" is a candidate for closer inspection. That
asymmetry is exactly what is needed to catch scoring artefacts.

Usage:
    python src/interaction_fingerprint.py --receptor 1IEP --seed 42
"""
import argparse, glob, json, os, re, sys
import numpy as np
from Bio.PDB import PDBParser
from rdkit import Chem, RDLogger
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DOCKING
RDLogger.DisableLog("rdApp.*")

KEY = {271: "Lys271", 286: "Glu286", 315: "Thr315", 318: "Met318", 381: "Asp381"}
POLAR = {"N", "O"}


def receptor_atoms(path):
    ch = next(iter(PDBParser(QUIET=True).get_structure("r", path)[0]))
    out = []
    for res in ch:
        if res.id[0] != " ":
            continue
        for a in res:
            out.append((res.id[1], res.get_resname(), a.get_id(), a.element, a.coord))
    return out


def pose_atoms(sdf):
    """Heavy atoms of the top-scoring pose (first record)."""
    supp = Chem.SDMolSupplier(sdf, removeHs=True, sanitize=False)
    for m in supp:
        if m is None:
            continue
        conf = m.GetConformer()
        return [(a.GetSymbol(), np.array(conf.GetAtomPosition(a.GetIdx())))
                for a in m.GetAtoms() if a.GetSymbol() != "H"]
    return None


def fingerprint(lig, rec, polar_cut, apolar_cut):
    res_contacts, polar_res, key_hits = set(), set(), {}
    for rnum, rname, aid, elem, rc in rec:
        for lsym, lc in lig:
            d = float(np.linalg.norm(rc - lc))
            if d > apolar_cut:
                continue
            res_contacts.add(rnum)
            if elem in POLAR and lsym in POLAR and d <= polar_cut:
                polar_res.add(rnum)
                if rnum in KEY:
                    tag = "backbone" if aid in ("N", "O") else "sidechain"
                    key_hits.setdefault(rnum, set()).add(tag)
    return res_contacts, polar_res, key_hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DOCKING)
    ap.add_argument("--receptor", default="1IEP")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--polar-cut", type=float, default=3.5)
    ap.add_argument("--apolar-cut", type=float, default=4.5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rec = receptor_atoms(os.path.join(args.dir, "struct", f"{args.receptor}_rec.pdb"))
    present = {r[0] for r in rec}
    missing = [f"{KEY[k]}" for k in KEY if k not in present]
    if missing:
        print(f"WARNING: key residues absent from {args.receptor}: {missing}")

    names = ["nilotinib", "imatinib", "dasatinib"] + [f"cand{i}" for i in range(1, 6)]
    rows = []
    print(f"receptor {args.receptor}  seed {args.seed}  "
          f"(polar<= {args.polar_cut} A, contact<= {args.apolar_cut} A)\n")
    print("%-11s %7s %8s %8s %8s %8s %8s" % (
        "ligand", "resid", "Met318", "Thr315", "Glu286", "Asp381", "Lys271"))
    print("-" * 64)
    for n in names:
        pat = os.path.join(args.dir, "out", f"{n}_{args.receptor}_s{args.seed}.sdf")
        if not os.path.exists(pat):
            alt = glob.glob(os.path.join(args.dir, "out", f"{n}_{args.receptor}*.sdf"))
            if not alt:
                print(f"{n:<11}  <no pose file>")
                continue
            pat = sorted(alt)[0]
        lig = pose_atoms(pat)
        if not lig:
            print(f"{n:<11}  <unreadable pose>")
            continue
        contacts, polar, key_hits = fingerprint(lig, rec, args.polar_cut, args.apolar_cut)
        cell = lambda k: ("+".join(sorted(key_hits[k]))[:8] if k in key_hits
                          else ("contact" if k in contacts else "-"))
        print("%-11s %7d %8s %8s %8s %8s %8s" % (
            n, len(contacts), cell(318), cell(315), cell(286), cell(381), cell(271)))
        rows.append({"ligand": n, "receptor": args.receptor, "seed": args.seed,
                     "n_residues_contacted": len(contacts),
                     "polar_residues": sorted(polar),
                     "key": {KEY[k]: sorted(v) for k, v in key_hits.items()},
                     "key_contact_only": [KEY[k] for k in KEY
                                          if k in contacts and k not in key_hits]})
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
