"""Prepare the ABL1 docking ensemble: two holo crystals + the AlphaFold model.

Ensemble rather than one receptor because ABL1 inhibitors split by the
conformation they require, and no single structure serves both:
  - Type I  (dasatinib) bind DFG-in, the active conformation      -> 2GQG
  - Type II (imatinib, nilotinib) bind DFG-out, which opens an
    allosteric back pocket that DOES NOT EXIST in DFG-in           -> 1IEP
Docking a type-II ligand into a DFG-in receptor fails for conformational
reasons, not chemical ones, and no scoring function repairs that. Each ligand
is therefore docked against every receptor and scored on its best result.

The AlphaFold arm (AF-P00519, model v6) is included to MEASURE what it costs
relative to experimental structures, not because ABL1 lacks coverage -- it is
among the best-characterised kinases in the PDB. Two things are done to it:

  1. **Kinase domain excised.** The full-length model is ~1130 residues and
     the AlphaFold API reports mean pLDDT 63.4 with 49% of residues at "very
     low" confidence: ABL1 is largely disordered outside its folded domains.
     Docking against the whole model would put a search box in a cloud of
     unreliable coordinates.
  2. **Superimposed onto the 1IEP frame** on matched CA atoms, so the crystal
     ligand centroid defines the same physical box for every receptor. Without
     this the AlphaFold arm would be searching a differently-placed box and
     any score difference would confound pocket location with model quality.

Box centre is the centroid of each structure's own co-crystallised inhibitor
(largest non-water HETATM residue, found rather than hard-coded).

Usage:  python src/prep_receptors.py --dir <docking dir>
"""
import argparse, json, os, sys
import numpy as np
from Bio.PDB import PDBParser, PDBIO, Select, Superimposer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DOCKING

# ABL1 (UniProt P00519, isoform 1a numbering) kinase domain. The crystals
# cover roughly this span; the AlphaFold model is trimmed to match so the
# superposition is domain-to-domain rather than domain-to-disorder.
KD_LO, KD_HI = 242, 495


class ProteinOnly(Select):
    """Drop waters, ions and the co-crystallised ligand.

    The ligand is removed AFTER its centroid has been taken -- docking into a
    pocket that still contains its original occupant would be meaningless.
    """
    def accept_residue(self, res):
        return res.id[0] == " "
    def accept_atom(self, atom):
        return atom.element != "H" and not atom.is_disordered() or atom.get_altloc() in (" ", "A")


def biggest_het(chain):
    best = None
    for res in chain:
        if res.id[0] == " " or res.get_resname() == "HOH":
            continue
        n = len([a for a in res])
        if best is None or n > best[1]:
            best = (res, n)
    return best[0] if best else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DOCKING)
    ap.add_argument("--pad", type=float, default=12.0, help="box half-width around ligand centroid")
    args = ap.parse_args()
    S, OUT = os.path.join(args.dir, "struct"), os.path.join(args.dir, "struct")
    p = PDBParser(QUIET=True)
    io = PDBIO()
    meta = {}

    ref_ca = None   # 1IEP CA atoms, the common frame
    for pdb in ("1IEP", "2GQG", "3CS9"):
        st = p.get_structure(pdb, os.path.join(S, f"{pdb}.pdb"))
        model = st[0]
        chain = next(c for c in model if any(r.id[0] == " " for r in c))
        lig = biggest_het(chain)
        if lig is None:
            print(f"{pdb}: no ligand found, skipping"); continue
        coords = np.array([a.coord for a in lig])
        centre = coords.mean(axis=0)
        # keep only the protein chain the ligand sits in
        for c in list(model):
            if c.id != chain.id:
                model.detach_child(c.id)
        io.set_structure(st)
        io.save(os.path.join(OUT, f"{pdb}_rec.pdb"), ProteinOnly())
        size = (coords.max(axis=0) - coords.min(axis=0)) + args.pad
        meta[pdb] = {"ligand": lig.get_resname(), "chain": chain.id,
                     "center": [round(float(x), 3) for x in centre],
                     "size": [round(float(max(x, 18.0)), 1) for x in size]}
        print(f"{pdb}: ligand {lig.get_resname()} chain {chain.id} "
              f"centre {meta[pdb]['center']} box {meta[pdb]['size']}")
        if pdb == "1IEP":
            ref_ca = {r.id[1]: r["CA"] for r in chain
                      if r.id[0] == " " and "CA" in r and KD_LO <= r.id[1] <= KD_HI}

    # ---- AlphaFold: trim to the kinase domain, then superimpose onto 1IEP
    st = p.get_structure("AF", os.path.join(S, "AF-P00519.pdb"))
    model = st[0]
    chain = next(iter(model))
    plddt_all = [a.bfactor for r in chain for a in r]
    for r in list(chain):
        if not (KD_LO <= r.id[1] <= KD_HI):
            chain.detach_child(r.id)
    af_ca = {r.id[1]: r["CA"] for r in chain if "CA" in r}
    common = sorted(set(af_ca) & set(ref_ca))
    sup = Superimposer()
    sup.set_atoms([ref_ca[i] for i in common], [af_ca[i] for i in common])
    sup.apply([a for r in chain for a in r])
    io.set_structure(st)
    io.save(os.path.join(OUT, "AF_rec.pdb"), ProteinOnly())
    plddt_kd = [a.bfactor for r in chain for a in r]
    meta["AF"] = {"ligand": None, "chain": chain.id,
                  "center": meta["1IEP"]["center"], "size": meta["1IEP"]["size"],
                  "aligned_on": "1IEP", "n_ca_aligned": len(common),
                  "ca_rmsd": round(float(sup.rms), 3),
                  "mean_plddt_full": round(float(np.mean(plddt_all)), 1),
                  "mean_plddt_kinase_domain": round(float(np.mean(plddt_kd)), 1)}
    print(f"AF : kinase domain {KD_LO}-{KD_HI}, {len(common)} CA aligned to 1IEP, "
          f"CA-RMSD {sup.rms:.2f} A")
    print(f"     mean pLDDT full-length {np.mean(plddt_all):.1f} -> kinase domain "
          f"{np.mean(plddt_kd):.1f}")
    json.dump(meta, open(os.path.join(args.dir, "receptors.json"), "w"), indent=1)
    print(f"\nwrote {args.dir}/receptors.json")


if __name__ == "__main__":
    main()
