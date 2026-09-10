"""Prepare protein and ligand inputs for a relative binding free energy campaign.

The single most important choice here is that ligand poses are built by
CONSTRAINED EMBEDDING onto the validated seed pose, not taken from independent
docking runs.

RBFE assumes every ligand in the series occupies the same binding mode and
differs only in the perturbed region; the alchemical transformation morphs R
groups in place and cannot fix a ligand that docked into a different
orientation. Independent docking gives each analogue its own best guess, and
gnina's own seed-to-seed spread on this series (up to 1.18 kcal/mol) shows
those guesses are not always stable. Taking one pose that was validated
against ABL1's pharmacophore -- cand4 reproduces the full imatinib/nilotinib
type-II signature, hinge Met318 backbone H-bond plus gatekeeper, Glu286 and
Asp381, on both DFG-out structures -- and forcing every analogue onto its
common core makes the shared-binding-mode assumption true by construction
rather than hoping docking respected it.

The receptor is 1IEP (imatinib-bound, DFG-out): the series is a type-II
chemotype, and docking a type-II ligand into DFG-in has already been shown on
this project to fail for conformational reasons no force field will repair.

Protein prep is PDBFixer -- missing heavy atoms and hydrogens added at pH 7.4,
waters and the co-crystallised ligand dropped. Note the crystal waters are
discarded: bridging waters can matter for kinase binding, and their omission
is a known systematic error in this kind of calculation, recorded here so it
is a stated assumption rather than an oversight.

Usage:
    python src/prep_rbfe.py
"""
import argparse, json, os, sys
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdFMCS
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DOCKING, RBFE, rel
RDLogger.DisableLog("rdApp.*")


def constrained_pose(smiles, ref_noh, timeout=60, seed=42):
    """Embed `smiles` with its shared core pinned to the reference's coordinates.

    Implemented as a coordinate map keyed off the MCS SMARTS matched
    independently into both molecules, rather than RDKit's ConstrainedEmbed on
    a core carved out of the reference. Carving a fragment out of a 3D pose
    leaves atoms whose valence and aromatic perception no longer agree with the
    same atoms in a molecule built fresh from SMILES, so the substructure match
    ConstrainedEmbed needs fails outright -- it did so for all 12 ligands here.
    Matching one SMARTS into each molecule separately sidesteps that entirely.

    After embedding, the core atoms are held fixed and the rest is
    MMFF-minimised, so the R-groups relax into sensible geometry without
    dragging the validated core off the reference pose.

    Returns (mol_with_H, n_core_atoms, core_rmsd_to_reference) or None.
    """
    lig = Chem.AddHs(Chem.MolFromSmiles(smiles))
    mcs = rdFMCS.FindMCS([ref_noh, Chem.RemoveHs(lig)], timeout=timeout,
                         completeRingsOnly=True, ringMatchesRingOnly=True,
                         atomCompare=rdFMCS.AtomCompare.CompareElements,
                         bondCompare=rdFMCS.BondCompare.CompareOrderExact)
    patt = Chem.MolFromSmarts(mcs.smartsString) if mcs.smartsString else None
    if patt is None:
        return None
    rm = ref_noh.GetSubstructMatch(patt)
    lm = lig.GetSubstructMatch(patt)
    if not rm or not lm or len(rm) != len(lm):
        return None
    conf = ref_noh.GetConformer()
    coord_map = {lm[i]: conf.GetAtomPosition(rm[i]) for i in range(len(rm))}
    if AllChem.EmbedMolecule(lig, coordMap=coord_map, randomSeed=seed,
                             useRandomCoords=True, maxAttempts=200) < 0:
        return None
    try:
        props = AllChem.MMFFGetMoleculeProperties(lig)
        ff = AllChem.MMFFGetMoleculeForceField(lig, props)
        for i in lm:
            ff.AddFixedPoint(i)
        ff.Minimize(maxIts=1000)
    except Exception:
        pass
    lc = lig.GetConformer()
    d = [lc.GetAtomPosition(lm[i]).Distance(conf.GetAtomPosition(rm[i])) for i in range(len(rm))]
    return lig, mcs.numAtoms, float(np.sqrt(np.mean(np.square(d))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rbfe-json", default=rel("rbfe_cand4.json"))
    ap.add_argument("--ref-pose", default=os.path.join(DOCKING, "out", "cand4_1IEP_s42.sdf"))
    ap.add_argument("--receptor", default=os.path.join(DOCKING, "struct", "1IEP_rec.pdb"))
    ap.add_argument("--out-dir", default=RBFE)
    ap.add_argument("--drop", nargs="*", default=["L08"],
                    help="ligand ids to exclude; L08 carries the largest seed spread "
                         "in the whole docking run (1.18 kcal/mol)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- reference pose (top mode of the validated seed docking)
    ref = next(m for m in Chem.SDMolSupplier(args.ref_pose, removeHs=False, sanitize=True)
               if m is not None)
    ref_noh = Chem.RemoveHs(ref)
    print(f"reference pose: {args.ref_pose}  ({ref_noh.GetNumAtoms()} heavy atoms)")

    data = json.load(open(args.rbfe_json))
    ligs = [L for L in data["ligands"] if L["id"] not in args.drop]
    print(f"{len(data['ligands'])} ligands, {len(ligs)} after dropping {args.drop}\n")

    written, failed = [], []
    w = Chem.SDWriter(os.path.join(args.out_dir, "ligands.sdf"))
    for L in ligs:
        res = constrained_pose(L["smiles"], ref_noh)
        if res is None:
            failed.append((L["id"], "MCS match / embed failed"))
            continue
        m, n_core, rms = res
        if n_core < 12:
            failed.append((L["id"], f"core too small ({n_core})"))
            continue
        m.SetProp("_Name", L["id"])
        m.SetProp("core_atoms", str(n_core))
        m.SetProp("p_active", str(L.get("p_active")))
        w.write(m)
        written.append((L["id"], n_core, rms, L.get("p_active")))
    w.close()

    print(f"{'id':<6} {'core':>5} {'coreRMSD':>9} {'RF P':>7}")
    for lid, n, rms, p in written:
        print(f"{lid:<6} {n:>5} {rms:>9.3f} {(('%7.3f' % p) if p else '      -')}")
    if failed:
        print("\nFAILED:")
        for lid, why in failed:
            print(f"  {lid}: {why}")
    print(f"\nwrote {args.out_dir}/ligands.sdf  ({len(written)} ligands)")

    # ---- receptor
    try:
        from pdbfixer import PDBFixer
        from openmm.app import PDBFile
        fixer = PDBFixer(filename=args.receptor)
        fixer.findMissingResidues()
        fixer.missingResidues = {}          # do not model unresolved loops de novo
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.removeHeterogens(keepWater=False)
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.4)
        out = os.path.join(args.out_dir, "protein.pdb")
        PDBFile.writeFile(fixer.topology, fixer.positions, open(out, "w"), keepIds=True)
        n_at = sum(1 for _ in fixer.topology.atoms())
        n_res = sum(1 for _ in fixer.topology.residues())
        print(f"wrote {out}  ({n_res} residues, {n_at} atoms, H added at pH 7.4, waters dropped)")
    except ImportError:
        print("pdbfixer not importable -- run inside the openfe env", file=sys.stderr)


if __name__ == "__main__":
    main()
