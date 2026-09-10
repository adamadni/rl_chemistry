"""Run relative binding free energy edges with OpenFE / OpenMM.

Star network centred on the pose-validated seed (cand4): every ligand is
morphed to/from the one molecule whose docked pose reproduces the full
imatinib/nilotinib type-II signature. A star needs n-1 edges rather than a
cycle basis, and every edge inherits the seed's validated binding mode, so a
failed edge points at one ligand rather than at an ambiguous pair.

## Cost is the design constraint, so the knobs are explicit

OpenFE's defaults are 3 repeats x 11 lambda windows x 5 ns production, per leg.
For 11 edges x 2 legs that is ~3600 ns of sampling -- on the order of 20-35
GPU-days on one 4090. The defaults are right for a publication-grade campaign
on hardware that is not one rented consumer card, and wrong here.

What this script changes, and what each costs in rigour:
  --repeats 1        Removes the repeat-to-repeat error estimate. Statistical
                     error then has to come from cycle closures instead, which
                     is why `rbfe_cand4.json` carries suggested closure edges.
  --production 2.5ns Below OpenFE's 5 ns default. Convergence should be checked
                     per edge (forward/reverse estimates, overlap matrix)
                     rather than assumed.
Both are recorded in the output JSON so no downstream reader mistakes this for
a default-settings run.

## Two things deliberately left at defaults
Charges are am1bcc via ambertools -- slow (~1 min/ligand) but the standard
choice, and cheap relative to the sampling. Ligand FF is openff-2.2.1 with
ff14SB protein and TIP3P water, HMR at 4 fs.

Usage:
    python src/run_rbfe.py --pilot                 # one edge, short, to time it
    python src/run_rbfe.py --edges L10 L12 --production 2.5
"""
import argparse, json, os, pathlib, sys, time
from rdkit import Chem
from openff.units import unit
import openfe
from openfe import (SmallMoleculeComponent, ProteinComponent, SolventComponent,
                    ChemicalSystem, Transformation)
from openfe.protocols.openmm_rfe import RelativeHybridTopologyProtocol
from openfe.setup import LomapAtomMapper
from gufe.protocols import execute_DAG
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import RBFE


def build_settings(args):
    s = RelativeHybridTopologyProtocol.default_settings()
    s.protocol_repeats = args.repeats
    s.lambda_settings.lambda_windows = args.windows
    s.simulation_settings.equilibration_length = args.equilibration * unit.nanosecond
    s.simulation_settings.production_length = args.production * unit.nanosecond
    try:
        s.engine_settings.compute_platform = "CUDA"
    except Exception:
        pass
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=RBFE)
    ap.add_argument("--seed-id", default="SEED")
    ap.add_argument("--edges", nargs="*", default=None,
                    help="ligand ids to connect to the seed; default = all")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--windows", type=int, default=11)
    ap.add_argument("--equilibration", type=float, default=0.5)
    ap.add_argument("--production", type=float, default=2.5)
    ap.add_argument("--pilot", action="store_true",
                    help="one edge, 0.2/0.5 ns, to measure real throughput before committing")
    ap.add_argument("--legs", nargs="+", default=["solvent", "complex"])
    args = ap.parse_args()
    if args.pilot:
        args.equilibration, args.production = 0.2, 0.5
        args.repeats = 1

    mols = {m.GetProp("_Name"): m for m in
            Chem.SDMolSupplier(os.path.join(args.dir, "ligands.sdf"), removeHs=False)
            if m is not None}
    print(f"{len(mols)} ligands: {sorted(mols)}")
    seed = SmallMoleculeComponent.from_rdkit(mols[args.seed_id])
    protein = ProteinComponent.from_pdb_file(os.path.join(args.dir, "protein.pdb"))
    solvent = SolventComponent()

    targets = args.edges or [k for k in mols if k != args.seed_id]
    if args.pilot:
        targets = targets[:1]
    settings = build_settings(args)
    protocol = RelativeHybridTopologyProtocol(settings)
    mapper = LomapAtomMapper(threed=True, element_change=False)

    print(f"\nsettings: repeats={args.repeats} windows={args.windows} "
          f"equil={args.equilibration}ns prod={args.production}ns legs={args.legs}")
    print(f"edges: {targets}\n", flush=True)

    outdir = pathlib.Path(args.dir) / ("pilot" if args.pilot else "run")
    outdir.mkdir(parents=True, exist_ok=True)
    results = []
    for tgt in targets:
        ligB = SmallMoleculeComponent.from_rdkit(mols[tgt])
        try:
            mapping = next(mapper.suggest_mappings(seed, ligB))
        except StopIteration:
            print(f"  {args.seed_id}->{tgt}: NO MAPPING, skipped", flush=True)
            continue
        n_map = len(mapping.componentA_to_componentB)
        print(f"=== edge {args.seed_id} -> {tgt}   mapped atoms {n_map} ===", flush=True)
        for leg in args.legs:
            if leg == "complex":
                a = ChemicalSystem({"protein": protein, "ligand": seed, "solvent": solvent})
                b = ChemicalSystem({"protein": protein, "ligand": ligB, "solvent": solvent})
            else:
                a = ChemicalSystem({"ligand": seed, "solvent": solvent})
                b = ChemicalSystem({"ligand": ligB, "solvent": solvent})
            name = f"{args.seed_id}_{tgt}_{leg}"
            tf = Transformation(stateA=a, stateB=b, mapping=mapping, protocol=protocol, name=name)
            shared = outdir / name / "shared"; shared.mkdir(parents=True, exist_ok=True)
            scratch = outdir / name / "scratch"; scratch.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            dag = tf.create()
            dagres = execute_DAG(dag, shared_basedir=shared, scratch_basedir=scratch,
                                 keep_shared=True, raise_error=False)
            dt = (time.time() - t0) / 60.0
            ok = dagres.ok()
            est, err = None, None
            if ok:
                try:
                    g = protocol.gather([dagres])
                    est = g.get_estimate(); err = g.get_uncertainty()
                except Exception as e:
                    print(f"    gather failed: {e}", flush=True)
            print(f"  {leg:<8} {'OK' if ok else 'FAILED'}  {dt:6.1f} min"
                  + (f"   dG = {est} +/- {err}" if est is not None else ""), flush=True)
            if not ok:
                for u in dagres.protocol_unit_failures:
                    print(f"    ! {str(u.exception)[:200]}", flush=True)
            results.append({"edge": f"{args.seed_id}->{tgt}", "leg": leg, "ok": bool(ok),
                            "minutes": round(dt, 1), "mapped_atoms": n_map,
                            "estimate": str(est) if est is not None else None,
                            "uncertainty": str(err) if err is not None else None})
            json.dump({"settings": {"repeats": args.repeats, "windows": args.windows,
                                    "equilibration_ns": args.equilibration,
                                    "production_ns": args.production},
                       "results": results},
                      open(outdir / "results.json", "w"), indent=1)
    print(f"\nwrote {outdir}/results.json")


if __name__ == "__main__":
    main()
