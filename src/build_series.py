"""Build an RBFE-ready congeneric series around a pose-validated candidate.

Why this exists: relative binding free energy (FEP/TI) is the rigorous method
for ranking binders, and it is the one method this project could not use. RBFE
alchemically morphs one ligand into another, which requires a COMMON CORE to
perturb around. Our five candidates were selected at pairwise Tanimoto < 0.45
precisely to be structurally diverse -- the diversity filter's whole purpose --
so the candidate set is the worst possible input to RBFE. The fix is not to
abandon diversity but to use it differently: diversify to FIND a chemotype,
then elaborate around the one that survived validation.

cand4 and cand5 are the two whose docked poses reproduce the full imatinib /
nilotinib type-II signature (hinge Met318 backbone H-bond, gatekeeper Thr315,
alphaC Glu286, DFG Asp381 backbone). Elaborating one of those is defensible in
a way that elaborating cand1 -- the RF's most confident molecule, whose pose
never makes a polar hinge contact -- would not be.

Two routes, run together, because they answer different questions:

  GENERATED  Sample the trained policy and keep molecules containing the seed's
             Bemis-Murcko core. Asks whether the RL model can populate a series
             on demand, which is the more interesting scientific claim and
             reuses the model rather than sidelining it.
  ENUMERATED Attach an R-group library at one defined position on the core.
             Guarantees a true single-point-variation series regardless of what
             the generator happens to produce, so the deliverable does not
             depend on sampling luck.

What makes the output RBFE-ready, and is checked rather than assumed:
  * every member contains the common core (substructure match, enforced)
  * perturbations are small -- heavy-atom delta from the seed is reported per
    member, since FEP edge cost and error both grow with perturbation size
  * NO net-charge changes: charge-changing perturbations need special handling
    (co-alchemical ions, finite-size corrections) and are excluded here
  * the series spans a range of predicted activity, since a series where every
    member is equipotent teaches the free-energy calculation nothing

Usage:
    python src/build_series.py --seed-mol cand5 --n-sample 40000
"""
import argparse, json, sys
import numpy as np
sys.path.insert(0, "/workspace/rl_chemistry/src")
from eval_policy import load, sample
from reward_model import RewardModel
from train_rl import ADReference, DATA, CKPT
from smiles_utils import canonicalize
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, Crippen, rdMolDescriptors, rdFMCS
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")

# Seed cores with one marked attachment point. Both seeds are the pose-validated
# pair; the marked position is a solvent-exposed substituent in the docked pose,
# which is where medicinal chemistry varies a series and where perturbations
# perturb affinity without destroying the binding mode.
SEEDS = {
    "cand5": {
        "smiles": "Cc1ccc(CC(=O)N2CCN(CC(N)=O)CC2)cc1-c1ccc2cc(NC(=O)C3CC3(C)C)ncc2c1",
        "scaffold_r": "Cc1ccc(CC(=O)N2CCN([*:1])CC2)cc1-c1ccc2cc(NC(=O)C3CC3(C)C)ncc2c1",
    },
    "cand4": {
        "smiles": "CNC(=O)NC(=O)c1ccccc1Sc1ccc2c(/C=C/c3ccccn3)n[nH]c2c1",
        "scaffold_r": "[*:1]NC(=O)c1ccccc1Sc1ccc2c(/C=C/c3ccccn3)n[nH]c2c1",
    },
}

# Neutral, synthetically ordinary R-groups spanning size, polarity and
# H-bonding. Charged groups are deliberately absent (see docstring).
RGROUPS = [
    ("H", "[H]"), ("Me", "C"), ("Et", "CC"), ("nPr", "CCC"), ("iPr", "C(C)C"),
    ("cPr", "C1CC1"), ("CH2CONH2", "CC(N)=O"), ("CH2CH2OH", "CCO"),
    ("CH2CN", "CC#N"), ("CH2CF3", "CC(F)(F)F"), ("MeOEt", "CCOC"),
    ("Ac", "C(C)=O"), ("MeSO2", "S(C)(=O)=O"), ("CH2-cPr", "CC1CC1"),
    ("oxetanyl", "C1COC1"), ("CH2Ph", "Cc1ccccc1"), ("THP", "C1CCOCC1"),
    ("CONHMe", "C(=O)NC"), ("CH2CH2OMe", "CCOC"), ("tBu", "C(C)(C)C"),
]


def props(m):
    return {"mw": round(Descriptors.MolWt(m), 1), "logp": round(Crippen.MolLogP(m), 2),
            "hbd": rdMolDescriptors.CalcNumHBD(m), "hba": rdMolDescriptors.CalcNumHBA(m),
            "rotb": rdMolDescriptors.CalcNumRotatableBonds(m),
            "heavy": m.GetNumHeavyAtoms(),
            "charge": Chem.GetFormalCharge(m)}


def enumerate_series(scaffold_r):
    """Attach each R-group at the [*:1] position."""
    out = []
    for name, r in RGROUPS:
        smi = scaffold_r.replace("[*:1]", r) if r != "[H]" else scaffold_r.replace("([*:1])", "").replace("[*:1]", "")
        c = canonicalize(smi)
        if c:
            out.append((name, c))
    # dedupe, keep first label
    seen, uniq = set(), []
    for name, c in out:
        if c not in seen:
            seen.add(c)
            uniq.append((name, c))
    return uniq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=f"{CKPT}/rl_v5d/policy_latest.pt")
    ap.add_argument("--seed-mol", choices=list(SEEDS), default="cand5")
    ap.add_argument("--n-sample", type=int, default=40000)
    ap.add_argument("--max-heavy-delta", type=int, default=12,
                    help="reject members whose heavy-atom count differs from the seed by more "
                         "than this; FEP edge cost and error both grow with perturbation size")
    ap.add_argument("--min-p", type=float, default=0.30)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    seed = SEEDS[args.seed_mol]
    seed_mol = Chem.MolFromSmiles(seed["smiles"])
    seed_props = props(seed_mol)
    core = MurckoScaffold.GetScaffoldForMol(seed_mol)
    core_smi = Chem.MolToSmiles(core)
    print(f"seed {args.seed_mol}  heavy {seed_props['heavy']}  charge {seed_props['charge']}")
    print(f"core (Bemis-Murcko): {core_smi}\n")

    rm = RewardModel()
    ad = ADReference()
    members = {}

    # ---- route 1: enumerated R-groups on the marked position
    for label, smi in enumerate_series(seed["scaffold_r"]):
        members[smi] = {"smiles": smi, "origin": "enumerated", "r_group": label}

    # ---- route 2: generated molecules containing the same core
    policy, vocab, step = load(args.policy)
    print(f"sampling {args.n_sample} from {args.policy} (step {step})...", flush=True)
    raw = sample(policy, vocab, args.n_sample, 100, 1.0)
    canon = {c for c in (canonicalize(s) if s else None for s in raw) if c}
    n_match = 0
    for smi in canon:
        m = Chem.MolFromSmiles(smi)
        if m is None or not m.HasSubstructMatch(core):
            continue
        n_match += 1
        members.setdefault(smi, {"smiles": smi, "origin": "generated", "r_group": None})
    print(f"  {len(canon)} unique valid; {n_match} contain the core "
          f"({100.0*n_match/max(1,len(canon)):.3f}%)\n")

    # ---- filter to an RBFE-usable set
    smis = list(members)
    sc = rm.score(smis)
    tan = ad.max_tanimoto(smis, [True] * len(smis))
    kept = []
    for i, smi in enumerate(smis):
        m = Chem.MolFromSmiles(smi)
        p = props(m)
        d = abs(p["heavy"] - seed_props["heavy"])
        if p["charge"] != seed_props["charge"]:
            continue                       # no charge-changing perturbations
        if d > args.max_heavy_delta:
            continue
        if sc["p_active"][i] < args.min_p:
            continue
        rec = dict(members[smi]); rec.update(p)
        rec.update({"p_active": float(sc["p_active"][i]),
                    "uncertainty": float(sc["uncertainty"][i]),
                    "heavy_delta_from_seed": d,
                    "max_tanimoto_train": float(tan[i]),
                    "is_seed": smi == canonicalize(seed["smiles"])})
        kept.append(rec)
    kept.sort(key=lambda r: (-r["p_active"]))

    # ---- verify the common core actually spans the series
    mols = [Chem.MolFromSmiles(r["smiles"]) for r in kept]
    mcs = rdFMCS.FindMCS(mols, timeout=60, completeRingsOnly=True,
                         ringMatchesRingOnly=True) if len(mols) > 1 else None
    print(f"kept {len(kept)} members "
          f"({sum(1 for r in kept if r['origin']=='enumerated')} enumerated, "
          f"{sum(1 for r in kept if r['origin']=='generated')} generated)")
    if mcs:
        print(f"MCS across series: {mcs.numAtoms} atoms / {mcs.numBonds} bonds")
        print(f"  {mcs.smartsString[:110]}")
        frac = mcs.numAtoms / np.mean([m.GetNumHeavyAtoms() for m in mols])
        print(f"  covers {100*frac:.0f}% of the average member -- "
              f"{'good RBFE core' if frac > 0.6 else 'CORE TOO SMALL for clean RBFE'}")
    pa = np.array([r["p_active"] for r in kept])
    hd = np.array([r["heavy_delta_from_seed"] for r in kept])
    print(f"predicted activity spans P {pa.min():.3f}-{pa.max():.3f} "
          f"({'usable spread' if pa.max()-pa.min() > 0.2 else 'TOO FLAT to be informative'})")
    print(f"perturbation size: mean {hd.mean():.1f}, max {hd.max()} heavy atoms from seed")

    print(f"\n{'#':>3} {'origin':<11} {'R':<10} {'P(act)':>7} {'unc':>6} {'dHeavy':>7} {'MW':>7}")
    for i, r in enumerate(kept[:30], 1):
        print(f"{i:>3} {r['origin']:<11} {str(r['r_group'] or '-'):<10} {r['p_active']:>7.3f} "
              f"{r['uncertainty']:>6.3f} {r['heavy_delta_from_seed']:>7d} {r['mw']:>7.1f}"
              + ("   <- seed" if r["is_seed"] else ""))

    out = args.out or f"/workspace/rl_chemistry/series_{args.seed_mol}.json"
    json.dump({"seed": args.seed_mol, "seed_smiles": seed["smiles"], "core": core_smi,
               "mcs": mcs.smartsString if mcs else None,
               "n_members": len(kept), "members": kept}, open(out, "w"), indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
