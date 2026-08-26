"""SMILES validity and canonicalization helpers shared across the pipeline.

Deliberately RDKit-only so the RL loop can score every sampled batch without
pulling in training-side imports.
"""
from rdkit import Chem, RDLogger

# RDKit writes parse failures straight to stderr; we detect them via None returns.
RDLogger.DisableLog("rdApp.*")


def canonicalize(smiles):
    """Canonical SMILES, or None if RDKit cannot parse *or* sanitize it.

    MolFromSmiles sanitizes by default, so this rejects both syntax errors
    (unclosed rings) and chemistry errors (impossible valences).
    """
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def is_valid(smiles):
    return canonicalize(smiles) is not None


def valid_fraction(smiles_list):
    """Fraction of a generated batch RDKit accepts -- the core RL/eval metric."""
    if not smiles_list:
        return 0.0
    return sum(is_valid(s) for s in smiles_list) / len(smiles_list)


def largest_fragment(smiles):
    """Strip salts/counterions by keeping the largest covalent fragment."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if not frags:
        return None
    return Chem.MolToSmiles(max(frags, key=lambda m: m.GetNumHeavyAtoms()))
