"""Structural tractability flags for ranked compounds.

Why this exists
---------------
The STITCH layer contributes chemical-protein associations for everything STITCH
calls a "chemical", which includes ions (chloride, magnesium), buffer and
cryoprotectant components (glycerol, sulphate), and endogenous metabolites and
cofactors (GDP, pyrophosphate, phospho-amino acids). These have very high degree
in STITCH, and degree-weighting damps but does not remove them: on the
full-interactome graph they occupy most of the top 15 network ranks.

They are not candidate allosteric inhibitors, and no amount of network evidence
makes them one. Two mechanical flags are applied, both computed from structure
so that the criterion is inspectable rather than a curated exclusion list:

`tractable_small_molecule`
    Organic (contains carbon), carries at least one ring, no metal or metalloid
    atom, 250 <= MW <= 700 Da, and at least 18 heavy atoms. This is a
    deliberately permissive lead-like floor - it is meant to exclude ions,
    single-element species and small metabolites, not to impose Lipinski-style
    drug-likeness on genuine hits. Fragment-sized true binders would be excluded
    by it, so the flag is a column on the full table and a shortlist criterion;
    nothing is deleted.

`nucleotide_cofactor`
    Matches a nucleoside mono/di/tri-phosphate substructure. eEF1A is a
    GTP-binding protein, so GDP, GTP and their analogues are its cofactor, and
    STITCH records them as associations. They pass the structural floor on size
    and rings, which is exactly why they need a separate flag.

Both flags are advisory columns; `rank_compounds.py` requires
``tractable_small_molecule and not nucleotide_cofactor`` for the purchasable
shortlist and reports the excluded rows alongside it.
"""
from __future__ import annotations

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

MW_MIN = 250.0
MW_MAX = 700.0
HEAVY_ATOM_MIN = 18
RING_MIN = 1

# Metals and metalloids. A compound containing one of these is not a candidate
# small-molecule allosteric inhibitor for this project (arsenic trioxide,
# cisplatin, selenomethionine, magnesium all reach the shortlist without this).
METAL_METALLOID = frozenset({
    "Li", "Be", "Na", "Mg", "Al", "Si", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn",
    "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te",
    "Cs", "Ba", "La", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "Gd",
})

# Nucleoside phosphate: nucleobase-attached sugar carrying a phosphate chain.
# Two patterns so that both purine and pyrimidine attachment are caught.
_NUCLEOTIDE_SMARTS = (
    "[$(n1cnc2c1ncnc2),$(n1ccc(=O)[nH]c1=O),$(n1ccc(N)nc1=O)]"
    "[CH1]1[CH1][CH1][CH1]([CH2]OP(=O)([OX2,OX1-])[OX2,OX1-])O1",
    "OP(=O)([OX2,OX1-])OC[CH1]1O[CH1]([$(n2cnc3c2ncnc3),$(n2ccc(=O)[nH]c2=O)])[CH1][CH1]1",
)
_NUCLEOTIDE_PATTERNS = [Chem.MolFromSmarts(s) for s in _NUCLEOTIDE_SMARTS]

# Fallback for nucleotides whose sugar stereochemistry or protonation defeats the
# SMARTS above: a phosphate plus a recognised nucleobase in the same molecule.
_PHOSPHATE = Chem.MolFromSmarts("P(=O)([OX2H,OX1-,OX2])[OX2H,OX1-,OX2]")
_NUCLEOBASES = [
    Chem.MolFromSmarts(s) for s in (
        "c1ncnc2[nH0,nH]cnc12",   # purine
        "c1nc(=O)[nH]cc1",        # pyrimidinone
    )
]


def _descriptors(smiles: str | float | None) -> dict:
    if not isinstance(smiles, str) or not smiles.strip():
        return dict(mw=None, n_heavy_atoms=None, n_rings=None,
                    has_carbon=None, has_metal=None, nucleotide_cofactor=None)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return dict(mw=None, n_heavy_atoms=None, n_rings=None,
                    has_carbon=None, has_metal=None, nucleotide_cofactor=None)
    elements = {a.GetSymbol() for a in mol.GetAtoms()}
    is_nt = any(mol.HasSubstructMatch(p) for p in _NUCLEOTIDE_PATTERNS if p is not None)
    if not is_nt and _PHOSPHATE is not None and mol.HasSubstructMatch(_PHOSPHATE):
        is_nt = any(mol.HasSubstructMatch(b) for b in _NUCLEOBASES if b is not None)
    return dict(
        mw=float(Descriptors.MolWt(mol)),
        n_heavy_atoms=int(mol.GetNumHeavyAtoms()),
        n_rings=int(rdMolDescriptors.CalcNumRings(mol)),
        has_carbon="C" in elements,
        has_metal=bool(elements & METAL_METALLOID),
        nucleotide_cofactor=bool(is_nt),
    )


def annotate_tractability(df: pd.DataFrame, smiles_col: str = "canonical_smiles") -> pd.DataFrame:
    """Add structural descriptors and the two advisory flags to `df`.

    Rows with no parsable structure get ``tractable_small_molecule = False`` and
    NaN descriptors: an unverifiable structure is not a shortlist candidate.
    """
    # `df` may carry duplicate index labels (one row per compound after a merge),
    # so align positionally rather than on the index.
    desc = pd.DataFrame([_descriptors(s) for s in df[smiles_col]])
    out = df.reset_index(drop=True)
    for col in desc.columns:
        out[col] = desc[col].values
    out["tractable_small_molecule"] = (
        out["has_carbon"].astype("boolean").fillna(False).astype(bool)
        & ~out["has_metal"].astype("boolean").fillna(True).astype(bool)
        & (out["n_rings"].fillna(0) >= RING_MIN)
        & (out["mw"].fillna(0) >= MW_MIN)
        & (out["mw"].fillna(1e9) <= MW_MAX)
        & (out["n_heavy_atoms"].fillna(0) >= HEAVY_ATOM_MIN)
    )
    out["nucleotide_cofactor"] = (
        out["nucleotide_cofactor"].astype("boolean").fillna(False).astype(bool)
    )
    # Every excluded row states why, in the same call that excludes it: a reader
    # of the ranked table should never have to reverse-engineer the criterion.
    out["structural_exclusion_reason"] = [
        "" if t else exclusion_reason(r)
        for t, (_, r) in zip(out["tractable_small_molecule"], out.iterrows())
    ]
    return out


def exclusion_reason(row: pd.Series) -> str:
    """Human-readable reason a compound failed the structural floor ('' if it passed)."""
    if row.get("mw") is None or pd.isna(row.get("mw")):
        return "no parsable structure"
    reasons = []
    if not row.get("has_carbon"):
        reasons.append("inorganic (no carbon)")
    if row.get("has_metal"):
        reasons.append("contains metal/metalloid")
    if (row.get("n_rings") or 0) < RING_MIN:
        reasons.append("acyclic")
    if row["mw"] < MW_MIN:
        reasons.append(f"MW {row['mw']:.0f} < {MW_MIN:.0f}")
    elif row["mw"] > MW_MAX:
        reasons.append(f"MW {row['mw']:.0f} > {MW_MAX:.0f}")
    if (row.get("n_heavy_atoms") or 0) < HEAVY_ATOM_MIN:
        reasons.append(f"{int(row.get('n_heavy_atoms') or 0)} heavy atoms < {HEAVY_ATOM_MIN}")
    return "; ".join(reasons)
