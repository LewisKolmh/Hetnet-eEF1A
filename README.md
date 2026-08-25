# EEF1A Hetnet Connectivity Search

A pure Hetnet Connectivity Search implementation for identifying
compounds with enriched network connectivity to EEF1A1 and EEF1A2.

## Primary objective

Identify compounds with statistically enriched heterogeneous-network
connectivity to EEF1A1 or EEF1A2 and explain those relationships using
ranked metapaths and individual paths.

## Initial biological context

The first disease-associated interaction examined will be the interaction
between eEF1A and HIV-1 reverse transcriptase.

## Methodological scope

The initial implementation follows the Hetnet Connectivity Search method:

1. Typed heterogeneous network construction
2. Metapath enumeration
3. Path-count calculation
4. Degree-weighted path counts
5. Degree-preserving network permutations
6. Degree-grouped null distributions
7. Gamma-hurdle significance calculations
8. Multiple-testing correction
9. Individual path ranking
10. Interactive graph visualisation

Machine learning, molecular docking, chemical fingerprints, graph
embeddings and NanoBiT results are excluded from the initial prediction
method.

## Status

Complete: the pipeline has been run end-to-end for two scopes,
`eef1a-only` (EEF1A1/EEF1A2 direct neighborhood only) and
`full-interactome` (all 18 seed proteins from the STRING interactome).

## Pipeline

Run from the repo root, `PYTHONPATH=src` or `PYTHONPATH=.` as noted:

1. `src/extract_seed_proteins.py` — parse `data/raw/eef1a_string_interactions.csv`
   into `data/raw/seed_proteins.tsv` (18 seed proteins).
2. `src/s01_download_chembl_compounds.py --scope {eef1a-only,full-interactome}` —
   ChEMBL compound-binds-gene edges, with on-disk JSON caching so an interrupted
   run resumes rather than restarting.
3. `src/s02_uniprot_and_go.py` — UniProt accession mapping + GO annotations.
4. `src/s03_reactome_pathways.py` — Reactome pathway membership.
5. `src/s04_prepare_string_edges.py` — STRING gene-interacts-gene edges,
   NCBI-gene-ID normalized.
6. `src/build_nodes.py`, `src/build_matrices.py` — typed node tables and
   per-metaedge sparse adjacency matrices.
7. `src/dwpc.py` / `src/compute_all_dwpcs.py` — degree-weighted path counts
   per metapath (CbG, CbGiG, CbGpBP, CbGpMF, CbGpCC, CbGpPW) for every
   compound against EEF1A1 and EEF1A2.
8. `src/xswap.py` / `src/compute_null_distribution.py` — XSwap degree-preserving
   permutation (200 permutations) and the resulting null DWPC distributions.
9. `src/fit_gamma_hurdle.py --scope <scope>` — method-of-moments gamma-hurdle
   fit per (metapath, target_gene) group.
10. `src/compute_pvalues.py --scope <scope>` — right-tail gamma p-values with
    Bonferroni correction per (metapath, target_gene) group.
11. `src/rank_compounds.py --scope <scope>` — final ranked compound table
    (`results/hetnet_ranked_compounds.<scope>.csv`), one row per compound
    at its most significant metapath/target.
12. `src/compare_to_resko.py --scope <scope> --resko-path <path>` — merges
    the hetnet ranking against a prior RESKO (direct-evidence, potency +
    relation-aware scoring) ranking for the same target set, and reports
    overlap/divergence between the two methods
    (`results/hetnet_vs_resko_comparison.<scope>.csv`).

Run `PYTHONPATH=. python -m pytest tests/ -q` to run the test suite
(35 tests covering the real fetched/computed data at every stage, plus
synthetic-data recovery tests for the gamma-hurdle fit).

## Key result

At `full-interactome` scope, every compound in the prior RESKO ranking is
independently flagged significant (Bonferroni p<0.05) by the hetnet
approach, with broadly consistent rank ordering between the two methods —
a cross-validating result despite the two methods scoring fundamentally
different evidence (RESKO: direct literature potency/relation evidence;
hetnet: network-context enrichment against a degree-matched null). At
`eef1a-only` scope, several RESKO top compounds lose significance once the
network is restricted to the immediate EEF1A1/EEF1A2 neighborhood,
showing that network scope materially changes what the hetnet approach
corroborates. See `results/hetnet_vs_resko_rank_comparison.png` and
`results/hetnet_network_graph.png`.
