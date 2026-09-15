"""Tests for assay provenance grading (src/s07_assay_provenance.py).

The grading exists because ChEMBL's quality fields cannot separate a targeted
binding measurement from a multiplexed chemoproteomics deposit: on this dataset
every such record carries confidence_score 9, assay_type B and an exact relation.
The discriminating signal is how many distinct seed proteins one document reports
for one compound. These tests use real assay descriptions from the deposits that
motivated the module.
"""
import pandas as pd

from s07_assay_provenance import (
    MULTIPLEX_MIN_PROTEINS,
    label_provenance,
)

KINOBEAD = "Binding affinity to human {} incubated for 45 mins by Kinobead based pull down assay"
LCMS = ("Inhibition of {} (unknown origin) assessed as fold change at 10 uM incubated "
        "for 1 hr by colloidal coomassie staining based LC-MS/MS analysis")
SPR = ("Binding affinity to {} expressed in human MDA-MB-231 cells using 1 uM biotinylated "
       "compound immobilized on streptavidin sensor chip by surface plasmon resonance method")
REPORTER = ("Inhibition of 17AAG-induced HSF1-mediated HSP72 expression in human U2OS cells "
            "preincubated for 1 hr followed by 17AAG addition measured after 18 hrs by ELISA")


def _frames(rows):
    """rows: (compound, protein, assay, document, description) -> (activities, assays)."""
    acts = pd.DataFrame([
        {"compound_id": c, "protein_id": p, "assay_chembl_id": a, "assay_type": "B"}
        for c, p, a, _, _ in rows
    ])
    assays = pd.DataFrame([
        {"assay_chembl_id": a, "assay_description": d, "assay_type_chembl": "B",
         "confidence_score": 9, "document_chembl_id": doc,
         "assay_organism": "Homo sapiens", "cell_chembl_id": None}
        for _, _, a, doc, d in rows
    ]).drop_duplicates("assay_chembl_id")
    return acts, assays


def test_multiplexed_pulldown_detected_from_document_breadth():
    proteins = ["EEF1A2", "EEF1B2", "EEF1D", "EEF1G", "ETF1", "RAN", "RPL3"]
    rows = [("C1", p, f"A{i}", "DOC1", KINOBEAD.format(p)) for i, p in enumerate(proteins)]
    out = label_provenance(*_frames(rows))
    assert len(proteins) >= MULTIPLEX_MIN_PROTEINS
    assert (out["evidence_grade"] == "multiplexed_pulldown").all()
    assert (out["max_seed_proteins_in_one_document"] == len(proteins)).all()


def test_targeted_pair_below_breadth_threshold_is_not_multiplexed():
    """A paralogue pair measured in one paper is targeted work, not a screen."""
    rows = [("C1", "EEF1A1", "A1", "DOC1", SPR.format("eEF1A1")),
            ("C1", "EEF1A2", "A2", "DOC1", SPR.format("eEF1A2"))]
    out = label_provenance(*_frames(rows))
    assert not out["multiplexed_proteomics"].any()
    assert (out["evidence_grade"] == "direct_biophysical").all()


def test_proteome_scale_readout_caught_below_the_breadth_threshold():
    """The LC-MS/MS fold-change deposit reports only 3 seed proteins per compound,
    under MULTIPLEX_MIN_PROTEINS, so document breadth alone would miss it."""
    rows = [("C1", p, f"A{i}", "DOC1", LCMS.format(p))
            for i, p in enumerate(["EEF1D", "EEF1G", "RPS3A"])]
    out = label_provenance(*_frames(rows))
    assert not out["multiplexed_proteomics"].any(), "fixture must sit below the threshold"
    assert (out["evidence_grade"] == "proteome_readout").all()


def test_biophysical_promotion_does_not_override_multiplexing():
    """A named method inside a proteome-wide deposit is still a proteome-wide
    deposit: the grade ladder checks multiplexing first."""
    proteins = ["EEF1A1", "EEF1A2", "RPL3", "RPS2", "RAN"]
    rows = [("C1", p, f"A{i}", "DOC1", SPR.format(p)) for i, p in enumerate(proteins)]
    out = label_provenance(*_frames(rows))
    assert (out["evidence_grade"] == "multiplexed_pulldown").all()


def test_pulldown_wording_alone_is_not_a_proteome_readout():
    """A biotinylated-probe pulldown read out by western blot against one named
    protein is targeted evidence. 'Pull down' must not demote it - three of the
    strongest eEF1A binders on the graph are measured this way."""
    rows = [("C1", "EEF1A1", "A1", "DOC1",
             "Binding affinity to eEF1A1 in human MDA-MB-231 cells using biotinylated "
             "compound using Western blot analysis based pull down assay")]
    out = label_provenance(*_frames(rows))
    assert not out["proteome_scale_readout"].any()
    assert out["evidence_grade"].iloc[0] == "assay_reported"


def test_cell_reporter_assay_is_not_promoted_to_biophysical():
    """ChEMBL types this HSF1 HSP72-expression ELISA as a binding assay (B). The
    module does not attempt a binding/functional split - it must simply not claim
    a biophysical method was used."""
    rows = [("C1", "HSF1", "A1", "DOC1", REPORTER)]
    out = label_provenance(*_frames(rows))
    assert out["evidence_grade"].iloc[0] == "assay_reported"
    assert not out["biophysical_method"].any()


def test_one_row_per_compound_protein_pair():
    rows = [("C1", "EEF1A1", "A1", "DOC1", SPR.format("eEF1A1")),
            ("C1", "EEF1A1", "A2", "DOC2", SPR.format("eEF1A1")),
            ("C2", "EEF1A1", "A3", "DOC1", SPR.format("eEF1A1"))]
    out = label_provenance(*_frames(rows))
    assert len(out) == 2
    assert not out.duplicated(["compound_id", "protein_id"]).any()
    c1 = out[out.compound_id == "C1"].iloc[0]
    assert c1["n_supporting_assays"] == 2
    assert c1["n_supporting_documents"] == 2


def test_every_edge_receives_exactly_one_grade():
    rows = [("C1", "EEF1A1", "A1", "DOC1", SPR.format("eEF1A1")),
            ("C2", "RPL3", "A2", "DOC2", KINOBEAD.format("RPL3")),
            ("C3", "EEF1D", "A3", "DOC3", LCMS.format("EEF1D")),
            ("C4", "HSF1", "A4", "DOC4", REPORTER)]
    out = label_provenance(*_frames(rows))
    grades = {"direct_biophysical", "assay_reported", "proteome_readout",
              "multiplexed_pulldown"}
    assert set(out["evidence_grade"]).issubset(grades)
    assert out["evidence_grade"].notna().all()
    assert len(out) == 4
