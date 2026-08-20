"""Create the curated EEF1A1-to-HIV-1 RT interaction edge."""

import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

NODE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nodes"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "edges"
    / "gene_interacts_disease_partner.tsv"
)

FIELDNAMES = [
    "source_id",
    "target_id",
    "relationship",
    "directed",
    "interaction_scope",
    "evidence_type",
    "experimental_methods",
    "affinity_type",
    "affinity_value",
    "affinity_unit",
    "partner_construct",
    "host_construct",
    "source_database",
    "source_record_id",
    "publication_id",
    "publication_year",
    "retrieval_date",
    "confidence",
    "isoform_assignment",
    "notes",
]

ROWS = [
    {
        "source_id": "NCBI_Gene:1915",
        "target_id": "HIV1_RT",
        "relationship": "interacts",
        "directed": "false",
        "interaction_scope": (
            "direct_physical_interaction"
        ),
        "evidence_type": (
            "biophysical_and_cellular"
        ),
        "experimental_methods": (
            "biolayer interferometry; "
            "co-immunoprecipitation"
        ),
        "affinity_type": "KD",
        "affinity_value": "3-4",
        "affinity_unit": "nM",
        "partner_construct": (
            "HIV-1 RT p66/p51 heterodimer "
            "and RT subunits"
        ),
        "host_construct": (
            "full-length human EEF1A1 with "
            "C-terminal MYC/DDK tag"
        ),
        "source_database": "Primary literature",
        "source_record_id": (
            "Li_et_al_2015_PLoS_Pathogens"
        ),
        "publication_id": (
            "DOI:10.1371/journal.ppat.1005289"
        ),
        "publication_year": "2015",
        "retrieval_date": "2026-08-20",
        "confidence": "direct_experimental",
        "isoform_assignment": "EEF1A1",
        "notes": (
            "The publication explicitly defines "
            "eEF1A1 as eEF1A and uses purified "
            "full-length human EEF1A1 in BLI "
            "experiments."
        ),
    },
]


def read_ids(path: Path) -> set:
    """Read node identifiers from a TSV file."""

    with path.open(
        encoding="utf-8",
        newline="",
    ) as handle:
        return {
            row["id"]
            for row in csv.DictReader(
                handle,
                delimiter="\t",
            )
        }


def main() -> None:
    """Write and validate the PPI edge table."""

    gene_ids = read_ids(
        NODE_DIR / "gene.tsv"
    )

    partner_ids = read_ids(
        NODE_DIR / "disease_partner.tsv"
    )

    for row in ROWS:
        assert row["source_id"] in gene_ids
        assert row["target_id"] in partner_ids

        assert (
            row["relationship"]
            == "interacts"
        )

        assert row["directed"] == "false"

        assert (
            row["isoform_assignment"]
            == "EEF1A1"
        )

        assert (
            row["confidence"]
            == "direct_experimental"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=FIELDNAMES,
            delimiter="\t",
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(ROWS)

    with OUTPUT_PATH.open(
        encoding="utf-8",
        newline="",
    ) as handle:
        saved_rows = list(
            csv.DictReader(
                handle,
                delimiter="\t",
            )
        )

    assert len(saved_rows) == 1

    assert (
        saved_rows[0]["source_id"]
        == "NCBI_Gene:1915"
    )

    assert (
        saved_rows[0]["target_id"]
        == "HIV1_RT"
    )

    assert (
        saved_rows[0]["affinity_value"]
        == "3-4"
    )

    print(f"Created: {OUTPUT_PATH}")
    print("Rows: 1")

    print(
        "Edge: EEF1A1 <-> interacts "
        "<-> HIV1_RT"
    )

    print(
        "Evidence: direct BLI and co-IP; "
        "reported KD approximately 3-4 nM"
    )

    print(
        "Seed gene-partner edge "
        "validation passed."
    )


if __name__ == "__main__":
    main()