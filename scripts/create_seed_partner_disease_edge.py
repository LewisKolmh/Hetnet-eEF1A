"""Create the initial DiseasePartner-to-Disease edge table."""

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
    / "disease_partner_contributes_to_disease.tsv"
)

FIELDNAMES = [
    "source_id",
    "target_id",
    "relationship",
    "directed",
    "evidence_type",
    "source_database",
    "source_record_id",
    "publication_id",
    "retrieval_date",
    "confidence",
    "notes",
]

ROWS = [
    {
        "source_id": "HIV1_RT",
        "target_id": "DOID:526",
        "relationship": "contributes_to",
        "directed": "true",
        "evidence_type": "disease_aetiology",
        "source_database": (
            "Disease Ontology; primary literature"
        ),
        "source_record_id": "DOID:526",
        "publication_id": "",
        "retrieval_date": "2026-08-20",
        "confidence": "curated",
        "notes": (
            "HIV-1 reverse transcriptase is a viral "
            "protein encoded by HIV-1, which contributes "
            "to HIV infectious disease."
        ),
    },
]


def read_ids(path: Path) -> set[str]:
    """Return node identifiers from a TSV node table."""

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
    """Write and validate the first production edge."""

    partner_ids = read_ids(
        NODE_DIR / "disease_partner.tsv"
    )

    disease_ids = read_ids(
        NODE_DIR / "disease.tsv"
    )

    for row in ROWS:
        assert row["source_id"] in partner_ids
        assert row["target_id"] in disease_ids

        assert (
            row["relationship"]
            == "contributes_to"
        )

        assert row["directed"] == "true"

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
        == "HIV1_RT"
    )

    assert (
        saved_rows[0]["target_id"]
        == "DOID:526"
    )

    print(f"Created: {OUTPUT_PATH}")
    print("Rows: 1")

    print(
        "Edge: HIV1_RT -> contributes_to "
        "-> DOID:526"
    )

    print(
        "Seed partner-disease edge "
        "validation passed."
    )


if __name__ == "__main__":
    main()
    