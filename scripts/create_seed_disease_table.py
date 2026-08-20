"""Create the initial disease node table."""

import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nodes"
    / "disease.tsv"
)

FIELDNAMES = [
    "id",
    "name",
    "synonym",
    "mondo_id",
    "mesh_id",
    "source",
    "source_record_id",
    "retrieval_date",
    "notes",
]

ROWS = [
    {
        "id": "DOID:526",
        "name": (
            "human immunodeficiency virus "
            "infectious disease"
        ),
        "synonym": "HIV infection",
        "mondo_id": "MONDO:0005109",
        "mesh_id": "MESH:D015658",
        "source": "Disease Ontology",
        "source_record_id": "DOID:526",
        "retrieval_date": "2026-08-20",
        "notes": (
            "Initial disease context for the HIV-1 RT "
            "and eEF1A interaction."
        ),
    },
]


def main() -> None:
    """Write and validate the Disease node table."""

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
    assert saved_rows[0]["id"] == "DOID:526"

    assert (
        saved_rows[0]["mondo_id"]
        == "MONDO:0005109"
    )

    assert (
        saved_rows[0]["synonym"]
        == "HIV infection"
    )

    print(f"Created: {OUTPUT_PATH}")
    print("Rows: 1")
    print("Disease: DOID:526 - HIV infection")
    print("Seed disease table validation passed.")


if __name__ == "__main__":
    main()