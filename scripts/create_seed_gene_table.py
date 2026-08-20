"""Create the initial EEF1A1 and EEF1A2 node table."""

import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nodes"
    / "gene.tsv"
)

FIELDNAMES = [
    "id",
    "symbol",
    "name",
    "taxon_id",
    "hgnc_id",
    "ensembl_gene_id",
    "uniprot_accession",
    "uniprot_reviewed",
    "source",
    "source_record_id",
    "retrieval_date",
    "isoform_assignment",
    "notes",
]

ROWS = [
    {
        "id": "NCBI_Gene:1915",
        "symbol": "EEF1A1",
        "name": (
            "eukaryotic translation elongation "
            "factor 1 alpha 1"
        ),
        "taxon_id": "9606",
        "hgnc_id": "HGNC:3189",
        "ensembl_gene_id": "ENSG00000156508",
        "uniprot_accession": "P68104",
        "uniprot_reviewed": "true",
        "source": "NCBI Gene; UniProt",
        "source_record_id": "1915; P68104",
        "retrieval_date": "2026-08-20",
        "isoform_assignment": "EEF1A1",
        "notes": "Canonical human EEF1A1 seed node.",
    },
    {
        "id": "NCBI_Gene:1917",
        "symbol": "EEF1A2",
        "name": (
            "eukaryotic translation elongation "
            "factor 1 alpha 2"
        ),
        "taxon_id": "9606",
        "hgnc_id": "HGNC:3192",
        "ensembl_gene_id": "ENSG00000101210",
        "uniprot_accession": "Q05639",
        "uniprot_reviewed": "true",
        "source": "NCBI Gene; UniProt",
        "source_record_id": "1917; Q05639",
        "retrieval_date": "2026-08-20",
        "isoform_assignment": "EEF1A2",
        "notes": "Canonical human EEF1A2 seed node.",
    },
]


def main() -> None:
    """Write and validate the initial Gene node table."""

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

    assert len(saved_rows) == 2

    assert {
        row["symbol"]
        for row in saved_rows
    } == {
        "EEF1A1",
        "EEF1A2",
    }

    assert len({
        row["id"]
        for row in saved_rows
    }) == 2

    assert all(
        row["taxon_id"] == "9606"
        for row in saved_rows
    )

    assert all(
        row["symbol"] != "eEF1A"
        for row in saved_rows
    )

    print(f"Created: {OUTPUT_PATH}")
    print("Rows: 2")
    print("Targets: EEF1A1, EEF1A2")
    print("Seed gene table validation passed.")


if __name__ == "__main__":
    main()