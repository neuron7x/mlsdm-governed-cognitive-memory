# Bibliography — Repository Standard (2025)

This directory is the canonical source of bibliographic metadata for this repository.

## Files (single source of truth)
- `REFERENCES_APA7.md` — human-readable bibliography in **APA 7**.
- `REFERENCES.bib` — machine-readable **BibTeX**.
- `metadata/identifiers.json` — offline canonical identifiers + frozen metadata.
- `VERIFICATION.md` — verification table generated from `identifiers.json`.

Root:
- `CITATION.cff` — repository-level citation metadata (GitHub "Cite this repository").

## Policy
- Allowed: peer-reviewed journals, top-tier conferences (ACM/IEEE/USENIX), academic books, official standards (NIST/ISO/IEEE), widely-used arXiv preprints.
- Disallowed: personal blogs, unreviewed claims, non-stable URLs, sources without DOI/arXiv/canonical issuer URL.

## Update workflow
1) Add entry to `REFERENCES.bib` (unique key; include title+year + one of doi/url/eprint/isbn).
2) Add same entry to `REFERENCES_APA7.md` with `<!-- key: ... -->` marker (APA 7).
3) Add/update the record in `metadata/identifiers.json` and regenerate the row in `VERIFICATION.md`.
4) Validate locally:
   - `python scripts/validate_bibliography.py`
   - `cffconvert --validate -i CITATION.cff`
5) Open PR; CI blocks invalid metadata.
