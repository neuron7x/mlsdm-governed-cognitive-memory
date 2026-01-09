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

## Risk-Aware Bibliography Governance
- Categorize sources by trust tier: **Standards**, **Peer-reviewed**, **arXiv**, **Official Report**.
- Critical subsystems (security, governance, safety) require a minimum trust tier of **Standards** or **Peer-reviewed** for their citations.
- High-risk subsystems (security, governance) must include **2+ Standards/Peer-reviewed** citations in the Literature Map.

## Authoritative Source Audit
Explicitly verify the authority of every source before adding it:
- **Publication type**: journal article, top-tier conference proceeding, academic book, or official standard/issuer publication.
- **Peer review**: confirm the venue has a formal peer-review process (journals/conferences/books) or is an official issuer (standards/government).
- **Canonical identifier**: must include **DOI**, **ISBN**, or an **official issuer URL** (e.g., NIST/ISO/IEEE). Use the canonical issuer URL only when DOI/ISBN is not applicable.
- **arXiv policy**: arXiv-only entries are allowed **only if no journal or conference version exists**. If a journal/conference version exists, use that canonical ID instead.

## Peer-review Upgrade
When a peer-reviewed version becomes available, upgrade canonical identifiers and metadata accordingly:
- **Journal/conference version exists** → switch `canonical_id_type` to **DOI** and update `canonical_id`/`canonical_url` to the DOI-based canonical record.
- **No peer-reviewed version exists** → keep the arXiv canonical ID **and** add a metadata note (e.g., `peer_review_note`) explaining why a peer-reviewed version is unavailable or not applicable.

## Authority-Proof Checklist (blocking)
Complete this checklist **before** starting the Update workflow:
- Source passes the Authoritative Source Audit (type + peer review + identifier).
- `metadata/identifiers.json` includes the authority-proof minimum fields:
  `canonical_id_type`, `canonical_id`, `canonical_url`, `verification_method`.
- If the source is arXiv-only, confirm and note that no journal/conference version exists.

## Update workflow
1) Add entry to `REFERENCES.bib` (unique key; include title+year + one of doi/url/eprint/isbn).
2) Add same entry to `REFERENCES_APA7.md` with `<!-- key: ... -->` marker (APA 7).
3) Add/update the record in `metadata/identifiers.json` and regenerate the row in `VERIFICATION.md`.
4) Validate locally:
   - `python scripts/validate_bibliography.py`
   - `cffconvert --validate -i CITATION.cff`
5) Open PR; CI blocks invalid metadata.

## Preflight Checklist
- Keep BibTeX, APA, metadata, and verification tables in sync (no drift between `REFERENCES.bib`, `REFERENCES_APA7.md`, `metadata/identifiers.json`, and `VERIFICATION.md`).
- Run `python scripts/validate_bibliography.py` (offline only; no network calls per `VERIFICATION.md` policy).
- Run `python scripts/docs/validate_literature_map.py` (offline only; no network calls per `VERIFICATION.md` policy).
- Before making changes, confirm every `paths:` entry exists in the repo (the `python scripts/docs/validate_literature_map.py` check enforces this).

## Safe Change Strategy
- one-commit update
- sync all four files
- verify locally

Incomplete updates (for example, adding a BibTeX entry without matching metadata/verification updates) are guaranteed to break CI.

## Literature map (CI-enforced)
- `docs/bibliography/LITERATURE_MAP.md` is required and validated in CI.
- Each subsystem entry must list 1–5 repo paths and **3+ citations** using `[@key]` from `REFERENCES.bib`.
- To add a subsystem: append a `## Subsystem Name` block with `paths:`, `citations:`, and a short rationale, then run `python scripts/docs/validate_literature_map.py`.

## Subsystem Coverage Audit
Any change to a subsystem or its API/modules **requires** auditing the corresponding block in `docs/bibliography/LITERATURE_MAP.md` before merge.

**Audit actions (blocking):**
1) **Path existence**: confirm every path in the subsystem block exists in the repo (same rule enforced by `python scripts/docs/validate_literature_map.py`).
2) **Citation completeness**: ensure the block still has **3+ citations** and each `[@key]` resolves to `REFERENCES.bib`.
3) **Rationale alignment**: update the rationale so it still matches the subsystem behavior, scope, and interfaces after the change.

Run `python scripts/docs/validate_literature_map.py` as the technical enforcement step whenever applicable.
