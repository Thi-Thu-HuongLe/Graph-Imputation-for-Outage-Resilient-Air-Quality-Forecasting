# Salt Lake County EPA AQS data

This directory contains prepared 2021--2024 development data, the completed 2025 temporal
holdout, availability summaries, preprocessing audits, network-selection records, and
redacted source manifests for seven Salt Lake County stations.

Raw API responses are not included. Reconstruct them with `scripts/download_epa_aqs.py`
using state FIPS `49`, county FIPS `035`, years 2021--2025, and pollutant codes `88101`,
`42602`, `44201`, `42101`, and `42401`. Protocol-specific download stages and chronology
are defined in `journal_protocol/`.

See the repository-level `DATA.md` for schema and reuse information.
