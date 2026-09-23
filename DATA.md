# Data documentation

## Source and scope

All measurements are public hourly sample records from the U.S. EPA Air Quality System
(AQS). The two released networks are:

- Salt Lake County, Utah (`state=49`, `county=035`); and
- Clark County, Nevada (`state=32`, `county=003`).

Both networks use seven monitoring sites and the parameter codes PM2.5 (`88101`), NO2
(`42602`), O3 (`44201`), CO (`42101`), and SO2 (`42401`). Pollutant support is unequal
across sites; the observation and target masks in each NPZ file must be respected.

Development data cover 2021--2024. The 2025 files are completed chronological holdouts.
The canonical time axis is hourly UTC. Calendar features use each AQS network's fixed
local-standard offset rather than daylight-saving civil time.

## Included files

Each `prepared_development/` and `prepared_holdout/` directory contains:

- a compressed NumPy tensor (`.npz`);
- `availability.csv`, summarizing native observation support; and
- a JSON preprocessing or preparation report.

The `development_audit/`, `selection_audit/`, and `provenance/` directories contain only
data-quality summaries, network-selection records, and redacted source manifests. Raw API
responses are intentionally omitted because they can be retrieved from AQS using the
included downloader.

## NPZ schema

The prepared tensors are loaded with `aqriskformer.data.PreparedAirQuality.load`. Their
arrays include:

- `timestamps_ns`: hourly timestamps;
- `values`: scaled, causally filled model inputs;
- `observed_mask`: native input availability before filling;
- `time_gaps`: hours since the most recent observation;
- `calendar`: cyclic time features;
- `station_static`: normalized station coordinates;
- `native_pollutants`: unscaled pollutant observations;
- `target_mask`: valid scored targets;
- `center` and `scale`: training-only robust-scaling parameters;
- `risk_thresholds`: training-only Q80, Q90, and Q95 thresholds;
- `mase_scale24`: training-only seasonal-naive error scales; and
- `train_correlation_graph`: retained compatibility field from the prepared schema.

Metadata embedded in each archive record the station identifiers, pollutants, feature
names, and chronological split bounds.

## Preprocessing rules

Only one-hour sample-duration records are accepted. Concurrent finite readings at the
same site, pollutant, and UTC hour are merged by their median. Unavailable and nonfinite
measurements remain missing. Short gaps in model inputs are filled causally; original
observation masks and unscaled targets are retained separately. Scaling, risk thresholds,
and MASE denominators are estimated from development training data only.

The authoritative details are in `journal_protocol/development_preprocessing_protocol.json`
and `journal_protocol/external_development_preprocessing_protocol.json`.

## Integrity and reuse

SHA-256 values for the original compressed downloads are recorded in the redacted
provenance manifests. The manifests contain no API credentials. Prepared-data checksums
are retained in the protocol records where applicable.

The source data are provided by the U.S. EPA rather than authored by this repository.
Please cite the EPA AQS source and document any downstream filtering or transformations.
