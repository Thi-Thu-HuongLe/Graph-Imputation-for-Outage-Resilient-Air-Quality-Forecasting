# Experimental protocols

These JSON files define the data splits, preprocessing rules, model families, random
seeds, outage conditions, calibration procedure, and statistical endpoints for the Salt
Lake County experiment and the Clark County geographic replication.

Only specifications and data-integrity records are distributed here. Generated freeze
files that contain hashes of trained checkpoints, as well as all checkpoints, predictions,
metrics, figures, and tables, are intentionally excluded. The included freeze scripts
recreate run-specific records after new development runs are completed.

The 2025 prepared tensors are completed holdouts. They must not be used to tune models,
select thresholds, choose comparator families, or refit calibration.
