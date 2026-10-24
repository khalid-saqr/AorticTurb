# Output contract

AorticTurb writes derived data only beneath the user-selected output root. It never writes
into the input dataset's `01_raw_data` tree.

## Branch directories

- `01_laminar_vs_unsteady/` — decomposition and Gate-A runs
- `02_kolmogorov_obukhov/` — spatial KO runs
- `03_taylor_hypothesis/` — necessary Taylor-admissibility runs
- `04_lumley_anisotropy/` — anisotropy runs
- `05_canonical_diagnostics/` — derivative/intermittency runs
- `05_results/` — integrated classification runs
- `study_runs/` — pointers tying one end-to-end execution together

Each run has a unique `run_*` identifier. `run_manifest.json` is finalized with hashes of
all produced artifacts except itself, so figures/tables can be traced to a specific
configuration and verified input checksum ledger.

`input_hashes.txt` includes the verified package checksum ledger plus hashes of derived
upstream summaries consumed by a stage. This binds every downstream claim to a concrete
upstream run without re-hashing the entire 2+ GB raw dataset for every stage.
