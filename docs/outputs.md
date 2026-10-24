# Output contract

AorticTurb treats the user-selected `--output` path as a study workspace root and mirrors
the `AGENTS.md` project directory contract. It never writes into the input dataset's
`01_raw_data` tree.

```text
WORKSPACE/
  03_processed_data/
  04_analysis/
    01_laminar_vs_unsteady/
    02_kolmogorov_obukhov/
    03_taylor_hypothesis/
    04_lumley_anisotropy/
    05_canonical_diagnostics/
  05_results/
    study_runs/
```

Reusable phase-decomposition arrays are stored under `03_processed_data/<run_id>/`.
Their SHA-256 values are recorded in `processed_outputs.json` inside the corresponding
analysis run. Analysis diagnostics, figures, tables, logs, and methods remain under their
respective `04_analysis` branch. Integrated classifications are stored under `05_results`.

Each run has a unique `run_*` identifier. `run_manifest.json` is finalized with hashes of
all produced in-run artifacts except itself, so figures/tables can be traced to a specific
configuration and verified input checksum ledger.

`input_hashes.txt` includes the verified package checksum ledger plus hashes of derived
upstream summaries consumed by a stage. This binds every downstream claim to a concrete
upstream run without re-hashing the entire raw dataset for every stage.
