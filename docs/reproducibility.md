# Reproducibility

A publication-grade AorticTurb result should report:

1. the curated Zenodo dataset DOI and archive SHA-256;
2. the AorticTurb Git commit or release;
3. the relevant run ID(s);
4. the configuration JSON stored in each run;
5. the input checksum ledger hash;
6. bootstrap sample count and random seed;
7. local window shape(s) actually used after adaptive admissibility selection;
8. every diagnostic marked unresolved or inadmissible; and
9. the phase-specific terminal classification.

The default seed is fixed for reproducibility. Bootstrap resampling is always across the
realization axis. Spatial points are never treated as independent realizations merely to
inflate sample size.
