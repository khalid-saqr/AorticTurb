# Pass 3 Colab integration

Pass 3 is an execution/integration pass over the already merged Pass-1 and Pass-2 scientific implementation.

The canonical notebook runs only:

1. verified local staging and the mandatory startup gate;
2. fresh phase-conditioned decomposition;
3. Gate A with the historical `legacy_global_fraction` support rule unchanged as the primary decision;
4. the pre-specified support-policy sensitivity analysis.

The runner records the exact Git commit used by every scientific run, validates the new decomposition support quantities (`union_support_nodes`, intersection/union, and exact mask identity), validates the locked four-policy sensitivity output, and writes a Pass-3 execution record under `results/colab_runs/`.

It does not call `run_all`, does not promote outputs to canonical `05_results`, and does not automatically execute KO, Taylor, Lumley, canonical diagnostics, or integrated classification.

The notebook intentionally pins an immutable commit so the scientific package and Pass-3 runner used in Colab are reproducible.
