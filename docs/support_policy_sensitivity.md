# Gate-A support-policy sensitivity

This analysis is a sensitivity study of **support readiness**, not a replacement turbulence classifier.
The primary Gate-A policy remains the historical `legacy_global_fraction` rule until an explicit
project-level decision changes it.

The canonical Pass-2 comparator set is locked to four policies:

1. `legacy_global_fraction` — common finite-support intersection divided by the complete deposited
   Cartesian lattice, using the threshold recorded by the source Gate-A run.
2. `strict_mask_consistency` — exact realization-to-realization support agreement,
   `intersection / union = 1`.
3. `fully_finite_local_windows` — at least one actual Gate-A analysis window with finite-support
   fraction `1.0`.
4. `strict_mask_and_local_windows` — conjunction of policies 2 and 3.

The exact mask criterion deliberately avoids a tunable near-unity threshold. The local criterion
uses the same fully finite rectangular windows already required by the spatial correlation and
coarse-grained transfer calculations. Neither criterion is chosen from the resulting turbulence
outcome.

The command is:

```bash
aorticturb support-sensitivity \
  --root <verified-dataset-root> \
  --decomposition <phase-decomposition-run> \
  --regime <source-gate-a-run> \
  --output <workspace-root>
```

The run executes the mandatory dataset startup gate and writes a reproducible analysis directory
under `04_analysis/01_laminar_vs_unsteady/`. Principal outputs are:

- `support_sensitivity.csv` — one row per phase and support policy;
- `support_sensitivity.json` — policy criteria, observations, readiness, and projected Gate-A
  decisions using the unchanged downstream evidence;
- `sensitivity_summary.json` — whether readiness or projected Gate-A decision is policy-sensitive;
- standard `run_manifest.json`, `config.json`, `input_hashes.txt`, `methods.md`, and logs.

Projected Gate-A decisions are explicitly counterfactual sensitivity results. The command does not
modify the source regime run, does not promote outputs to `05_results`, and does not trigger the KO
branch.
