# AGENTS.md — Hasler–Obrist Tomo-PIV Turbulence Study

**Status:** CANONICAL OPERATING CONTRACT  
**Contract version:** 1.0  
**Project:** `Hasler_Obrist_TomoPIV_Turbulence_Study`  
**Canonical data source:** Zenodo record 1163562, DOI `10.5281/zenodo.1163562`  
**Article:** Hasler & Obrist, PLOS ONE (2018), DOI `10.1371/journal.pone.0194384`

This file governs every coding, analysis, research, visualization, and manuscript agent operating inside this project. Read it before touching project data or generating scientific conclusions.

If a future instruction conflicts with this file, stop and request an explicit human decision unless the new instruction explicitly reopens or replaces the relevant rule. Do not silently reinterpret this contract.

---

## 1. Scientific mission

The project asks:

> **What kind of flow exists downstream of the prosthetic aortic valve, and which classical turbulence framework—if any—is admissible for describing it?**

The analysis must not assume turbulence, Kolmogorov–Obukhov behavior, isotropy, homogeneity, or Taylor's frozen-turbulence hypothesis in advance.

The required logical order is:

1. phase-conditioned ensemble decomposition;
2. determine whether the ensemble residual represents genuine stochastic multiscale motion rather than deterministic phase evolution or unresolved/noise-dominated variability;
3. if turbulence is established, test Kolmogorov–Obukhov (KO) behavior directly in space;
4. only after the turbulence decision, assess Taylor-hypothesis admissibility;
5. characterize Reynolds-stress anisotropy with Lumley invariants/maps;
6. apply canonical turbulence diagnostics as consistency and characterization tests;
7. synthesize one of the allowed terminal classifications.

Allowed primary classifications:

- `LAMINAR_UNSTEADY`
- `KO_TURBULENCE`
- `NON_KO_TURBULENCE`
- `UNRESOLVED_AT_AVAILABLE_RESOLUTION`

No other primary classification may be substituted without an explicit project-level change.

---

## 2. Canonical acquisition state

### 2.1 Velocity data

The acquired and verified velocity subset is exactly:

- geometry: `M` (medium aortic root);
- measurement domain: `AAo` (ascending aorta);
- phases: `t018`, `t021`, `t024`;
- physical phase labels: 0.18 s, 0.21 s, 0.24 s;
- realizations per phase: 16;
- realization names: `B00001.npy` through `B00016.npy`;
- total velocity fields: 48;
- bytes per field: 47,164,880;
- total velocity bytes: 2,263,914,240.

Raw path pattern:

`01_raw_data/M_medium_root/AAo/<phase>/Bxxxxx.npy`

The selected three-phase kernel is centered on source-paper peak flow at 0.21 s, with adjacent deposited phases at ±0.03 s.

Do not silently broaden the acquired dataset. Velocity data for M/SOV, S, L, and NS are not part of the canonical acquired velocity subset.

### 2.2 Velocity array structure

The repository defines velocity files as NumPy arrays with:

- shape `(N_nodes, 6)`;
- columns `[X, Y, Z, U, V, W]`;
- `U, V, W` the three instantaneous velocity components.

The validated representative M/AAo/t021 field has:

- shape `(982600, 6)`;
- dtype `float64`;
- fully finite coordinate columns;
- source NaNs in the velocity columns outside the valid measurement support;
- 390,465 rows with finite `U,V,W`;
- 1,776,405 non-finite velocity entries in total.

Source NaNs are part of the deposited measurement support/mask. They are not file corruption.

Never replace source NaNs with zero, nearest neighbors, interpolation, extrapolation, or synthetic values in raw data.

### 2.3 Geometry

Canonical original STL geometry files are:

- `M_aortic_root.STL`
- `S_aortic_root.STL`
- `L_aortic_root.STL`
- `NS_aortic_root.STL`

All four were acquired from their corresponding Zenodo archives and verified by size, MD5, and SHA-256. The M geometry is the geometry associated with the acquired M velocity subset and is in the velocity-data reference frame.

Reconstructed or approximate geometry may not substitute for the deposited original STL.

### 2.4 Repository phase discrepancy

Zenodo's record description states phases `t000, t003, ..., t039`.

Direct enumeration of the actual deposited **medium archive** gives:

`t000, t003, t006, t009, t012, t015, t018, t021, t024, t027, t030, t033, t036`

`t039` is not present in the enumerated M archive.

Preserve both facts in reporting. Use actual archive enumeration for acquisition accounting. Never invent or reconstruct `t039`.

### 2.5 Historical Pass-1 note

`00_provenance/transfer_manifests/pass1_repository_audit.json` is retained as historical provenance and records an earlier transport limitation. Later verified Zenodo archive-member enumeration and Passes 2–5 supersede that limitation.

For the current acquisition state, the authoritative hierarchy is:

1. root `METADATA.json`;
2. `pass5_integrity_audit.json`;
3. `pass4_velocity_transfer_audit.json`;
4. `acquisition_manifest_v1.json`;
5. `geometry_manifest.json`;
6. stored Zenodo record snapshot and checksum ledgers.

Do not rewrite historical manifests merely to make them appear consistent with later work.

---

## 3. Directory contract

Use project directories only for their defined roles.

- `00_provenance/` — source records, immutable audit history, manifests, checksums, transfer records.
- `01_raw_data/` — immutable byte-for-byte selected source members only.
- `02_metadata_geometry/` — original geometry, coordinate conventions, experimental metadata.
- `03_processed_data/` — reusable transformed/derived arrays.
- `04_analysis/01_laminar_vs_unsteady/` — phase decomposition and regime classification.
- `04_analysis/02_kolmogorov_obukhov/` — spatial spectra, increments, structure functions, scale-to-scale flux.
- `04_analysis/03_taylor_hypothesis/` — Taylor-admissibility analysis only.
- `04_analysis/04_lumley_anisotropy/` — Reynolds-stress anisotropy tensors, invariants, maps.
- `04_analysis/05_canonical_diagnostics/` — vorticity, strain, enstrophy, dissipation-related diagnostics, intermittency, supplementary canonical measures.
- `05_results/` — promoted figures, tables, classification summaries, final numerical results.
- `06_code/` — analysis code, tests, configuration, environment specifications.
- `07_manuscript/` — manuscript text and publication material.

Never write derived results into `01_raw_data`.

---

## 4. Raw-data immutability

`01_raw_data` is read-only scientific evidence.

Forbidden operations on raw data include:

- overwriting;
- renaming canonical source members without an explicit provenance-preserving migration;
- filtering;
- interpolation;
- extrapolation;
- remeshing;
- downsampling;
- smoothing;
- inpainting;
- coordinate transformation;
- unit conversion;
- gradient computation;
- phase averaging;
- ensemble averaging;
- masking that modifies stored source bytes;
- NaN replacement;
- compression that replaces the canonical raw object;
- appending derived columns.

If a transformation is scientifically required, write a new object under `03_processed_data` and record:

- source path(s);
- source SHA-256(s);
- code/version used;
- parameters;
- date/time;
- output SHA-256;
- any loss of support or resolution.

---

## 5. Mandatory startup gate

Before any scientific analysis run:

1. Read root `METADATA.json`.
2. Read this `AGENTS.md`.
3. Read `00_provenance/transfer_manifests/pass5_integrity_audit.json`.
4. Read `00_provenance/transfer_manifests/acquisition_manifest_v1.json`.
5. Read `00_provenance/transfer_manifests/geometry_manifest.json`.
6. Confirm the velocity selection is M/AAo/t018,t021,t024 with 16 realizations per phase.
7. Confirm the required filenames are exactly `B00001.npy`–`B00016.npy` in each selected phase.
8. Confirm each raw velocity file is 47,164,880 bytes unless the authoritative manifest is explicitly superseded.
9. Confirm M geometry resolves to canonical `M_aortic_root.STL`.
10. Confirm no analysis code is about to write into `01_raw_data`.

If any mandatory check fails, abort the scientific run. Write a failure report under the relevant analysis folder and do not continue to scientific inference.

---

## 6. Numerical support and mask rules

Before computing statistics:

- treat coordinate columns and velocity columns separately;
- determine valid finite support for every realization;
- quantify support differences between realizations and phases;
- do not assume all velocity nodes are valid merely because coordinates are finite;
- do not silently fill holes;
- do not allow mask differences to create artificial fluctuations;
- for ensemble quantities, define and record the support rule used at every node/window;
- if a statistic requires common support, use the common finite-support intersection or a clearly documented alternative justified for that statistic;
- preserve original native coordinate values;
- do not assume physical coordinate or velocity units beyond what is explicitly supported by source metadata or verified project metadata.

Any resampling required by a specific method must occur only in processed data and must be sensitivity-tested against the native grid.

---

## 7. Phase-conditioned decomposition

For each selected phase `phi` and spatial location `x`, with `N = 16` independent phase-locked realizations:

`u^(n)(x,phi) = u_bar(x,phi) + u'^(n)(x,phi)`

where

`u_bar(x,phi) = (1/N) * sum_n u^(n)(x,phi)`

and, by construction,

`<u'(x,phi)>_n = 0`.

The phase-conditioned Reynolds-stress tensor is:

`R_ij(x,phi) = <u'_i u'_j>_n`

and turbulent/fluctuation kinetic energy is:

`k(x,phi) = 0.5 * R_ii`.

Do not pool `t018`, `t021`, and `t024` as if they were 48 realizations of one stationary state. They are different phase conditions.

Do not use change in `u_bar` between phases as turbulent fluctuation. That change is deterministic phase evolution unless demonstrated otherwise.

Mandatory outputs of this stage:

- phase mean field;
- phase residual fields;
- Reynolds-stress tensor components;
- `k`;
- realization-convergence diagnostics;
- uncertainty estimates;
- support/mask diagnostics;
- spatial maps showing where ensemble statistics are trustworthy.

---

## 8. Gate A — turbulence versus deterministic unsteadiness

The first scientific decision is whether the ensemble residual supports a genuine stochastic multiscale interpretation.

No single scalar threshold is pre-authorized as a universal turbulence detector.

Evidence must be synthesized from multiple independent diagnostics, including as feasible:

- magnitude and spatial organization of ensemble residuals;
- convergence of first- and second-order ensemble statistics;
- Reynolds stresses and `k`;
- spatial two-point correlations;
- scale-dependent velocity increments;
- increment PDFs;
- spectral content measured spatially;
- evidence for scale-to-scale transfer;
- sensitivity to measurement support, masks, window location, realization resampling, and resolution.

A `-5/3` spectral interval alone is never sufficient evidence of turbulence.

If stochastic multiscale behavior cannot be separated defensibly from deterministic phase evolution, measurement limitations, or insufficient ensemble/statistical resolution, classify the result as `UNRESOLVED_AT_AVAILABLE_RESOLUTION`, not as turbulence by default.

---

## 9. Kolmogorov–Obukhov analysis

KO analysis is performed only after Gate A supports turbulence.

Because the dataset contains direct 3-D spatial velocity fields, primary KO tests are spatial. Taylor's hypothesis is not required for them.

### 9.1 Energy spectrum

For a locally admissible region, the K41 inertial-range baseline is:

`E(kappa) = C_K * epsilon^(2/3) * kappa^(-5/3)`.

Do not compute one naive FFT over the complete irregular/bounded aortic measurement domain and call it a turbulence spectrum.

Use local/windowed methods compatible with finite support and geometry. Record:

- window location and physical/native size;
- support fraction;
- tapering/window function;
- directional versus isotropized treatment;
- wavenumber convention;
- resolution limits;
- fit interval;
- fit uncertainty;
- sensitivity to window choice.

### 9.2 Longitudinal increments

For separation vector `r`:

`delta u_L(r) = [u(x+r) - u(x)] dot r_hat`.

Second-order structure function:

`S_2(r) = <[delta u_L(r)]^2>`.

K41 baseline:

`S_2(r) proportional to (epsilon * r)^(2/3)`.

Third-order longitudinal structure function:

`S_3(r) = <[delta u_L(r)]^3>`.

Kolmogorov four-fifths law under its assumptions:

`S_3(r) = -(4/5) * epsilon * r`.

The four-fifths relation is a stronger test than observing a `-5/3` spectral slope, but it is admissible only where assumptions and resolved scale separation are defensible.

Higher-order absolute structure functions:

`S_p(r) = <|delta u_L(r)|^p> proportional to r^(zeta_p)`.

K41 baseline:

`zeta_p = p/3`.

Extended self-similarity may be tested with:

`S_p(r) proportional to |S_3(r)|^(zeta_p)`,

with the exact exponent convention documented in the run.

### 9.3 Coarse-grained energy flux

For filter scale `ell`:

`tau_ij^(ell) = overline(u_i u_j)^(ell) - ubar_i^(ell) * ubar_j^(ell)`

`Sbar_ij^(ell) = 0.5 * [d ubar_i^(ell)/dx_j + d ubar_j^(ell)/dx_i]`

`Pi_ell = - tau_ij^(ell) * Sbar_ij^(ell)`.

A forward cascade should show a physically consistent scale interval in which the mean transfer is approximately scale-independent within uncertainty, with sign interpreted under the exact convention used.

### 9.4 KO decision rule

KO support requires concordance among multiple resolved diagnostics, not one fitted exponent.

A defensible `KO_TURBULENCE` classification should be supported, where admissible, by mutually consistent evidence from:

- spatial spectrum;
- `S_2`;
- `S_3`;
- higher-order/ESS behavior;
- coarse-grained flux;
- uncertainty and locality tests.

If turbulence is established but KO assumptions or scale separation are inadequate, do not label KO "false"; use `UNRESOLVED_AT_AVAILABLE_RESOLUTION` for the KO question.

Use `NON_KO_TURBULENCE` only when turbulence itself is established and KO predictions fail systematically in diagnostics that are sufficiently resolved and scientifically admissible.

---

## 10. Taylor-hypothesis analysis

Taylor analysis is downstream of the turbulence classification.

A necessary small-fluctuation metric may be expressed as:

`chi = u'_rms / |U|`.

Classical frozen-advection form:

`partial u'/partial t + (U_c dot grad) u' approximately 0`.

A true space-time correlation test would use:

`R(r,tau) = <u'(x,t) dot u'(x+r,t+tau)> / sqrt(<|u'(x,t)|^2><|u'(x+r,t+tau)|^2>)`

and look for a ridge near:

`r approximately U_c * tau`.

The acquired data are independent phase-locked realizations at three phases, not continuous turbulent trajectories.

Therefore:

- do not construct a turbulent time series by concatenating realizations;
- do not treat `t018`, `t021`, and `t024` as a high-frequency turbulent trajectory;
- do not convert frequency spectra to wavenumber spectra with `kappa = 2*pi*f/U` as if Taylor were established;
- do not claim a full temporal Taylor test from this subset.

Permitted Taylor work is limited to necessary/local admissibility diagnostics and clearly labeled phase-mean evolution tests unless additional continuous temporal data are explicitly acquired later.

Taylor failure does not imply absence of turbulence and does not invalidate direct spatial KO tests.

---

## 11. Lumley anisotropy analysis

From phase-conditioned Reynolds stresses:

`k = 0.5 * R_ii`

and the normalized anisotropy tensor is:

`b_ij = R_ij/(2*k) - (1/3)*delta_ij`.

By construction:

`b_ii = 0`.

Use the invariant convention:

`II_b = -0.5 * b_ij * b_ji`

`III_b = (1/3) * b_ij * b_jk * b_ki`.

Let anisotropy eigenvalues satisfy:

`lambda_1 + lambda_2 + lambda_3 = 0`.

Use eigenvalues/invariants to assess realizability and locate the state in an appropriate Lumley invariant or barycentric representation.

Always report the invariant/sign convention used. Do not mix conventions between figures, code, and manuscript text.

Anisotropy is a characterization axis independent of KO classification. Strongly anisotropic turbulence may still be genuine turbulence.

---

## 12. Canonical diagnostics

Where spatial resolution, masks, units, and derivative accuracy permit:

Vorticity:

`omega = curl(u)`.

Strain-rate tensor:

`S_ij = 0.5 * [du_i/dx_j + du_j/dx_i]`.

Enstrophy density:

`Omega = 0.5 * |omega|^2`.

For fluctuating strain and known/verified kinematic viscosity:

`epsilon = 2 * nu * <S'_ij S'_ij>`.

Do not report a physical dissipation rate if `nu`, derivative accuracy, or small-scale resolution is not adequately supported. In that case report the resolved surrogate explicitly as a surrogate.

Longitudinal-increment flatness:

`F(r) = <[delta u_L(r)]^4> / <[delta u_L(r)]^2>^2`.

A Gaussian reference has `F = 3`; scale-dependent departures may characterize intermittency but do not by themselves prove a cascade.

Possible supplementary diagnostics include:

- divergence/error checks;
- Q/R or velocity-gradient topology when derivative quality permits;
- directional correlations;
- integral length scales;
- local Reynolds-number-like measures only when required dimensional inputs are verified;
- resolved coherent-structure indicators.

Every derivative-based result must include a resolution/sensitivity statement.

---

## 13. Locality and bounded-domain rule

The aortic-root/ascending-aorta field is bounded, inhomogeneous, anisotropic, and geometry-dependent.

Therefore:

- prefer local/windowed statistics;
- test window-location sensitivity;
- distinguish streamwise, cross-stream, and isotropized statistics where meaningful;
- never assume global homogeneity;
- never assume global isotropy;
- never infer an inertial range solely from visual slope similarity;
- identify geometric boundaries and invalid-support regions before spatial differencing;
- do not allow a boundary/mask discontinuity to masquerade as a physical small-scale gradient.

If a canonical turbulence identity requires assumptions that cannot be defended locally, mark that test `INADMISSIBLE` for that region rather than forcing a numerical answer.

---

## 14. Uncertainty and finite-ensemble requirements

There are 16 independent phase-locked realizations per selected phase.

Every ensemble statistic must acknowledge finite-ensemble uncertainty.

Use realization-level resampling/bootstrap or another documented finite-sample method appropriate to the statistic.

At minimum, report:

- estimator definition;
- number of realizations used;
- support rule;
- confidence/uncertainty interval or standard error where meaningful;
- sensitivity to leaving out realizations;
- convergence trend with ensemble size where feasible.

For spectra, structure functions, fluxes, anisotropy invariants, and fitted exponents, propagate uncertainty far enough to prevent false precision.

Do not increase apparent sample size by treating spatial nodes as independent realizations without a defensible dependence model.

Do not combine phases to inflate `N`.

---

## 15. Reproducible run structure

Each substantial analysis execution must create a uniquely identified run, preferably:

`run_YYYYMMDD_HHMMSS`

inside the relevant `04_analysis/<branch>/` folder.

Each run should contain, as applicable:

- `run_manifest.json`
- `config.json`
- `input_hashes.txt`
- `methods.md`
- `metrics.csv` or another transparent tabular format
- `uncertainty.csv`
- `diagnostics.json`
- `figures/`
- `logs/`

The run manifest must record:

- run ID;
- UTC timestamp;
- scientific branch;
- input file paths;
- input SHA-256 values;
- geometry SHA-256;
- code version or commit identifier;
- environment/package versions;
- random seed(s);
- mask/support policy;
- numerical parameters;
- spatial windows;
- filters/tapers if used;
- output file hashes;
- warnings and failed gates.

Use deterministic random seeds unless a method explicitly requires otherwise; record them.

Promote only reviewed/final outputs from analysis runs into `05_results`.

---

## 16. Code requirements

Code belongs in `06_code`.

Requirements:

- Python is the default analysis language unless another language is explicitly justified.
- Use vectorized/chunked computation appropriate to field size.
- Never modify raw files in place.
- Unit-test critical tensor, gradient, spectral, and structure-function routines.
- Include synthetic sanity tests where they can expose sign, normalization, tensor-index, or FFT-convention errors.
- Make coordinate ordering explicit.
- Make NaN/mask handling explicit.
- Check dimensional consistency when units are known.
- Record all Fourier and wavenumber normalization conventions.
- Record all tensor invariant conventions.
- Separate data loading, validation, processing, diagnostics, plotting, and classification logic.
- A plotting function must not silently perform scientific transformations that are absent from the numerical pipeline.
- Warnings about insufficient support/resolution must propagate to final classification logic.

Never hard-code a desired turbulence classification.

---

## 17. Prohibited scientific shortcuts

The following are forbidden:

1. Declaring turbulence because a spectrum appears close to `-5/3`.
2. Declaring KO turbulence from a single statistic.
3. Using a whole-domain FFT as the primary spectrum without addressing boundaries, masks, and inhomogeneity.
4. Assuming Taylor's hypothesis before testing its admissibility.
5. Converting frequency to wavenumber using Taylor when no valid temporal turbulent trajectory exists.
6. Treating the three selected phases as a continuous turbulent time series.
7. Pooling phases to create `N = 48`.
8. Treating source NaNs as zeros.
9. Interpolating raw NaNs and then presenting the result as measured velocity.
10. Using reconstructed geometry in place of the canonical deposited STL.
11. Mixing M velocity with S, L, or NS geometry.
12. Claiming isotropy or homogeneity globally.
13. Reporting physical dissipation if viscosity/units/resolution are unsupported.
14. Hiding an inadmissible canonical test by substituting an easier proxy without labeling it.
15. Treating absence of KO behavior as absence of turbulence.
16. Treating Taylor failure as absence of turbulence.
17. Treating a non-significant result as proof of laminarity when statistical power is inadequate.
18. Silently acquiring or introducing external data into the canonical raw subset.
19. Editing `METADATA.json`, `AGENTS.md`, or provenance manifests merely to make results fit an analysis.
20. Deleting failed runs; failed runs are part of reproducibility unless explicitly designated as disposable scratch work.

---

## 18. Classification gate

The final classification must be evidence-driven.

### `LAMINAR_UNSTEADY`

Use only when the data support deterministic phase-dependent flow with no defensible stochastic multiscale turbulent component at the resolved scales, with uncertainty sufficient to make that conclusion meaningful.

### `KO_TURBULENCE`

Use only when turbulence is independently established and multiple admissible KO diagnostics are mutually consistent over resolved scales within uncertainty.

### `NON_KO_TURBULENCE`

Use only when turbulence is independently established, resolution/statistical power are adequate for the tests applied, and admissible KO predictions fail systematically rather than merely being unresolvable.

### `UNRESOLVED_AT_AVAILABLE_RESOLUTION`

Use when the available spatial resolution, finite ensemble, masks, scale separation, dimensional metadata, or statistical uncertainty prevent a defensible classification.

This is a valid scientific result and must not be converted into a stronger claim for narrative convenience.

---

## 19. Required synthesis order

The final scientific report must preserve this causal order:

`phase decomposition`
→ `turbulence versus deterministic unsteadiness`
→ `KO admissibility and tests`
→ `Taylor admissibility`
→ `Lumley anisotropy`
→ `canonical diagnostics`
→ `integrated classification`

Do not reorganize results so that a downstream assumption appears to justify an upstream classification.

---

## 20. Manuscript and claim discipline

Every manuscript-level claim must be traceable to:

- a verified raw input;
- a reproducible analysis run;
- a recorded numerical method;
- an uncertainty statement where relevant;
- a figure/table/result in `05_results`.

Use language such as:

- "supported within the resolved spatial range";
- "not resolved by the available measurements";
- "inadmissible under the required assumptions";
- "consistent with";
- "inconsistent with under the tested conditions";

when stronger causal statements are not warranted.

Do not use visual agreement alone as quantitative evidence.

---

## 21. Project metadata that must remain explicit

Always preserve these facts in analysis documentation:

- Zenodo record: 1163562;
- DOI: `10.5281/zenodo.1163562`;
- source version: 1.0;
- license: CC BY 4.0;
- creators: David Hasler and Dominik Obrist;
- method: Tomo-PIV, 3D-3C;
- heart rate reported by Zenodo: 72 bpm;
- cardiac output reported by Zenodo description: 4.8 L (interpreted project metadata: L/min);
- selected velocity geometry/domain: M/AAo;
- selected phases: 0.18, 0.21, 0.24 s;
- 16 phase-locked independent realizations per selected phase;
- actual enumerated M phases end at `t036`;
- source record description states `t039`, creating a documented discrepancy;
- raw selected velocity bytes and geometry files are integrity-verified.

Do not silently restore the earlier stale 4.9 L/min value.

---

## 22. Provenance and integrity

Canonical checksums live under:

`00_provenance/checksums/`

Canonical transfer/acquisition audits live under:

`00_provenance/transfer_manifests/`

Before publication-grade computation, verify input hashes against the current final checksum ledger.

If a raw-file hash changes:

- stop;
- do not overwrite provenance;
- quarantine the changed object;
- report the mismatch;
- require explicit human resolution before continuing.

No scientific run may "repair" a canonical raw file.

---

## 23. Expansion of the dataset

Future acquisition of additional phases, SOV data, or S/L/NS velocity fields requires a new explicit acquisition pass or project-level instruction.

When expanding:

- preserve current files and hashes;
- create a new acquisition manifest version;
- identify exactly what was added;
- verify source hashes;
- update root metadata;
- update this contract only if the scientific program changes.

Do not silently redefine the current three-phase kernel.

---

## 24. Agent stop conditions

Stop analysis and report rather than improvising when:

- a canonical file is missing or has a mismatched hash;
- coordinate/velocity semantics are ambiguous;
- units required for a claimed physical quantity are unverified;
- support/mask behavior materially changes a statistic and no defensible policy is available;
- a requested canonical law is inadmissible under local assumptions;
- a fitted scale range is shorter than required for a defensible claim;
- uncertainty spans competing classifications;
- a future instruction conflicts with raw-data immutability or the locked scientific sequence.

A transparent `UNRESOLVED` or `INADMISSIBLE` result is preferable to an unjustified positive finding.

---

## 25. Pass-6 lock

This `AGENTS.md` is the Pass-6 operating specification.

It does not itself perform the turbulence analysis. It freezes how the analysis must be performed.

After Pass 6 is verified:

- Passes 2–5 remain immutable acquisition/provenance history;
- the final root metadata records Pass 6 as complete;
- future work begins with analysis implementation, not acquisition redesign, unless the user explicitly reopens acquisition scope.

**End of canonical operating contract.**
