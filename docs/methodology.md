# AorticTurb methodology

`AGENTS.md` is the canonical scientific contract. This document explains how the code
implements it.

## 1. Phase decomposition

Each selected phase is analyzed independently across 16 realizations. The code computes
the phase mean, residuals, Reynolds stresses and `k` on the common finite-velocity
intersection. No phase pooling is allowed. Bootstrap uncertainty is realization-based.

## 2. Gate A

Gate A asks whether fixed-phase residuals support resolved stochastic multiscale dynamics.
It never uses an apparent `-5/3` slope. Automatic turbulence establishment requires:

- adequate ensemble/common support;
- positive short-separation residual correlation;
- statistically supported correlation decay with separation; and
- positive forward coarse-grained energy transfer across all screening scales.

Flux constancy is not tested until the KO stage. Failure to meet Gate A defaults to
unresolved. `LAMINAR_UNSTEADY` is automatically supported only when an external
measurement-noise RMS ceiling is configured and residual uncertainty lies below it while
forward transfer is absent.

## 3. Kolmogorov–Obukhov

Only Gate-A-positive phases enter KO testing. AorticTurb selects local fully finite
rectangular windows and performs direct spatial analysis. Directional spectra and
longitudinal structure functions are retained for x/y/z; an isotropized spectrum is
secondary. Scaling ranges are selected by power-law linearity and interval length without
using a target exponent.

Core evidence is:

- directional spectral slope versus `-5/3`;
- directional S2 slope versus `2/3`;
- S3 slope versus 1 plus the four-fifths amplitude ratio using independently estimated
  positive coarse-grained flux as epsilon; and
- scale independence and positive sign of mean coarse-grained flux.

Target consistency is decided by whether the theoretical value lies inside the
realization-bootstrap interval. KO is supported only when all core diagnostics are
resolved and concordant. `NON_KO_TURBULENCE` requires at least two resolved core failures.
Otherwise the KO question remains unresolved.

## 4. Taylor hypothesis

The dataset does not contain continuous turbulent trajectories. AorticTurb therefore
reports the fluctuation-to-mean ratio and mean-flow spatial descriptors only. It always
marks the true temporal Taylor test unavailable and never converts frequency to
wavenumber using Taylor.

## 5. Anisotropy

AorticTurb uses

`b_ij = R_ij/(2k) - delta_ij/3`

with

`II_b = -0.5 b_ij b_ji`

and

`III_b = (1/3) b_ij b_jk b_ki`.

Descending eigenvalues are converted to barycentric C1/C2/C3 weights and checked for
realizability. These diagnostics characterize turbulence; they do not override upstream
classification gates.

## 6. Canonical diagnostics

Local fully finite windows are used for velocity gradients, strain, vorticity, enstrophy,
divergence quality and `2 S'_ij S'_ij`. The latter is a dissipation surrogate unless
kinematic viscosity is supplied. Scale-dependent longitudinal-increment flatness is
reported against the Gaussian reference value 3.

## 7. Classification

Primary labels are phase-specific. If phases differ, AorticTurb reports the difference
rather than inventing a fifth overall class. Downstream characterization never creates an
upstream turbulence decision.
