# Methodology

The governing scientific specification is `AGENTS.md`.

AorticTurb implements one scientific engine shared by the CLI and any future notebook
interface. No notebook or wrapper should independently reimplement the equations.

## Phase-conditioned decomposition

For each phase and realization:

`u^(n) = u_bar + u'^(n)`

with the ensemble mean computed across the 16 realizations at the same phase only.

The Reynolds-stress tensor is

`R_ij = <u'_i u'_j>`

and

`k = 0.5 * R_ii`.

## Regime gate

Turbulence is never declared from one threshold or one `-5/3` fit. The regime layer
collects ensemble convergence, fluctuation magnitude, spatial correlation and
scale-dependent evidence and records uncertainty. If the available data cannot
separate stochastic multiscale motion from deterministic phase evolution or
measurement limitations, the result is `UNRESOLVED_AT_AVAILABLE_RESOLUTION`.

## Kolmogorov–Obukhov

The primary tests are spatial. Taylor's hypothesis is not required to compute spatial
spectra or structure functions from 3-D snapshots.

AorticTurb provides local/windowed spectra, longitudinal structure functions,
higher-order exponents, and coarse-grained flux primitives. Scientific classification
requires concordant evidence across admissible diagnostics.

## Taylor hypothesis

Only necessary/local admissibility diagnostics are permitted for this three-phase
subset. The software never concatenates phases or realizations into a turbulent
time series and never performs frequency-to-wavenumber conversion by default.

## Anisotropy

`b_ij = R_ij/(2k) - delta_ij/3`

with

`II_b = -0.5 b_ij b_ji`

`III_b = (1/3) b_ij b_jk b_ki`.

The sign convention is fixed and tested.

## Uncertainty

The finite ensemble has N=16 at each phase. Bootstrap resampling is performed at the
realization level, not by treating spatial nodes as independent observations.
