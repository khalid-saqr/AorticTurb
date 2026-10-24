# AorticTurb

**A reproducible framework for turbulence classification and canonical turbulence diagnostics from tomographic PIV.**

AorticTurb is the study codebase for the curated Hasler–Obrist three-dimensional,
three-component tomographic-PIV dataset downstream of a bioprosthetic aortic valve.
The scientific workflow is governed by [`AGENTS.md`](AGENTS.md). The software does not
assume turbulence, Kolmogorov–Obukhov scaling, isotropy, homogeneity, or Taylor's
frozen-turbulence hypothesis in advance.

## Scientific sequence

1. Phase-conditioned ensemble decomposition.
2. Gate A: turbulence versus deterministic unsteadiness.
3. Kolmogorov–Obukhov admissibility and direct spatial tests.
4. Taylor-hypothesis admissibility.
5. Lumley/barycentric Reynolds-stress anisotropy.
6. Canonical derivative and intermittency diagnostics.
7. Integrated phase-specific classification.

The only primary terminal classifications are:

- `LAMINAR_UNSTEADY`
- `KO_TURBULENCE`
- `NON_KO_TURBULENCE`
- `UNRESOLVED_AT_AVAILABLE_RESOLUTION`

AorticTurb deliberately allows `UNRESOLVED_AT_AVAILABLE_RESOLUTION` when assumptions,
scale separation, measurement support, or the 16-realization ensemble do not justify a
stronger conclusion.

## Dataset

The analysis dataset will be released separately on Zenodo as a compressed,
provenance-preserving package of the finalized project tree. Until that curated record is
published, `dataset_manifest.json` intentionally contains placeholder Zenodo fields.

Original source:

- Zenodo: `10.5281/zenodo.1163562`
- Article: `10.1371/journal.pone.0194384`
- Measurement: 3D–3C tomographic PIV
- Selected velocity subset: M geometry / AAo / `t018`, `t021`, `t024`
- 16 independent phase-locked realizations per phase
- 48 velocity fields plus four canonical STL geometries

The future archive preserves `METADATA.json`, `AGENTS.md`, `00_provenance`,
`01_raw_data`, and `02_metadata_geometry`, plus a package-level `SHA256SUMS.txt`.

## Installation

```bash
python -m pip install -e .
```

For development:

```bash
python -m pip install -e ".[dev]"
pytest
```

## First step: obtain the input dataset

After the curated Zenodo record is published and entered in `dataset_manifest.json`:

```bash
aorticturb dataset fetch --output data/hasler_obrist_tomopiv
aorticturb dataset verify --root data/hasler_obrist_tomopiv
aorticturb dataset inspect --root data/hasler_obrist_tomopiv
```

`dataset fetch` resolves Zenodo, downloads the archive, verifies its archive checksum when
configured, rejects unsafe ZIP paths, extracts the package, checks every package checksum,
and runs structural/content validation. A local extracted dataset can be verified before
the new Zenodo record exists.

## Complete study run

```bash
aorticturb run-all \
  --root data/hasler_obrist_tomopiv \
  --output outputs
```

`run-all` executes the locked sequence. Downstream phases are skipped automatically when
Gate A does not establish turbulence; the code never forces a downstream diagnosis to
rescue an unresolved upstream result.

Each stage produces a timestamped run bundle containing, as applicable:

- `run_manifest.json`
- `config.json`
- `input_hashes.txt`
- `methods.md`
- `metrics.csv`
- `uncertainty.csv`
- `diagnostics.json`
- `figures/`
- `processed/`
- `logs/run.log`
- output SHA-256 values in the finalized run manifest

A study-level `study_summary.json` records the exact stage-run paths and final
phase-specific classifications.

## Individual stages

The decomposition stage starts directly from raw data:

```bash
aorticturb decompose --root DATASET --output outputs
```

Downstream standalone stages require explicit upstream run directories, preventing silent
reuse of an unrelated analysis:

```bash
aorticturb regime --root DATASET --decomposition DECOMP_RUN --output outputs

aorticturb kolmogorov --root DATASET \
  --decomposition DECOMP_RUN --regime REGIME_RUN --output outputs

aorticturb taylor --root DATASET \
  --decomposition DECOMP_RUN --regime REGIME_RUN --output outputs

aorticturb anisotropy --root DATASET \
  --decomposition DECOMP_RUN --regime REGIME_RUN --output outputs

aorticturb diagnostics --root DATASET \
  --decomposition DECOMP_RUN --regime REGIME_RUN --output outputs

aorticturb classify --root DATASET \
  --regime REGIME_RUN --kolmogorov KO_RUN --output outputs
```

## Methodological safeguards

- Raw data are immutable.
- Source NaNs are preserved and never zero-filled.
- The three phases are never pooled into `N=48`.
- Bootstrap resampling is across realizations, not spatial nodes.
- Gate A does not use a `-5/3` slope.
- Automatic turbulence establishment requires coherent residual structure and resolved
  forward scale transfer.
- Laminarity is not inferred from failure to establish turbulence; an external
  measurement-noise ceiling is required for that automatic conclusion.
- Spectra and derivatives use local fully finite rectangular windows.
- Whole-domain naive FFTs are prohibited.
- KO scaling intervals are selected without reference to their target exponents.
- KO support requires concordance among directional spectra, S2, S3/4-5-law, and
  coarse-grained flux.
- Taylor's full temporal test is explicitly unavailable for these independent
  phase-locked realizations, and frequency-to-wavenumber conversion is prohibited.
- Lumley invariant sign conventions are fixed and reported.
- Dissipation is reported as a resolved surrogate unless viscosity is supplied, and even
  then no claim is made that all dissipative scales are resolved.

## Configuration

`configs/default.yaml` contains numerical execution defaults. These values control window
sizes, bootstrap count, and operational diagnostics; they are not hidden physical truth
thresholds. Custom configurations are deep-merged with the packaged defaults, while the
canonical three phases and 16-realization design remain locked.

## Repository layout

```text
src/aorticturb/      scientific engine and CLI
tests/               numerical, scientific-contract, packaging and CLI tests
configs/             reproducible run configuration
docs/                methodology, dataset and output documentation
dataset_manifest.json
AGENTS.md
```

## License and citation

AorticTurb software is BSD-3-Clause licensed. The experimental data are distributed
separately under their deposition license. Cite the original Hasler–Obrist dataset and
article, and cite AorticTurb using `CITATION.cff` when the software is used in published
work.
