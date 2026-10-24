# AorticTurb

**A reproducible framework for turbulence classification and canonical turbulence diagnostics from tomographic PIV.**

AorticTurb is a Python package and command-line interface for the curated Hasler–Obrist
three-dimensional, three-component tomographic-PIV dataset downstream of a
bioprosthetic aortic valve.

The scientific workflow is governed by [`AGENTS.md`](AGENTS.md). The software does
not assume turbulence, Kolmogorov–Obukhov scaling, isotropy, homogeneity, or
Taylor's frozen-turbulence hypothesis in advance.

## Scientific sequence

1. Phase-conditioned ensemble decomposition.
2. Turbulence versus deterministic unsteadiness.
3. Kolmogorov–Obukhov admissibility and spatial tests.
4. Taylor-hypothesis admissibility.
5. Lumley Reynolds-stress anisotropy.
6. Canonical turbulence diagnostics.
7. Integrated classification.

The only primary terminal classifications are:

- `LAMINAR_UNSTEADY`
- `KO_TURBULENCE`
- `NON_KO_TURBULENCE`
- `UNRESOLVED_AT_AVAILABLE_RESOLUTION`

## Dataset

The analysis dataset will be released separately on Zenodo as a compressed,
provenance-preserving package of the finalized project tree.

Until that curated Zenodo record is published, `dataset_manifest.json` intentionally
contains a placeholder record ID and archive checksum.

Original source:

- Zenodo: `10.5281/zenodo.1163562`
- Article: `10.1371/journal.pone.0194384`
- Measurement: 3D–3C tomographic PIV
- Selected velocity subset: M geometry / AAo / `t018`, `t021`, `t024`
- 16 independent phase-locked realizations per phase

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

When the curated Zenodo record has been published and its ID inserted into
`dataset_manifest.json`:

```bash
aorticturb dataset fetch --output data/hasler_obrist_tomopiv
aorticturb dataset verify --root data/hasler_obrist_tomopiv
aorticturb dataset inspect --root data/hasler_obrist_tomopiv
```

The fetch command resolves the Zenodo record, downloads the archive, verifies the
archive SHA-256 when provided, extracts it, then executes the dataset verification
gate. The archive preserves `00_provenance`, `01_raw_data`, and
`02_metadata_geometry`; it is not reorganized into a second data layout.

Before the Zenodo record is published, a local extracted dataset can be verified
directly:

```bash
aorticturb dataset verify --root /path/to/extracted/dataset
```

## Analysis CLI

```bash
aorticturb decompose --root data/hasler_obrist_tomopiv
aorticturb regime --root data/hasler_obrist_tomopiv
aorticturb kolmogorov --root data/hasler_obrist_tomopiv
aorticturb taylor --root data/hasler_obrist_tomopiv
aorticturb anisotropy --root data/hasler_obrist_tomopiv
aorticturb diagnostics --root data/hasler_obrist_tomopiv
aorticturb classify --root data/hasler_obrist_tomopiv
aorticturb run-all --root data/hasler_obrist_tomopiv
```

Each substantial execution produces a timestamped run directory with configuration,
input hashes, metrics, diagnostics, logs, and output hashes.

## Important methodological constraints

- The raw dataset is immutable.
- Source NaNs are preserved and are never replaced with zeros.
- The three selected phases are never pooled as 48 samples of one stationary state.
- A `-5/3` spectral slope alone cannot establish turbulence.
- Whole-domain naive FFTs are not used as the primary turbulence spectrum.
- Taylor's hypothesis is downstream of the turbulence decision.
- The three phase-locked snapshots are not a turbulent time series.
- Uncertainty from only 16 realizations per phase is propagated explicitly.
- An inadmissible diagnostic is reported as inadmissible rather than forced.

## Repository layout

```text
src/aorticturb/      scientific engine and CLI
tests/               numerical and contract tests
configs/             reproducible run configuration
docs/                methodology and dataset documentation
dataset_manifest.json
AGENTS.md
```

## License

The AorticTurb software is distributed under the BSD 3-Clause License.

The experimental data are distributed separately under the license associated with
their Zenodo deposition. Cite the original Hasler–Obrist dataset and article when
using the data.
