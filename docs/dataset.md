# Dataset contract

AorticTurb targets the curated analysis subset of the Hasler–Obrist tomographic-PIV
deposition.

## Canonical selected velocity subset

- Geometry: M
- Domain: AAo
- Phases: t018, t021, t024
- Physical phase labels: 0.18 s, 0.21 s, 0.24 s
- Realizations per phase: 16
- Files per phase: B00001.npy through B00016.npy
- Total velocity files: 48
- Expected bytes per velocity file: 47,164,880

The deposited arrays use columns `[X, Y, Z, U, V, W]`. Source NaNs in velocity
components are measurement-support information and must be preserved.

## Phase catalogue discrepancy

The original Zenodo record description states phases through t039. Direct enumeration
of the deposited medium archive gives t000 through t036 in 0.03 s increments. AorticTurb
uses the actual archive enumeration for acquisition accounting and never reconstructs
t039.

## Curated archive

The curated Zenodo archive is expected to contain:

```text
METADATA.json
AGENTS.md
SHA256SUMS.txt
raw/M_medium_root/AAo/t018/B00001.npy ... B00016.npy
raw/M_medium_root/AAo/t021/B00001.npy ... B00016.npy
raw/M_medium_root/AAo/t024/B00001.npy ... B00016.npy
geometry/M_aortic_root.STL
geometry/S_aortic_root.STL
geometry/L_aortic_root.STL
geometry/NS_aortic_root.STL
```

The curated Zenodo record ID, DOI, archive filename, and archive SHA-256 remain
placeholders until the archive is published.
