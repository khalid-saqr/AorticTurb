# Dataset contract

AorticTurb targets a compressed, provenance-preserving package of the finalized
Hasler–Obrist Tomo-PIV project tree. The future Zenodo archive should preserve the same
scientific layout used during acquisition rather than reorganize raw data.

## Canonical selected velocity subset

- geometry: M
- domain: AAo
- phases: `t018`, `t021`, `t024`
- physical phase labels: 0.18 s, 0.21 s, 0.24 s
- realizations: `B00001.npy` through `B00016.npy` at every phase
- total velocity fields: 48
- bytes per velocity field: 47,164,880

The deposited arrays use columns `[X, Y, Z, U, V, W]`. Source NaNs in velocity
components are measurement-support information and must be preserved.

## Required archive layout

At minimum the archive must contain:

```text
METADATA.json
AGENTS.md
SHA256SUMS.txt

00_provenance/
  checksums/
    final_acquired_SHA256SUMS.txt
  transfer_manifests/
    pass5_integrity_audit.json
    acquisition_manifest_v1.json
    geometry_manifest.json
  ...

01_raw_data/
  M_medium_root/
    AAo/
      t018/B00001.npy ... B00016.npy
      t021/B00001.npy ... B00016.npy
      t024/B00001.npy ... B00016.npy

02_metadata_geometry/
  M_medium_root/original_geometry/M_aortic_root.STL
  S_small_root/original_geometry/S_aortic_root.STL
  L_large_root/original_geometry/L_aortic_root.STL
  NS_non_sinus_root/original_geometry/NS_aortic_root.STL
```

`SHA256SUMS.txt` is the package checksum ledger for the exact archive-relative paths.
AorticTurb also requires the historical Pass-5 `final_acquired_SHA256SUMS.txt` ledger and
cross-checks the package hashes against it. The historical ledger retains older geometry
relative paths, so geometry provenance is resolved by canonical STL filename and hash;
velocity paths are matched exactly. The historical ledger is never rewritten.

## Phase catalogue discrepancy

The original Zenodo record description states phases through t039. Direct enumeration
of the deposited medium archive gives t000 through t036 in 0.03 s increments. AorticTurb
uses the actual archive enumeration for acquisition accounting and never reconstructs
t039.

The curated Zenodo record ID, DOI, archive filename, and archive SHA-256 remain
placeholders until the archive is published.
