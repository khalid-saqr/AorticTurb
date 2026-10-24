# Pass 3 Colab execution

`notebooks/AorticTurb_Colab.ipynb` is a thin, restart-safe execution driver for the public AorticTurb package.

The notebook was rebuilt from scratch. It does not clone the repository into a working tree, change the process working directory, manipulate `sys.modules`, copy the canonical 2.2-GB dataset into scratch storage, or depend on a notebook-specific analysis runner.

Execution model:

1. create an isolated virtual environment under `/content`;
2. install one immutable public Git commit using a PEP-508 Git reference;
3. verify the installed commit from PEP-610 `direct_url.json` metadata;
4. mount Google Drive and use the canonical project folder directly as the dataset root;
5. run the mandatory startup gate;
6. run fresh phase-conditioned decomposition;
7. run Gate A using the unchanged primary implementation;
8. run the locked support-policy sensitivity analysis;
9. validate run manifests, support columns, phases, realization count, and comparator names;
10. write a compact Pass-3 execution record under `results/colab_runs/`.

Every scientific stage is invoked through the installed `aorticturb` CLI in a fresh subprocess. The notebook kernel never imports `aorticturb`, so prior kernel imports cannot change the code executed by the scientific stages.

The installed package commit is recorded automatically in each run manifest from PEP-610 VCS installation metadata, with the repository checkout used only as a fallback for developer/local execution.

The notebook does not call `run-all`, Kolmogorov–Obukhov, Taylor, Lumley anisotropy, canonical diagnostics, or integrated classification. It writes analysis outputs under the project `results/` workspace and does not promote anything into the root canonical `05_results` folder.
