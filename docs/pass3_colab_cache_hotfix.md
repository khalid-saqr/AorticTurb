# Pass 3 Colab cache hotfix

This hotfix addresses Colab kernel reuse after the Pass-3 integration merge.

A Colab runtime can retain an already-imported older `aorticturb` module in `sys.modules`. Reinstalling the same package version (`0.2.0`) does not guarantee that Python evicts those cached modules, so a rerun can execute stale analysis code even when the detached Git checkout contains the newer Pass-1/Pass-2 implementation.

The notebook now force-reinstalls the pinned checkout, purges cached `aorticturb` modules, prepends the pinned checkout's `src` directory to `sys.path`, invalidates import caches, and verifies that `aorticturb.__file__` resolves inside the pinned checkout before any scientific analysis starts.

No scientific criteria, support thresholds, Gate-A logic, sensitivity policies, or downstream analysis logic are changed.
