import numpy as np

from aorticturb.taylor.admissibility import assess


def test_taylor_never_claims_temporal_test_available():
    mean = np.ones((20, 3), dtype=float)
    residuals = np.zeros((16, 20, 3), dtype=float)
    result = assess(mean, residuals)
    assert result.temporal_test_available is False
    assert "NECESSARY_SPATIAL_ADMISSIBILITY_ONLY" in result.conclusion
