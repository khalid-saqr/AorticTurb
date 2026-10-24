import numpy as np

from aorticturb.validation.masks import common_support


def test_common_support_intersection_preserves_missing_support():
    fields = np.ones((2, 3, 3), dtype=float)
    fields[0, 1, 0] = np.nan
    mask = common_support(fields, policy="intersection")
    assert mask.tolist() == [True, False, True]
