from aorticturb.classification.decision import classify


def test_resolved_flag_without_ko_decision_stays_unresolved():
    result = classify(turbulence_established=True, ko_resolved=True, ko_supported=None)
    assert result.label == "UNRESOLVED_AT_AVAILABLE_RESOLUTION"
