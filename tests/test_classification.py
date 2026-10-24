from aorticturb.classification.decision import classify


def test_classification_contract():
    assert classify(
        turbulence_established=False, ko_resolved=None, ko_supported=None
    ).label == "LAMINAR_UNSTEADY"
    assert classify(
        turbulence_established=True, ko_resolved=True, ko_supported=True
    ).label == "KO_TURBULENCE"
    assert classify(
        turbulence_established=True, ko_resolved=True, ko_supported=False
    ).label == "NON_KO_TURBULENCE"
    assert classify(
        turbulence_established=True, ko_resolved=False, ko_supported=None
    ).label == "UNRESOLVED_AT_AVAILABLE_RESOLUTION"
