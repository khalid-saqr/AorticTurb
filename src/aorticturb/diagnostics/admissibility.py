from __future__ import annotations


def physical_dissipation_admissibility(config: dict) -> dict:
    physical = config.get("physical", {})
    diagnostics = config.get("diagnostics", {})
    nu = physical.get("kinematic_viscosity")
    nu_verified = bool(physical.get("kinematic_viscosity_verified", False))
    small_scale_verified = bool(
        diagnostics.get("small_scale_resolution_verified", False)
    )
    if nu is None:
        return {
            "admissible": False,
            "status": "INADMISSIBLE_MISSING_VERIFIED_VISCOSITY",
            "kinematic_viscosity": None,
        }
    if not nu_verified:
        return {
            "admissible": False,
            "status": "INADMISSIBLE_VISCOSITY_NOT_VERIFIED",
            "kinematic_viscosity": float(nu),
        }
    if not small_scale_verified:
        return {
            "admissible": False,
            "status": "INADMISSIBLE_SMALL_SCALE_RESOLUTION_NOT_VERIFIED",
            "kinematic_viscosity": float(nu),
        }
    return {
        "admissible": True,
        "status": "ADMISSIBLE_RESOLVED_DISSIPATION_WITH_VERIFIED_NU",
        "kinematic_viscosity": float(nu),
    }
