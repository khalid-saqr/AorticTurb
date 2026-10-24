from __future__ import annotations

PHASES = ("t018", "t021", "t024")
PHASE_SECONDS = {"t018": 0.18, "t021": 0.21, "t024": 0.24}
REALIZATIONS = tuple(f"B{i:05d}.npy" for i in range(1, 17))
VELOCITY_COLUMNS = ("X", "Y", "Z", "U", "V", "W")
BYTES_PER_VELOCITY_FILE = 47_164_880
ALLOWED_CLASSIFICATIONS = (
    "LAMINAR_UNSTEADY",
    "KO_TURBULENCE",
    "NON_KO_TURBULENCE",
    "UNRESOLVED_AT_AVAILABLE_RESOLUTION",
)
