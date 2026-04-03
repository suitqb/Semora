"""Single source of truth for scored field names and GT column mapping."""
from __future__ import annotations

PERSON_FIELDS: list[str] = [
    "atomic_action",
    "simple_context",
    "communicative",
    "transporting",
    "age",
]

VEHICLE_FIELDS: list[str] = [
    "motion_status",
    "trunk_open",
    "doors_open",
]

GT_FIELD_MAP: dict[str, str] = {
    "atomic_action":  "Atomic Actions",
    "simple_context": "Simple Context",
    "communicative":  "Communicative",
    "transporting":   "Transporting",
    "age":            "Age",
    "motion_status":  "Motion Status",
    "trunk_open":     "Trunk Open",
    "doors_open":     "Doors Open",
}
