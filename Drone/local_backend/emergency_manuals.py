"""
Generates the local knowledge base for the disaster knowledge agent.
Creates ./data/emergency_manuals/ and standard manual documents if they do not exist.
Used by knowledge_agent and startup to ensure vector DB has disaster response content.
"""

import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_DRONE2_ROOT = _HERE.parent.parent
DATA_MANUALS = _DRONE2_ROOT / "data" / "emergency_manuals"


def ensure_manuals_dir() -> Path:
    """Create data/emergency_manuals if it does not exist. Return path."""
    DATA_MANUALS.mkdir(parents=True, exist_ok=True)
    return DATA_MANUALS


CONTENT = {
    "collapse_protocol.txt": """Building collapse search and rescue procedures.

Before entry: Structural assessment by qualified person. Identify load-bearing walls and collapse zones. Check for secondary collapse risk, gas, electrical hazards. Establish communication protocol (hand signals, radio) and backup. Never enter alone; buddy system required.

Victim location: Use systematic search (grid or quadrant). Listen for tapping, voice, use acoustic sensors if available. Mark searched areas. Document locations for rescue teams. Use canines if available for live scent.

Structural assessment: Crack classification — hairline (monitor), moderate (restrict access), severe (evacuate). Assess foundation movement, leaning walls, hanging loads. When in doubt, do not enter. Communicate findings to incident command.""",

    "survivor_extraction.txt": """Survivor extraction procedures.

Triage: Use START (Simple Triage and Rapid Treatment). RPM — Respiration, Perfusion, Mental status. Priority order: Immediate (red), Delayed (yellow), Minor (green), Deceased (black). Allocate resources to Immediate first; do not spend time on deceased.

Extraction: Stabilize spine if mechanism suggests injury. Use minimal movement. Create space (cribbing, lifting) rather than pulling. Protect airway during move. Document condition before and after move.

Medical: Control hemorrhage, maintain airway, treat for shock. Coordinate with EMS for handoff. Team coordination: one team leader, clear roles (extraction, medical, safety).""",

    "structural_hazard.txt": """Structural hazard classification and risk levels.

Crack types: Hairline — often cosmetic; monitor. Moderate — width >3mm or progressive; restrict access, engineer assessment. Severe — significant width, displacement, or active; evacuate area.

Risk levels: LOW — minor damage, no load-bearing compromise; proceed with caution. MEDIUM — localized damage, possible load path change; limit occupancy, brief entries only. HIGH — significant damage, collapse potential; evacuate, no entry except rescue with shoring.

When to evacuate: Unstable facade, falling debris, gas leak, fire spread, or when structural engineer recommends. When to proceed: Low risk, clear objective, PPE, buddy system, and incident command approval.""",

    "fire_response.txt": """Fire behavior in structures and safe entry.

Fire behavior: Compartment fires — growth, flashover risk, backdraft in under-ventilated spaces. Smoke movement — rises, then banks down; stay low. Identify flow path (air in vs out).

Safe entry conditions: Enter only when IC approves. Full PPE and SCBA. Check door temperature; if hot, do not open. Vent before entry when possible. Maintain egress path. Do not commit beyond point of no return.

Suppression priorities: Life safety first. Protect egress. Contain spread. Coordinate with fire service. In wildfire-affected structures, check for smoldering and hotspots before re-entry.""",

    "triage_guide.txt": """START triage system and resource allocation.

START: 30-second assessment per victim. Respiration: >30/min → Immediate. <30 → check perfusion. Perfusion: Cap refill >2s or no radial pulse → Immediate. Mental status: Cannot follow simple commands → Immediate. Otherwise tag Delayed (yellow) or Minor (green).

Priority classifications: Immediate (red) — need treatment and transport within minutes. Delayed (yellow) — need care within hours. Minor (green) — walking wounded, treat last. Deceased (black) — no spontaneous respiration after airway open; do not allocate resources.

Resource allocation: By survivor count and condition. One critical victim — focus resources. Many critical — prioritize salvageable. Document triage tags and hand off to EMS.""",
}


def ensure_all_manuals() -> int:
    """Create each manual file in data/emergency_manuals/ if it does not exist. Return count created."""
    ensure_manuals_dir()
    created = 0
    for name, text in CONTENT.items():
        path = DATA_MANUALS / name
        if not path.is_file():
            path.write_text(text, encoding="utf-8")
            created += 1
    return created
