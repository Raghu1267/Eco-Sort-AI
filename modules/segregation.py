"""Rule-based e-waste segregation engine.

This module contains NO machine learning. It maps the class names predicted by the
YOLOv8s model to general e-waste handling guidance. The mappings are software
recommendations and are NOT legally certified waste classifications.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import pandas as pd

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
CAT_BATTERY = "Batteries and hazardous components"
CAT_IT = "IT and communication equipment"
CAT_LARGE = "Large household appliances"
CAT_SMALL = "Small household appliances"
CAT_DISPLAY = "Displays and lighting equipment"
CAT_PCB = "Electronic circuit boards and components"
CAT_RENEW = "Renewable energy electronic equipment"

BIN_BATTERY = "Battery and hazardous waste"
BIN_IT = "IT and electronics recycling"
BIN_LARGE = "Large appliance recycling"
BIN_SMALL = "Small appliance recycling"
BIN_DISPLAY = "Display and lighting recycling"
BIN_PCB = "PCB and component recovery"
BIN_RENEW = "Renewable energy equipment recycling"

HIGH, MEDIUM, LOW = "High", "Medium", "Low"
HAZARD_ORDER = [HIGH, MEDIUM, LOW]
HAZARD_COLORS = {HIGH: "#ff5d6c", MEDIUM: "#ffb84d", LOW: "#3ddc97"}

DISCLAIMER = (
    "These are software-generated, general recommendations based on the object class "
    "predicted by the AI model. They are not legally certified waste classifications. "
    "General safety guidance is not a substitute for official local disposal "
    "requirements: always follow the rules of your local authority, a licensed e-waste "
    "collector or the manufacturer's take-back scheme."
)

GENERAL_HAZARD_WARNING = (
    "Never dismantle, open, crush, puncture, burn or incinerate electronic items or "
    "batteries. Damaged or swollen batteries can cause fire."
)


@dataclass(frozen=True)
class SegregationRule:
    category: str
    bin: str
    hazard: str
    disposal: str
    recycling: str
    safety: str


def _r(category: str, bin_: str, hazard: str, disposal: str, recycling: str, safety: str) -> SegregationRule:
    return SegregationRule(category, bin_, hazard, disposal, recycling, safety)


# --------------------------------------------------------------------------- #
# Rules for the 37 trained classes (keys are the exact model class names)
# --------------------------------------------------------------------------- #
SEGREGATION_RULES: Dict[str, SegregationRule] = {
    "Battery": _r(
        CAT_BATTERY, BIN_BATTERY, HIGH,
        "Drop at a battery collection point or an authorised e-waste centre. Do not put in household or general waste.",
        "Metals such as cobalt, nickel, lithium, zinc and manganese can be recovered by specialist battery recyclers.",
        "Fire and chemical hazard. Tape the terminals of lithium batteries, store in a cool dry place away from metal, and never crush, puncture or burn. Keep damaged or swollen batteries separate.",
    ),
    "Blood-Pressure-Monitor": _r(
        CAT_BATTERY, BIN_BATTERY, MEDIUM,
        "Take to an e-waste or medical-device take-back point. Hand in any batteries through the battery stream.",
        "Plastics, electronics and sensors can be recovered; batteries go to battery recyclers.",
        "Contains batteries and electronics. Older mercury-based devices need hazardous-waste handling: do not break them. Do not open the unit.",
    ),
    "Boiler": _r(
        CAT_LARGE, BIN_LARGE, MEDIUM,
        "Arrange collection by a licensed appliance recycler or installer. Do not leave it in household waste.",
        "Steel, copper, aluminium and brass are highly recoverable; control boards go to PCB recovery.",
        "Heavy item: use proper lifting help. May contain residual water, gas fittings, controls and, in very old units, insulation of unknown composition. Do not cut or open it yourself.",
    ),
    "Clothes-Iron": _r(
        CAT_SMALL, BIN_SMALL, LOW,
        "Take to a small appliance recycling point or an e-waste drop-off.",
        "Metal soleplate, copper wiring and plastic housing can be recovered.",
        "Unplug and let it cool fully. Empty any water. Cut off or secure the power cord. Do not disassemble.",
    ),
    "Coffee-Machine": _r(
        CAT_SMALL, BIN_SMALL, LOW,
        "Take to a small appliance recycling point or a retailer take-back scheme.",
        "Steel, aluminium, copper and plastics can be recovered; the control board goes to PCB recovery.",
        "Unplug, empty water and grounds, and let it cool. Do not open the heating or pump assembly.",
    ),
    "Computer-Keyboard": _r(
        CAT_IT, BIN_IT, LOW,
        "Drop at an IT equipment collection point or e-waste centre.",
        "Plastics, copper and small circuit boards can be recovered.",
        "Wireless models may contain batteries: follow local battery rules. Do not pry the casing open.",
    ),
    "Computer-Mouse": _r(
        CAT_IT, BIN_IT, LOW,
        "Drop at an IT equipment collection point or e-waste centre.",
        "Plastics, copper cable and small circuit boards can be recovered.",
        "Wireless models may contain batteries: follow local battery rules. Do not pry the casing open.",
    ),
    "Cooling-Display": _r(
        CAT_LARGE, BIN_LARGE, HIGH,
        "Arrange collection by a licensed refrigeration or appliance recycler. Do not scrap it informally.",
        "Refrigerant and oil must be recovered by certified technicians; steel, copper, aluminium and glass can then be recycled.",
        "Contains refrigerant gas and compressor oil that harm the ozone layer and climate. Never cut pipes or tip the unit. Keep the cooling circuit intact and use a certified handler.",
    ),
    "Desktop-PC": _r(
        CAT_IT, BIN_IT, MEDIUM,
        "Drop at an IT equipment collection point, or use a certified IT asset-disposal service.",
        "Metals, circuit boards, drives and power supplies can be recovered; precious metals are recoverable from boards.",
        "Securely erase personal data first. Contains capacitors and a small cell battery. Do not open the power supply unit.",
    ),
    "Digital-Oscilloscope": _r(
        CAT_IT, BIN_IT, MEDIUM,
        "Send to a certified electronics recycler or return it to the manufacturer or lab asset-disposal office.",
        "Circuit boards, display module, metals and plastics can be recovered.",
        "Test equipment may hold capacitors, batteries and a backlit display. Unplug and do not open the case.",
    ),
    "Drone": _r(
        CAT_BATTERY, BIN_BATTERY, HIGH,
        "Take the drone and its battery to a battery or e-waste collection point. Do not force the battery out.",
        "Lithium battery, motors, magnets and circuit boards can be recovered by specialist recyclers.",
        "Lithium-polymer batteries can ignite if damaged. Store the battery discharged and cool, in a non-flammable container, and never crush or puncture it.",
    ),
    "Electric-Guitar": _r(
        CAT_SMALL, BIN_SMALL, LOW,
        "Consider repair, resale or donation first. If it is beyond reuse, take it to an e-waste drop-off.",
        "Pickups, wiring and electronics can be recovered; wood and metal parts are separated by the recycler.",
        "Active pickups may contain a battery: follow local battery rules. Cut or secure loose cables. Do not take the electronics apart.",
    ),
    "Electronic-Keyboard": _r(
        CAT_SMALL, BIN_SMALL, LOW,
        "Consider reuse or donation first. Otherwise take to an e-waste drop-off.",
        "Circuit boards, speakers, wiring and plastics can be recovered.",
        "May contain batteries or a power adapter. Unplug it and follow local battery rules. Do not open the casing.",
    ),
    "Flashlight": _r(
        CAT_BATTERY, BIN_BATTERY, MEDIUM,
        "Take to a battery or small-appliance collection point. Batteries go in the battery stream.",
        "Metal body, LED module and batteries can be recovered separately.",
        "Contains batteries that can leak or overheat. Do not crush, burn or force it open.",
    ),
    "Flat-Panel-Monitor": _r(
        CAT_DISPLAY, BIN_DISPLAY, MEDIUM,
        "Take to a display-equipment collection point or e-waste centre. Do not put in general waste.",
        "Glass, plastics, metals and circuit boards can be recovered by a specialist recycler.",
        "Fragile screen; older LCD backlights may contain mercury. Do not break the screen or open the casing. Carry it upright with care.",
    ),
    "Flat-Panel-TV": _r(
        CAT_DISPLAY, BIN_DISPLAY, MEDIUM,
        "Take to a display-equipment collection point, or use a retailer or manufacturer take-back scheme.",
        "Glass, plastics, metals and circuit boards can be recovered by a specialist recycler.",
        "Fragile screen and heavy; older LCD backlights may contain mercury. Do not break the screen or open the casing. Use two people to move large TVs.",
    ),
    "Glucose-Meter": _r(
        CAT_SMALL, BIN_BATTERY, MEDIUM,
        "Take to an e-waste or medical-device take-back point. Hand in the battery through the battery stream.",
        "Plastics, small circuit boards and battery metals can be recovered.",
        "May carry biological contamination. Used lancets and test strips are medical sharps or biohazard waste: dispose of them separately following local medical-waste rules. Do not open the device.",
    ),
    "HDD": _r(
        CAT_IT, BIN_IT, LOW,
        "Use a certified IT recycler with a data-destruction service.",
        "Aluminium, circuit boards and rare-earth magnets can be recovered.",
        "Data security risk: erase or physically destroy data through a certified service. Contains very strong magnets. Do not open the drive.",
    ),
    "Laptop": _r(
        CAT_IT, BIN_IT, MEDIUM,
        "Drop at an IT equipment collection point, or use a manufacturer or certified IT take-back service.",
        "Metals, circuit boards, display and the lithium battery can be recovered by specialist recyclers.",
        "Securely erase personal data first. The lithium-ion battery is a fire risk if damaged or swollen: keep the laptop intact and do not open it.",
    ),
    "Microwave": _r(
        CAT_SMALL, BIN_SMALL, MEDIUM,
        "Take to a small or large appliance recycling point, or use a retailer take-back scheme.",
        "Steel, copper and circuit boards can be recovered.",
        "Capacitors inside can hold a dangerous charge even when unplugged. Never open the casing. Cut or secure the power cord.",
    ),
    "Music-Player": _r(
        CAT_IT, BIN_IT, MEDIUM,
        "Take to an e-waste or electronics collection point.",
        "Circuit boards, plastics and the built-in battery can be recovered by specialist recyclers.",
        "Built-in lithium batteries can overheat if damaged. Erase personal data first. Do not open or crush the device.",
    ),
    "Oven": _r(
        CAT_LARGE, BIN_LARGE, LOW,
        "Arrange collection by a licensed appliance recycler or a retailer take-back scheme.",
        "Steel, copper wiring, glass and control boards can be recovered.",
        "Heavy and can have sharp edges: get help lifting. Have it disconnected from power and gas by a qualified person. Do not dismantle it.",
    ),
    "PCB": _r(
        CAT_PCB, BIN_PCB, MEDIUM,
        "Keep separate from general waste and deliver to a certified e-waste or component-recovery facility.",
        "Copper, gold, silver, palladium and tin can be recovered through specialist smelting and refining.",
        "May contain lead, brominated flame retardants and other substances. Handle with gloves, avoid sharp solder points, and never burn or acid-treat boards.",
    ),
    "Photovoltaic-Panel": _r(
        CAT_RENEW, BIN_RENEW, MEDIUM,
        "Arrange collection through a solar-panel take-back scheme or certified recycler.",
        "Glass, aluminium frames, copper and silicon can be recovered; some panel types need specialist treatment.",
        "Can still produce voltage in light: keep cables insulated. Broken glass is a cut hazard; some thin-film types contain toxic metals. Do not break or cut panels.",
    ),
    "Projector": _r(
        CAT_DISPLAY, BIN_DISPLAY, MEDIUM,
        "Take to a display-equipment collection point or return to the manufacturer.",
        "Optics, circuit boards, metals and plastics can be recovered.",
        "The lamp in many projectors contains mercury: do not break it. Let it cool completely before moving. Do not open the casing.",
    ),
    "Refrigerator": _r(
        CAT_LARGE, BIN_LARGE, HIGH,
        "Arrange collection by a licensed appliance recycler or a retailer take-back scheme. Do not scrap it informally.",
        "Refrigerant and oil must be recovered by certified technicians; steel, copper, aluminium and plastics can then be recycled.",
        "Contains refrigerant gas and compressor oil that harm the ozone layer and climate. Never cut pipes or tip the unit. Remove food, defrost, and keep the cooling circuit intact. Use two people to move it.",
    ),
    "Rotary-Mower": _r(
        CAT_LARGE, BIN_LARGE, MEDIUM,
        "Take to a metal or appliance recycling centre. Petrol models must be drained by a qualified person first.",
        "Steel, aluminium and motors can be recovered; cordless models have a battery for the battery stream.",
        "Fuel, oil and sharp blades are hazards. Remove the spark plug lead or battery before handling and wear cut-resistant gloves. Do not drain fuel yourself without proper containers and rules.",
    ),
    "Router": _r(
        CAT_IT, BIN_IT, LOW,
        "Drop at an IT or telecom equipment collection point, or return to your internet provider.",
        "Circuit boards, metals and plastics can be recovered.",
        "Reset the device to remove saved network passwords. Do not open the casing.",
    ),
    "Server": _r(
        CAT_IT, BIN_IT, MEDIUM,
        "Use a certified IT asset-disposal service with data destruction.",
        "Metals, circuit boards, drives, memory and power supplies can be recovered.",
        "Heavy equipment: use lifting help. Data security risk: destroy data through a certified service. Contains batteries and capacitors. Do not open the power supply.",
    ),
    "Smartphone": _r(
        CAT_IT, BIN_IT, MEDIUM,
        "Use a manufacturer, retailer or e-waste take-back point. Consider resale or donation if it still works.",
        "Precious and base metals, glass and the lithium battery can be recovered by specialist recyclers.",
        "Back up and erase personal data first. The lithium-ion battery can ignite if swollen or damaged: do not open, bend or crush the phone.",
    ),
    "Smoke-Detector": _r(
        CAT_BATTERY, BIN_BATTERY, HIGH,
        "Do not put in household waste. Return to the manufacturer, a hazardous-waste facility or an authorised e-waste collector.",
        "Plastics, circuit boards and batteries can be recovered; ionisation types need specialist handling.",
        "Some ionisation smoke detectors contain a tiny sealed radioactive source. Never open or damage the unit. Check the label and ask the supplier or local authority for take-back rules.",
    ),
    "Straight-Tube-Fluorescent-Lamp": _r(
        CAT_DISPLAY, BIN_DISPLAY, HIGH,
        "Take to a lamp collection point or hazardous-waste facility. Never put in general waste.",
        "Glass, aluminium end-caps and mercury phosphor powder can be recovered by specialist lamp recyclers.",
        "Contains mercury vapour. Handle gently to avoid breakage. If broken, ventilate the room, avoid skin contact, do not vacuum, and follow local clean-up guidance.",
    ),
    "Street-Lamp": _r(
        CAT_DISPLAY, BIN_DISPLAY, MEDIUM,
        "Return through the lighting operator or municipality to a lamp and luminaire recycler.",
        "Metal housing, LED modules, drivers and glass can be recovered; discharge lamps need specialist handling.",
        "Only a qualified electrician should disconnect street lighting. Discharge lamps can contain mercury or sodium. Do not break lamps or open driver units.",
    ),
    "TV-Remote-Control": _r(
        CAT_SMALL, BIN_SMALL, LOW,
        "Drop at a small electronics collection point.",
        "Plastics and the small circuit board can be recovered.",
        "Contains batteries: follow local battery rules and keep used batteries out of general waste. Do not force the casing open.",
    ),
    "Telephone-Set": _r(
        CAT_IT, BIN_IT, LOW,
        "Drop at an IT or telecom equipment collection point.",
        "Plastics, copper wiring and circuit boards can be recovered.",
        "Cordless handsets contain rechargeable batteries: follow local battery rules. Do not open the casing.",
    ),
    "USB-Flash-Drive": _r(
        CAT_IT, BIN_IT, LOW,
        "Drop at an IT equipment collection point or e-waste centre.",
        "Small circuit boards and memory chips can be recovered.",
        "Data security risk: erase the drive securely before disposal, or use a certified data-destruction service.",
    ),
    "Washing-Machine": _r(
        CAT_LARGE, BIN_LARGE, LOW,
        "Arrange collection by a licensed appliance recycler or a retailer take-back scheme.",
        "Steel, copper motor windings, aluminium and control boards can be recovered.",
        "Very heavy: use two people and a trolley. Disconnect power and water, drain remaining water, and secure the drum and cable. Do not remove the counterweights.",
    ),
}

# Fallback for a class that exists in the model but has no rule (never invents a class).
UNMAPPED_RULE = SegregationRule(
    category="Unmapped class",
    bin="Check local e-waste rules",
    hazard=MEDIUM,
    disposal="No rule is defined for this class. Contact your local e-waste authority or a licensed collector.",
    recycling="Not available.",
    safety="Treat as potentially hazardous. Do not open, crush or burn the item.",
)

RULE_COLUMNS = [
    "E-Waste Category",
    "Segregation Bin",
    "Hazard Priority",
    "Disposal Guidance",
    "Recycling Recommendation",
    "Safety Instructions",
]


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def get_rule(class_name: str) -> SegregationRule:
    """Return the rule for an exact model class name (or the unmapped fallback)."""
    return SEGREGATION_RULES.get(class_name, UNMAPPED_RULE)


def check_mapping(model_class_names: Iterable[str]) -> Dict[str, List[str]]:
    """Compare model class names with the rule table.

    Returns the model classes that have no rule ("unmapped") and the rules that no
    model class uses ("unused").
    """
    names = set(model_class_names)
    return {
        "unmapped": sorted(names - set(SEGREGATION_RULES)),
        "unused": sorted(set(SEGREGATION_RULES) - names),
    }


def add_segregation(detections: pd.DataFrame) -> pd.DataFrame:
    """Append the rule columns to a detections DataFrame that has a 'Class' column."""
    out = detections.copy()
    if out.empty:
        for col in RULE_COLUMNS:
            out[col] = pd.Series(dtype="object")
        return out
    rules = [get_rule(c) for c in out["Class"]]
    out["E-Waste Category"] = [r.category for r in rules]
    out["Segregation Bin"] = [r.bin for r in rules]
    out["Hazard Priority"] = [r.hazard for r in rules]
    out["Disposal Guidance"] = [r.disposal for r in rules]
    out["Recycling Recommendation"] = [r.recycling for r in rules]
    out["Safety Instructions"] = [r.safety for r in rules]
    return out


def count_by(df: pd.DataFrame, column: str, order: Optional[List[str]] = None) -> pd.DataFrame:
    """Count detections by a column; returns DataFrame[column, Count]."""
    if df.empty or column not in df:
        return pd.DataFrame({column: [], "Count": []})
    counts = df[column].value_counts()
    if order:
        counts = counts.reindex([o for o in order if o in counts.index])
    return counts.rename_axis(column).reset_index(name="Count")
