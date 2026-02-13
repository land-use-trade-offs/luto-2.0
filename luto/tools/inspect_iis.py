"""
Inspect the IIS (Irreducible Infeasible Subsystem) from a Gurobi .ilp file.

Parses the ILP file exported after `model.computeIIS()` + `model.write(...)`,
then maps variable/constraint names back to human-readable land uses, cells,
management options, and constraint types.

Usage:
    python jinzhu_inspect_code/inspect_iis.py <path_to_ilp_file>

If no argument is given, defaults to the latest debug ILP file.
"""

import re
import sys
import os
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

# ──────────────────────────── lookup tables ───────────────────────────────

# Agricultural land uses (index → name), matches input/ag_landuses.csv
AG_LU_NAMES = {
    0: "Apples",
    1: "Beef - modified land",
    2: "Beef - natural land",
    3: "Citrus",
    4: "Cotton",
    5: "Dairy - modified land",
    6: "Dairy - natural land",
    7: "Grapes",
    8: "Hay",
    9: "Nuts",
    10: "Other non-cereal crops",
    11: "Pears",
    12: "Plantation fruit",
    13: "Rice",
    14: "Sheep - modified land",
    15: "Sheep - natural land",
    16: "Stone fruit",
    17: "Sugar",
    18: "Summer cereals",
    19: "Summer legumes",
    20: "Summer oilseeds",
    21: "Tropical stone fruit",
    22: "Unallocated - modified land",
    23: "Unallocated - natural land",
    24: "Vegetables",
    25: "Winter cereals",
    26: "Winter legumes",
    27: "Winter oilseeds",
}

# Non-agricultural land uses (index → name), from settings.NON_AG_LAND_USES
NON_AG_LU_NAMES = {
    0: "Environmental Plantings",
    1: "Riparian Plantings",
    2: "Sheep Agroforestry",
    3: "Beef Agroforestry",
    4: "Carbon Plantings (Block)",
    5: "Sheep Carbon Plantings (Belt)",
    6: "Beef Carbon Plantings (Belt)",
    7: "BECCS",
    8: "Destocked - natural land",
}

# Agricultural management snake_case → display name
AM_SNAKE_TO_DISPLAY = {
    "asparagopsis_taxiformis": "Asparagopsis taxiformis",
    "precision_agriculture": "Precision Agriculture",
    "ecological_grazing": "Ecological Grazing",
    "savanna_burning": "Savanna Burning",
    "agtech_ei": "AgTech EI",
    "biochar": "Biochar",
    "hir_-_beef": "HIR - Beef",
    "hir_-_sheep": "HIR - Sheep",
    "utility_solar_pv": "Utility Solar PV",
    "onshore_wind": "Onshore Wind",
}

# AG_MANAGEMENTS_TO_LAND_USES — j_idx within each AM maps to these land uses
AM_TO_LU_NAMES = {
    "Asparagopsis taxiformis": [
        "Beef - modified land", "Sheep - modified land",
        "Dairy - natural land", "Dairy - modified land",
    ],
    "Precision Agriculture": [
        "Hay", "Summer cereals", "Summer legumes", "Summer oilseeds",
        "Winter cereals", "Winter legumes", "Winter oilseeds",
        "Cotton", "Other non-cereal crops", "Rice", "Sugar", "Vegetables",
        "Apples", "Citrus", "Grapes", "Nuts", "Pears", "Plantation fruit",
        "Stone fruit", "Tropical stone fruit",
    ],
    "Ecological Grazing": [
        "Beef - modified land", "Sheep - modified land", "Dairy - modified land",
    ],
    "Savanna Burning": [
        "Beef - natural land", "Dairy - natural land",
        "Sheep - natural land", "Unallocated - natural land",
    ],
    "AgTech EI": [
        "Hay", "Summer cereals", "Summer legumes", "Summer oilseeds",
        "Winter cereals", "Winter legumes", "Winter oilseeds",
        "Cotton", "Other non-cereal crops", "Rice", "Sugar", "Vegetables",
        "Apples", "Citrus", "Grapes", "Nuts", "Pears", "Plantation fruit",
        "Stone fruit", "Tropical stone fruit",
    ],
    "Biochar": [
        "Hay", "Summer cereals", "Summer legumes", "Summer oilseeds",
        "Winter cereals", "Winter legumes", "Winter oilseeds",
        "Apples", "Citrus", "Grapes", "Nuts", "Pears", "Plantation fruit",
        "Stone fruit", "Tropical stone fruit",
    ],
    "HIR - Beef": ["Beef - natural land"],
    "HIR - Sheep": ["Sheep - natural land"],
    "Utility Solar PV": [
        "Unallocated - modified land",
        "Beef - modified land", "Sheep - modified land", "Dairy - modified land",
        "Summer cereals", "Summer legumes", "Summer oilseeds",
        "Winter cereals", "Winter legumes", "Winter oilseeds",
    ],
    "Onshore Wind": [
        "Unallocated - modified land",
        "Beef - modified land", "Sheep - modified land", "Dairy - modified land",
        "Hay", "Summer cereals", "Summer legumes", "Summer oilseeds",
        "Winter cereals", "Winter legumes", "Winter oilseeds",
        "Cotton", "Other non-cereal crops", "Rice", "Sugar", "Vegetables",
    ],
}

# Build the DESC2AGLU mapping (land use name → j index)
DESC2AGLU = {v: k for k, v in AG_LU_NAMES.items()}

# Build am2j: AM name → list of j indices (matching solver's am2j property)
AM2J = {}
for am_name, lu_list in AM_TO_LU_NAMES.items():
    AM2J[am_name] = [DESC2AGLU[lu] for lu in lu_list]

# Commodity names (index → name), derived from products list
# Products = sorted(crop products + livestock products)
# Commodities = sorted, deduplicated, lowercase, collapse NATURAL/MODIFIED
COMMODITY_NAMES = {
    0: "apples",
    1: "beef lexp",
    2: "beef meat",
    3: "citrus",
    4: "cotton",
    5: "dairy",
    6: "grapes",
    7: "hay",
    8: "nuts",
    9: "other non-cereal crops",
    10: "pears",
    11: "plantation fruit",
    12: "rice",
    13: "sheep lexp",
    14: "sheep meat",
    15: "sheep wool",
    16: "stone fruit",
    17: "sugar",
    18: "summer cereals",
    19: "summer legumes",
    20: "summer oilseeds",
    21: "tropical stone fruit",
    22: "vegetables",
    23: "winter cereals",
    24: "winter legumes",
    25: "winter oilseeds",
}


# ──────────────────────────── parsers ─────────────────────────────────────

def parse_ilp(filepath: str):
    """
    Parse a Gurobi ILP (IIS) file and return:
      - constraints: list of (name, full_text) for each constraint
      - bounds: list of (var_name, bound_text) for each variable bound in IIS
    """
    with open(filepath, "r") as f:
        content = f.read()

    # Locate sections
    subj_idx = content.index("Subject To")
    bounds_idx = content.index("\nBounds\n")

    constraints_block = content[subj_idx + len("Subject To"):bounds_idx]
    bounds_block = content[bounds_idx + len("\nBounds\n"):]

    # ── Parse constraints ──
    # Constraint names start at column 1 (with leading space) and end with ':'
    # Multi-line constraints have continuation lines indented further
    constraint_pattern = re.compile(
        r"^ (\S+):\s*(.*?)(?=^ \S+:|\Z)", re.MULTILINE | re.DOTALL
    )
    constraints = []
    for m in constraint_pattern.finditer(constraints_block):
        name = m.group(1)
        body = m.group(2).strip()
        # Collapse multi-line into single line
        body = re.sub(r"\s+", " ", body)
        constraints.append((name, body))

    # ── Parse bounds ──
    bounds = []
    for line in bounds_block.strip().split("\n"):
        line = line.strip()
        if not line or line == "End":
            continue
        bounds.append(line)

    return constraints, bounds


# ──────────────────────────── decoders ────────────────────────────────────

def decode_var(var_name: str) -> dict:
    """Decode a variable name into human-readable components."""
    info = {"raw": var_name}

    # X_ag_dry_{j}_{r}  or  X_ag_irr_{j}_{r}
    m = re.match(r"X_ag_(dry|irr)_(\d+)_(\d+)", var_name)
    if m:
        lm = m.group(1)
        j = int(m.group(2))
        r = int(m.group(3))
        info.update({
            "type": "ag",
            "land_mgmt": "dryland" if lm == "dry" else "irrigated",
            "lu_idx": j,
            "lu_name": AG_LU_NAMES.get(j, f"Unknown({j})"),
            "cell": r,
        })
        return info

    # X_non_ag_{k}_{r}
    m = re.match(r"X_non_ag_(\d+)_(\d+)", var_name)
    if m:
        k = int(m.group(1))
        r = int(m.group(2))
        info.update({
            "type": "non_ag",
            "lu_idx": k,
            "lu_name": NON_AG_LU_NAMES.get(k, f"Unknown({k})"),
            "cell": r,
        })
        return info

    # X_ag_man_{dry|irr}_{am_snake}_{j}_{r}
    # NOTE: The variable name uses the GLOBAL land use index j (not the local j_idx within the AM).
    # The AM snake_case name can contain underscores, so we match known patterns.
    m = re.match(r"X_ag_man_(dry|irr)_(.+?)_(\d+)_(\d+)$", var_name)
    if m:
        lm = m.group(1)
        am_snake = m.group(2)
        j = int(m.group(3))  # Global AG land use index
        r = int(m.group(4))

        am_display = AM_SNAKE_TO_DISPLAY.get(am_snake, am_snake)
        lu_name = AG_LU_NAMES.get(j, f"Unknown LU({j})")

        info.update({
            "type": "ag_man",
            "land_mgmt": "dryland" if lm == "dry" else "irrigated",
            "am_name": am_display,
            "lu_idx": j,
            "lu_name": lu_name,
            "cell": r,
        })
        return info

    # Penalty variables
    if var_name.startswith("V"):
        info["type"] = "demand_penalty"
        return info
    if var_name == "E":
        info["type"] = "ghg_penalty"
        return info
    if var_name.startswith("W"):
        info["type"] = "water_penalty"
        return info

    info["type"] = "unknown"
    return info


def decode_constraint(name: str) -> dict:
    """Decode a constraint name into human-readable components."""
    info = {"raw": name}

    # const_cell_usage_{r}
    m = re.match(r"const_cell_usage_(\d+)", name)
    if m:
        r = int(m.group(1))
        info.update({"type": "cell_usage", "cell": r})
        return info

    # const_ag_mam_{dry|irr}_usage_{am}_{j}_{r}
    m = re.match(r"const_ag_mam_(dry|irr)_usage_(.+?)_(\d+)_(\d+)$", name)
    if m:
        lm = m.group(1)
        am_raw = m.group(2)
        j = int(m.group(3))
        r = int(m.group(4))
        am_display = AM_SNAKE_TO_DISPLAY.get(am_raw.lower(), am_raw)
        info.update({
            "type": "ag_mam_usage",
            "land_mgmt": "dryland" if lm == "dry" else "irrigated",
            "am_name": am_display,
            "lu_idx": j,
            "lu_name": AG_LU_NAMES.get(j, f"Unknown({j})"),
            "cell": r,
        })
        return info

    # const_ag_mam_adoption_limit_{am}_{j}
    m = re.match(r"const_ag_mam_adoption_limit_(.+?)_(\d+)$", name)
    if m:
        am_raw = m.group(1)
        j = int(m.group(2))
        am_display = AM_SNAKE_TO_DISPLAY.get(am_raw.lower(), am_raw)
        info.update({
            "type": "adoption_limit",
            "am_name": am_display,
            "lu_idx": j,
            "lu_name": AG_LU_NAMES.get(j, f"Unknown({j})"),
        })
        return info

    # demand_soft_bound_lower[{c}]
    m = re.match(r"demand_soft_bound_lower\[(\d+)\]", name)
    if m:
        c = int(m.group(1))
        info.update({"type": "demand", "commodity_idx": c})
        return info

    # water_yield_limit_{region_name}
    m = re.match(r"water_yield_limit_(.*)", name)
    if m:
        reg = m.group(1).replace("_", " ")
        info.update({"type": "water_limit", "region": reg})
        return info

    # renewable_{solar|wind}_target_{state}
    m = re.match(r"renewable_(solar|wind)_target_(.*)", name)
    if m:
        re_type = m.group(1)
        state = m.group(2).replace("_", " ")
        info.update({"type": "renewable_target", "re_type": re_type, "state": state})
        return info

    # ghg_emissions_limit_*
    if name.startswith("ghg_emissions"):
        info.update({"type": "ghg_limit", "subtype": name})
        return info

    # bio_GBF2_priority_degraded_area_limit
    if name.startswith("bio_GBF2"):
        info.update({"type": "bio_GBF2"})
        return info

    # bio_GBF3_NVIS_limit_{v_name}
    m = re.match(r"bio_GBF3_NVIS_limit_(.*)", name)
    if m:
        info.update({"type": "bio_GBF3_NVIS", "vegetation_group": m.group(1).replace("_", " ")})
        return info

    # bio_GBF3_IBRA_limit_{v_name}
    m = re.match(r"bio_GBF3_IBRA_limit_(.*)", name)
    if m:
        info.update({"type": "bio_GBF3_IBRA", "bioregion": m.group(1).replace("_", " ")})
        return info

    # bio_GBF4_SNES_limit_{x_name}
    m = re.match(r"bio_GBF4_SNES_limit_(.*)", name)
    if m:
        info.update({"type": "bio_GBF4_SNES", "species": m.group(1).replace("_", " ")})
        return info

    # bio_GBF4_ECNES_limit_{x_name}
    m = re.match(r"bio_GBF4_ECNES_limit_(.*)", name)
    if m:
        info.update({"type": "bio_GBF4_ECNES", "community": m.group(1).replace("_", " ")})
        return info

    # bio_GBF8_limit_{s_name}
    m = re.match(r"bio_GBF8_limit_(.*)", name)
    if m:
        info.update({"type": "bio_GBF8", "species": m.group(1).replace("_", " ")})
        return info

    # reg_adopt_limit_{ag|non_ag}_{lu}_{reg_id}
    m = re.match(r"reg_adopt_limit_(ag|non_ag)_(.+?)_(\d+)$", name)
    if m:
        info.update({
            "type": "regional_adoption",
            "category": m.group(1),
            "lu_name": m.group(2).replace("_", " "),
            "region_id": int(m.group(3)),
        })
        return info

    info["type"] = "unknown"
    return info


def decode_bound(bound_line: str) -> dict:
    """Decode a bound line (e.g., 'X_ag_dry_23_0 = 1' or '0.5 <= X_ag_dry_1_5 <= 1')."""
    info = {"raw": bound_line}

    # Range: "LB <= VAR <= UB"  (check this FIRST since it's more specific)
    m = re.match(r"([\d.e+-]+)\s*<=\s*(\S+)\s*<=\s*([\d.e+-]+)", bound_line)
    if m:
        lb = float(m.group(1))
        var_name = m.group(2)
        ub = float(m.group(3))
        var_info = decode_var(var_name)
        info.update({
            "var": var_info,
            "operator": "range",
            "lb": lb,
            "ub": ub,
            "value": lb,  # for filtering, use lb as the effective bound
        })
        return info

    # "VAR = VALUE"
    m = re.match(r"(X_\S+|V\S*|E|W\S*)\s*=\s*([\d.e+-]+)", bound_line)
    if m:
        var_name = m.group(1)
        value = float(m.group(2))
        var_info = decode_var(var_name)
        info.update({"var": var_info, "operator": "=", "value": value})
        return info

    # "VAR >= VALUE"
    m = re.match(r"(X_\S+|V\S*|E|W\S*)\s*>=\s*([\d.e+-]+)", bound_line)
    if m:
        var_name = m.group(1)
        value = float(m.group(2))
        var_info = decode_var(var_name)
        info.update({"var": var_info, "operator": ">=", "value": value})
        return info

    # "VAR <= VALUE"
    m = re.match(r"(X_\S+|V\S*|E|W\S*)\s*<=\s*([\d.e+-]+)", bound_line)
    if m:
        var_name = m.group(1)
        value = float(m.group(2))
        var_info = decode_var(var_name)
        info.update({"var": var_info, "operator": "<=", "value": value})
        return info

    # "-infinity <= VAR <= UB"
    m = re.match(r"-infinity\s*<=\s*(\S+)\s*<=\s*([\d.e+-]+)", bound_line)
    if m:
        var_name = m.group(1)
        ub = float(m.group(2))
        var_info = decode_var(var_name)
        info.update({"var": var_info, "operator": "free_ub", "value": 0, "ub": ub})
        return info

    # "VAR free"
    m = re.match(r"(\S+)\s+free", bound_line)
    if m:
        var_name = m.group(1)
        var_info = decode_var(var_name)
        info.update({"var": var_info, "operator": "free", "value": 0})
        return info

    # Fallback
    info["var"] = {"type": "unparsed", "raw": bound_line}
    return info


# ──────────────────────────── analysis ────────────────────────────────────

def analyze_iis(filepath: str):
    """Full IIS analysis: parse, decode, summarize, and identify conflicts."""

    print(f"{'='*80}")
    print(f"IIS ANALYSIS: {Path(filepath).name}")
    print(f"{'='*80}\n")

    constraints, bounds = parse_ilp(filepath)

    # ── 1. Constraint summary ──
    print(f"[1] CONSTRAINT SUMMARY ({len(constraints)} constraints in IIS)")
    print("-" * 60)

    constraint_types = Counter()
    decoded_constraints = []
    for name, body in constraints:
        dec = decode_constraint(name)
        decoded_constraints.append((name, body, dec))
        constraint_types[dec["type"]] += 1

    for ctype, count in constraint_types.most_common():
        print(f"  {ctype:40s} : {count:>6,d}")
    print()

    # ── 2. Variable (bounds) summary ──
    print(f"\n[2] VARIABLE BOUNDS SUMMARY ({len(bounds)} variable bounds in IIS)")
    print("-" * 60)

    var_types = Counter()
    decoded_bounds = []
    for bline in bounds:
        dec = decode_bound(bline)
        decoded_bounds.append(dec)
        var_info = dec.get("var", {})
        vtype = var_info.get("type", "unknown")
        if vtype == "ag_man":
            vtype = f"ag_man ({var_info.get('am_name', '?')})"
        var_types[vtype] += 1

    for vt, count in var_types.most_common():
        print(f"  {vt:40s} : {count:>6,d}")
    print()

    # ── 3. Cells involved in IIS ──
    print(f"\n[3] CELLS INVOLVED IN IIS")
    print("-" * 60)

    cells_in_constraints = set()
    cells_in_bounds = set()
    for _, _, dec in decoded_constraints:
        if "cell" in dec:
            cells_in_constraints.add(dec["cell"])
    for dec in decoded_bounds:
        var_info = dec.get("var", {})
        if "cell" in var_info:
            cells_in_bounds.add(var_info["cell"])

    all_cells = cells_in_constraints | cells_in_bounds
    print(f"  Cells in constraints : {len(cells_in_constraints):,d}")
    print(f"  Cells in bounds      : {len(cells_in_bounds):,d}")
    print(f"  Total unique cells   : {len(all_cells):,d}")
    print(f"  Cell range           : [{min(all_cells)}, {max(all_cells)}]")
    print()

    # ── 4. Non-trivial constraints (the "global" constraints causing conflict) ──
    print(f"\n[4] GLOBAL CONSTRAINTS IN IIS (non cell-level)")
    print("-" * 60)
    print("These are the high-level targets/limits that conflict with cell-level decisions:\n")

    global_constraints = []
    for name, body, dec in decoded_constraints:
        if dec["type"] not in ("cell_usage", "ag_mam_usage"):
            global_constraints.append((name, body, dec))

    for name, body, dec in global_constraints:
        ctype = dec["type"]

        if ctype == "adoption_limit":
            print(f"  ADOPTION LIMIT: {dec['am_name']} on {dec['lu_name']} (j={dec['lu_idx']})")
            # Extract the adoption fraction from body
            m = re.search(r"<=\s*0", body)
            if m:
                print(f"    -> RHS = 0 (forces adoption to zero or below)")
            else:
                # Try to find the coefficient
                print(f"    -> Body (truncated): {body[:200]}...")

        elif ctype == "demand":
            c = dec["commodity_idx"]
            cm_name = COMMODITY_NAMES.get(c, f"Unknown({c})")
            print(f"  DEMAND PENALTY: commodity {c} = {cm_name}")
            # Show first few terms
            print(f"    -> Body (truncated): {body[:200]}...")

        elif ctype == "bio_GBF2":
            print(f"  BIODIVERSITY GBF2: Priority degraded area restoration")
            # Extract RHS
            m = re.search(r">=\s*([\d.e+-]+)", body)
            if m:
                print(f"    -> Target (rescaled) >= {m.group(1)}")
            print(f"    -> Body (truncated): {body[:200]}...")

        elif ctype == "bio_GBF3_NVIS":
            print(f"  BIODIVERSITY GBF3 NVIS: {dec.get('vegetation_group', '?')}")
        elif ctype == "bio_GBF3_IBRA":
            print(f"  BIODIVERSITY GBF3 IBRA: {dec.get('bioregion', '?')}")
        elif ctype == "bio_GBF4_SNES":
            print(f"  BIODIVERSITY GBF4 SNES: {dec.get('species', '?')}")
        elif ctype == "bio_GBF4_ECNES":
            print(f"  BIODIVERSITY GBF4 ECNES: {dec.get('community', '?')}")
        elif ctype == "bio_GBF8":
            print(f"  BIODIVERSITY GBF8: {dec.get('species', '?')}")
        elif ctype == "water_limit":
            print(f"  WATER LIMIT: {dec.get('region', '?')}")
        elif ctype == "renewable_target":
            print(f"  RENEWABLE TARGET: {dec.get('re_type', '?')} in {dec.get('state', '?')}")
        elif ctype == "ghg_limit":
            print(f"  GHG LIMIT: {dec.get('subtype', '?')}")
        else:
            print(f"  {ctype}: {name}")

        print()

    # ── 5. Analyze variable lower-bound conflicts ──
    print(f"\n[5] VARIABLE LOWER BOUNDS ANALYSIS")
    print("-" * 60)
    print("Variables forced to non-zero values (lower bounds > 0 from previous year lock-in):\n")

    # Separate locked-in variables (lb close to 1) vs weakly constrained
    locked_in = []   # lb >= 0.5
    weakly_lb = []   # 0 < lb < 0.5
    for dec in decoded_bounds:
        op = dec.get("operator")
        val = dec.get("value")
        var_info = dec.get("var", {})
        if op == ">=" and val is not None and val > 0:
            if val >= 0.5:
                locked_in.append((var_info, val))
            else:
                weakly_lb.append((var_info, val))

    # Summarize locked-in by type
    locked_by_type = Counter()
    locked_cells_by_am = defaultdict(list)
    for var_info, val in locked_in:
        vtype = var_info.get("type", "?")
        if vtype == "ag_man":
            am = var_info.get("am_name", "?")
            key = f"ag_man: {am}"
            locked_cells_by_am[am].append((var_info.get("cell"), var_info.get("lu_name", "?"), val))
        elif vtype == "ag":
            key = f"ag ({var_info.get('land_mgmt', '?')})"
        elif vtype == "non_ag":
            key = f"non_ag ({var_info.get('lu_name', '?')})"
        else:
            key = vtype
        locked_by_type[key] += 1

    print(f"  Variables locked in (lb >= 0.5): {len(locked_in):,d}")
    print(f"  Variables weakly bounded (0 < lb < 0.5): {len(weakly_lb):,d}")
    print()
    print("  Locked-in breakdown:")
    for key, count in locked_by_type.most_common():
        print(f"    {key:50s} : {count:>6,d}")
    print()

    # ── 6. Conflict diagnosis ──
    print(f"\n[6] CONFLICT DIAGNOSIS")
    print("=" * 80)
    print(textwrap.dedent("""
    The IIS identifies the MINIMAL set of constraints + bounds that together
    are infeasible. Removing ANY ONE element would make the rest feasible.

    The conflict arises because:
    """))

    # Identify which cells are locked to renewable energy (onshore wind / solar PV)
    renewable_locked_cells = defaultdict(list)
    for var_info, val in locked_in:
        if var_info.get("type") == "ag_man" and var_info.get("am_name") in ("Onshore Wind", "Utility Solar PV"):
            cell = var_info.get("cell")
            am = var_info.get("am_name")
            lu = var_info.get("lu_name", "?")
            lm = var_info.get("land_mgmt", "?")
            renewable_locked_cells[am].append((cell, lu, lm, val))

    # Identify which cells are locked to non-ag land uses
    nonag_locked_cells = defaultdict(list)
    for dec in decoded_bounds:
        op = dec.get("operator")
        val = dec.get("value")
        var_info = dec.get("var", {})
        if var_info.get("type") == "non_ag" and op == ">=" and val is not None and val > 0:
            nonag_locked_cells[var_info.get("lu_name", "?")].append((var_info.get("cell"), val))

    # Print cell-level lock-ins
    if renewable_locked_cells:
        for am, cells in renewable_locked_cells.items():
            locked_count = len([c for c in cells if c[3] >= 0.5])
            weak_count = len(cells) - locked_count
            print(f"  [RENEWABLE] {am}: {len(cells)} cells in IIS "
                  f"({locked_count} locked >=0.5, {weak_count} weakly bounded)")
            # Show sample cells
            sample = sorted(cells, key=lambda x: -x[3])[:10]
            for cell, lu, lm, val in sample:
                print(f"    cell {cell:>6d}: {lm:>9s} {lu:<30s} lb={val:.6f}")
            if len(cells) > 10:
                print(f"    ... and {len(cells) - 10} more cells")
            print()

    if nonag_locked_cells:
        for lu, cells in nonag_locked_cells.items():
            locked = [c for c in cells if c[1] >= 0.5]
            print(f"  [NON-AG] {lu}: {len(cells)} cells in IIS ({len(locked)} locked >=0.5)")
            sample = sorted(cells, key=lambda x: -x[1])[:5]
            for cell, val in sample:
                print(f"    cell {cell:>6d}: lb={val:.6f}")
            if len(cells) > 5:
                print(f"    ... and {len(cells) - 5} more cells")
            print()

    # Print how global constraints conflict with cell-level decisions
    print("\n  CONSTRAINT INTERACTIONS:")
    print("  " + "-" * 60)

    # Find cells that appear in BOTH: ag management usage constraints AND bounds
    # These are cells where the AM variable is forced (by lb) but also constrained
    # to be <= the ag variable, which itself may be constrained by cell_usage = 1

    # Collect cells from adoption limit constraints
    adoption_limit_info = []
    for name, body, dec in decoded_constraints:
        if dec["type"] == "adoption_limit":
            am = dec["am_name"]
            lu = dec["lu_name"]
            j = dec["lu_idx"]
            adoption_limit_info.append((am, lu, j))

    if adoption_limit_info:
        print("\n  ADOPTION LIMITS in IIS:")
        for am, lu, j in adoption_limit_info:
            # Count how many cells are locked for this AM + lu combo
            if am in locked_cells_by_am:
                relevant = [(c, l, v) for c, l, v in locked_cells_by_am[am]
                            if l == lu or True]  # show all for this AM
                print(f"    {am} on {lu} (j={j}): {len(relevant)} cells locked in this AM")

    # Identify the conflict pattern
    print("\n\n  CONFLICT SUMMARY:")
    print("  " + "=" * 60)

    # Build explanation
    explanations = []

    # Check for cell_usage + locked renewables conflict
    cell_usage_cells = {dec["cell"] for _, _, dec in decoded_constraints if dec["type"] == "cell_usage"}
    re_locked_cells_set = set()
    for am, cells in renewable_locked_cells.items():
        for cell, lu, lm, val in cells:
            if val >= 0.5:
                re_locked_cells_set.add(cell)

    overlap_re_cell = cell_usage_cells & re_locked_cells_set
    if overlap_re_cell:
        explanations.append(
            f"  {len(overlap_re_cell):,d} cells have RENEWABLE ENERGY installations locked in "
            f"(non-reversible from previous years).\n"
            f"  These cells' ag management vars are forced to ~1.0 (via lower bounds),\n"
            f"  which forces the underlying ag land use variable to ~1.0 as well\n"
            f"  (via ag_mam_usage constraints: X_ag_man <= X_ag).\n"
            f"  This leaves NO room for other land uses on these cells."
        )

    if global_constraints:
        for name, body, dec in global_constraints:
            ctype = dec["type"]
            if ctype == "bio_GBF2":
                explanations.append(
                    f"  The GBF2 BIODIVERSITY constraint requires a minimum area of priority\n"
                    f"  degraded land to be restored. This needs certain cells to adopt land uses\n"
                    f"  with high biodiversity contribution, potentially conflicting with\n"
                    f"  cells locked into renewable energy or other non-reversible uses."
                )
            elif ctype == "demand":
                c = dec.get("commodity_idx", "?")
                cm_name = COMMODITY_NAMES.get(c, f"Unknown({c})")
                explanations.append(
                    f"  DEMAND constraint for '{cm_name}' (commodity {c}) requires minimum production.\n"
                    f"  Cells locked into non-reversible uses (renewable energy, non-ag plantings)\n"
                    f"  cannot produce this commodity, reducing available production capacity."
                )
            elif ctype == "adoption_limit":
                explanations.append(
                    f"  ADOPTION LIMIT for {dec['am_name']} on {dec['lu_name']}: the total\n"
                    f"  area using this management option is bounded. When cells from previous\n"
                    f"  years are locked in, new cells cannot compensate."
                )
            elif ctype == "water_limit":
                explanations.append(
                    f"  WATER constraint for {dec.get('region', '?')}: net water yield must\n"
                    f"  exceed a minimum. Land use changes locked in from prior years\n"
                    f"  may reduce water yield below the target."
                )
            elif ctype == "renewable_target":
                explanations.append(
                    f"  RENEWABLE TARGET for {dec.get('re_type', '?')} in {dec.get('state', '?')}:\n"
                    f"  state-level generation target requires cells to install renewables,\n"
                    f"  but this may conflict with other constraints on the same cells."
                )

    for i, expl in enumerate(explanations, 1):
        print(f"\n  [{i}] {expl}")

    # ── 7. Cell overlap analysis ──
    print(f"\n\n[7] CELL OVERLAP ANALYSIS")
    print("-" * 60)

    # Find cells that appear in multiple constraint types
    cell_to_constraint_types = defaultdict(set)
    for _, _, dec in decoded_constraints:
        if "cell" in dec:
            cell_to_constraint_types[dec["cell"]].add(dec["type"])

    # Cells with the most constraint types
    multi_constrained = {c: types for c, types in cell_to_constraint_types.items() if len(types) > 1}
    if multi_constrained:
        type_combos = Counter()
        for c, types in multi_constrained.items():
            type_combos[frozenset(types)] += 1

        print(f"  Cells appearing in multiple constraint types: {len(multi_constrained):,d}")
        print()
        print("  Most common constraint-type combinations:")
        for combo, count in type_combos.most_common(10):
            print(f"    {count:>5,d} cells: {' + '.join(sorted(combo))}")
    print()

    # ── 8. Export detailed cell list ──
    output_dir = os.path.dirname(filepath)
    csv_path = os.path.join(output_dir, "iis_analysis_cells.csv")

    rows = []
    for dec in decoded_bounds:
        var_info = dec.get("var", {})
        if "cell" not in var_info:
            continue
        rows.append({
            "cell": var_info.get("cell"),
            "var_type": var_info.get("type"),
            "land_mgmt": var_info.get("land_mgmt", ""),
            "lu_name": var_info.get("lu_name", ""),
            "am_name": var_info.get("am_name", ""),
            "operator": dec.get("operator", ""),
            "bound_value": dec.get("value", ""),
            "raw": dec.get("raw", ""),
        })

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        print(f"\n[8] Detailed cell list exported to: {csv_path}")
        print(f"    Total rows: {len(df):,d}")

    # Also export constraint list
    csv_constr_path = os.path.join(output_dir, "iis_analysis_constraints.csv")
    constr_rows = []
    for name, body, dec in decoded_constraints:
        constr_rows.append({
            "constraint_name": name,
            "type": dec.get("type"),
            "cell": dec.get("cell", ""),
            "am_name": dec.get("am_name", ""),
            "lu_name": dec.get("lu_name", ""),
            "lu_idx": dec.get("lu_idx", ""),
            "region": dec.get("region", dec.get("state", "")),
            "species": dec.get("species", dec.get("vegetation_group", dec.get("bioregion", ""))),
            "commodity_idx": dec.get("commodity_idx", ""),
        })

    if constr_rows:
        df_c = pd.DataFrame(constr_rows)
        df_c.to_csv(csv_constr_path, index=False)
        print(f"    Constraint list exported to: {csv_constr_path}")
        print(f"    Total constraints: {len(df_c):,d}")

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")


# ──────────────────────────── main ────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1:
        ilp_path = sys.argv[1]
    else:
        # Default path
        ilp_path = (
            "F:/Users/jinzhu/Documents/luto-2.0/output/"
            "2026_02_13__11_13_14_RF13_2010-2050/debug_model2015_2020.ilp"
        )

    if not os.path.isfile(ilp_path):
        print(f"ERROR: File not found: {ilp_path}")
        sys.exit(1)

    analyze_iis(ilp_path)
