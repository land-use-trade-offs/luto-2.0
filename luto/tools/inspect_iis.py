"""
Inspect the IIS (Irreducible Infeasible Subsystem) from a Gurobi .ilp file.

Parses the ILP file exported after `model.computeIIS()` + `model.write(...)`,
then maps variable/constraint names back to human-readable land uses, cells,
management options, and constraint types.

Usage (from within a simulation):
    from luto.tools.inspect_iis import analyze_iis
    analyze_iis(ilp_path, data)

Or standalone (loads data automatically):
    python -m luto.tools.inspect_iis <path_to_ilp_file>
"""

import io
import re
import sys
import os
import textwrap
import pandas as pd
import luto.settings as settings

from collections import Counter, defaultdict
from contextlib import redirect_stdout
from pathlib import Path
from luto.tools import am_name_snake_case



# ──────────────────────────── lookup tables ───────────────────────────────

def build_lookup_tables(data):
    """
    Build all lookup tables from the data object.

    Returns a dict with keys:
        AG_LU_NAMES, NON_AG_LU_NAMES, AM_SNAKE_TO_DISPLAY,
        DESC2AGLU, AM2J, COMMODITY_NAMES
    """
    # Agricultural land uses (index → name) from data.AGLU2DESC
    AG_LU_NAMES = {k: v for k, v in data.AGLU2DESC.items() if k >= 0}

    # Non-agricultural land uses (index → name), 0-based for ILP variable indexing
    NON_AG_LU_NAMES = {i: lu for i, lu in enumerate(data.NON_AGRICULTURAL_LANDUSES)}

    # Agricultural management snake_case → display name
    # Built from all AM keys in settings (enabled or not), using the same
    # snake_case transform as the solver: am_name_snake_case()
    AM_SNAKE_TO_DISPLAY = {
        am_name_snake_case(am): am
        for am in settings.AG_MANAGEMENTS
    }

    # Land use name → j index
    DESC2AGLU = data.DESC2AGLU

    # AM name → list of j indices
    AM2J = {
        am: [DESC2AGLU[lu] for lu in lu_list]
        for am, lu_list in settings.AG_MANAGEMENTS_TO_LAND_USES.items()
        if am in settings.AG_MANAGEMENTS
    }

    # Commodity names (index → name) from data.COMMODITIES (sorted list)
    COMMODITY_NAMES = {i: name for i, name in enumerate(data.COMMODITIES)}

    return {
        "AG_LU_NAMES": AG_LU_NAMES,
        "NON_AG_LU_NAMES": NON_AG_LU_NAMES,
        "AM_SNAKE_TO_DISPLAY": AM_SNAKE_TO_DISPLAY,
        "DESC2AGLU": DESC2AGLU,
        "AM2J": AM2J,
        "COMMODITY_NAMES": COMMODITY_NAMES,
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
    bounds_idx = content.index("Bounds\n")

    constraints_block = content[subj_idx + len("Subject To"):bounds_idx]
    bounds_block = content[bounds_idx + len("Bounds\n"):]

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
    for line in bounds_block.strip().split(""):
        line = line.strip()
        if not line or line == "End":
            continue
        bounds.append(line)

    return constraints, bounds


# ──────────────────────────── decoders ────────────────────────────────────

def decode_var(var_name: str, luts: dict) -> dict:
    """Decode a variable name into human-readable components."""
    info = {"raw": var_name}
    AG_LU_NAMES = luts["AG_LU_NAMES"]
    NON_AG_LU_NAMES = luts["NON_AG_LU_NAMES"]
    AM_SNAKE_TO_DISPLAY = luts["AM_SNAKE_TO_DISPLAY"]

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


def decode_constraint(name: str, luts: dict) -> dict:
    """Decode a constraint name into human-readable components."""
    info = {"raw": name}
    AG_LU_NAMES = luts["AG_LU_NAMES"]
    AM_SNAKE_TO_DISPLAY = luts["AM_SNAKE_TO_DISPLAY"]

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


def decode_bound(bound_line: str, luts: dict) -> dict:
    """Decode a bound line (e.g., 'X_ag_dry_23_0 = 1' or '0.5 <= X_ag_dry_1_5 <= 1')."""
    info = {"raw": bound_line}

    # Range: "LB <= VAR <= UB"  (check this FIRST since it's more specific)
    m = re.match(r"([\d.e+-]+)\s*<=\s*(\S+)\s*<=\s*([\d.e+-]+)", bound_line)
    if m:
        lb = float(m.group(1))
        var_name = m.group(2)
        ub = float(m.group(3))
        var_info = decode_var(var_name, luts)
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
        var_info = decode_var(var_name, luts)
        info.update({"var": var_info, "operator": "=", "value": value})
        return info

    # "VAR >= VALUE"
    m = re.match(r"(X_\S+|V\S*|E|W\S*)\s*>=\s*([\d.e+-]+)", bound_line)
    if m:
        var_name = m.group(1)
        value = float(m.group(2))
        var_info = decode_var(var_name, luts)
        info.update({"var": var_info, "operator": ">=", "value": value})
        return info

    # "VAR <= VALUE"
    m = re.match(r"(X_\S+|V\S*|E|W\S*)\s*<=\s*([\d.e+-]+)", bound_line)
    if m:
        var_name = m.group(1)
        value = float(m.group(2))
        var_info = decode_var(var_name, luts)
        info.update({"var": var_info, "operator": "<=", "value": value})
        return info

    # "-infinity <= VAR <= UB"
    m = re.match(r"-infinity\s*<=\s*(\S+)\s*<=\s*([\d.e+-]+)", bound_line)
    if m:
        var_name = m.group(1)
        ub = float(m.group(2))
        var_info = decode_var(var_name, luts)
        info.update({"var": var_info, "operator": "free_ub", "value": 0, "ub": ub})
        return info

    # "VAR free"
    m = re.match(r"(\S+)\s+free", bound_line)
    if m:
        var_name = m.group(1)
        var_info = decode_var(var_name, luts)
        info.update({"var": var_info, "operator": "free", "value": 0})
        return info

    # Fallback
    info["var"] = {"type": "unparsed", "raw": bound_line}
    return info


# ──────────────────────────── analysis helpers ────────────────────────────

def _var_category(var_info):
    """Descriptive category for a decoded variable, used for grouping."""
    vtype = var_info.get("type", "?")
    if vtype == "ag_man":
        return f"ag_man ({var_info.get('am_name', '?')})"
    if vtype == "ag":
        return f"ag ({var_info.get('land_mgmt', '?')})"
    if vtype == "non_ag":
        return f"non_ag ({var_info.get('lu_name', '?')})"
    return vtype


def _constraint_label(dec, commodity_names):
    """Build a human-readable label for a decoded constraint."""
    ctype = dec["type"]
    if ctype == "adoption_limit":
        return f"ADOPTION LIMIT: {dec['am_name']} on {dec['lu_name']} (j={dec['lu_idx']})"
    if ctype == "demand":
        c = dec.get("commodity_idx", "?")
        return f"DEMAND: commodity {c} = {commodity_names.get(c, f'Unknown({c})')}"
    if ctype == "bio_GBF2":
        return "BIODIVERSITY GBF2: Priority degraded area restoration"
    if ctype == "bio_GBF3_NVIS":
        return f"BIODIVERSITY GBF3 NVIS: {dec.get('vegetation_group', '?')}"
    if ctype == "bio_GBF3_IBRA":
        return f"BIODIVERSITY GBF3 IBRA: {dec.get('bioregion', '?')}"
    if ctype == "bio_GBF4_SNES":
        return f"BIODIVERSITY GBF4 SNES: {dec.get('species', '?')}"
    if ctype == "bio_GBF4_ECNES":
        return f"BIODIVERSITY GBF4 ECNES: {dec.get('community', '?')}"
    if ctype == "bio_GBF8":
        return f"BIODIVERSITY GBF8: {dec.get('species', '?')}"
    if ctype == "water_limit":
        return f"WATER LIMIT: {dec.get('region', '?')}"
    if ctype == "renewable_target":
        return f"RENEWABLE TARGET: {dec.get('re_type', '?')} in {dec.get('state', '?')}"
    if ctype == "ghg_limit":
        return f"GHG LIMIT: {dec.get('subtype', '?')}"
    if ctype == "regional_adoption":
        return (f"REGIONAL ADOPTION: {dec.get('category', '?')} "
                f"{dec.get('lu_name', '?')} (region={dec.get('region_id', '?')})")
    # Fallback for unknown / future constraint types
    return f"{ctype.upper()}: {dec.get('raw', '?')}"


def _extract_body_vars(body):
    """Extract variable names from a constraint body string."""
    return set(re.findall(r'(X_[A-Za-z0-9_]+|V\[\d+\]|E|W\[\d+\])', body))


# ──────────────────────────── output tee ──────────────────────────────────

class _TeeIO:
    """Write to both the real stdout and a StringIO buffer."""

    def __init__(self, real_stdout):
        self._real = real_stdout
        self._buf = io.StringIO()

    def write(self, s):
        self._real.write(s)
        self._buf.write(s)

    def flush(self):
        self._real.flush()

    def getvalue(self):
        return self._buf.getvalue()


# ──────────────────────────── analysis ────────────────────────────────────

def analyze_iis(filepath: str, data):
    """Full IIS analysis: parse, decode, summarize, and identify conflicts.

    All printed output is also saved to ``iis_analysis_summary.txt``
    next to the input ``.ilp`` file.

    Parameters
    ----------
    filepath : str
        Path to the Gurobi .ilp file.
    data : luto.data.Data
        The LUTO data object, used to build lookup tables dynamically.
    """
    tee = _TeeIO(sys.stdout)

    # Redirect all print() inside _analyze_iis_inner to the tee
    with redirect_stdout(tee):
        _analyze_iis_inner(filepath, data)

    # Write captured output to summary file
    summary_path = os.path.join(os.path.dirname(filepath), "iis_analysis_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(tee.getvalue())
    print(f"Summary saved to: {summary_path}", flush=True)


def _analyze_iis_inner(filepath: str, data):
    """Core IIS analysis logic (all output via print)."""
    luts = build_lookup_tables(data)
    COMMODITY_NAMES = luts["COMMODITY_NAMES"]

    print(f"{'='*80}", flush=True)
    print(f"IIS ANALYSIS: {Path(filepath).name}", flush=True)
    print(f"{'='*80}\n", flush=True)

    constraints, bounds = parse_ilp(filepath)

    # ── 1. Constraint summary ──
    print(f"[1] CONSTRAINT SUMMARY ({len(constraints)} constraints in IIS)", flush=True)
    print("-" * 60, flush=True)

    constraint_types = Counter()
    decoded_constraints = []
    for name, body in constraints:
        dec = decode_constraint(name, luts)
        decoded_constraints.append((name, body, dec))
        constraint_types[dec["type"]] += 1

    for ctype, count in constraint_types.most_common():
        print(f"  {ctype:40s} : {count:>6,d}", flush=True)
    print(flush=True)

    # ── 2. Variable (bounds) summary ──
    print(f"[2] VARIABLE BOUNDS SUMMARY ({len(bounds)} variable bounds in IIS)", flush=True)
    print("-" * 60, flush=True)

    var_types = Counter()
    decoded_bounds = []
    for bline in bounds:
        dec = decode_bound(bline, luts)
        decoded_bounds.append(dec)
        var_info = dec.get("var", {})
        vtype = var_info.get("type", "unknown")
        if vtype == "ag_man":
            vtype = f"ag_man ({var_info.get('am_name', '?')})"
        var_types[vtype] += 1

    for vt, count in var_types.most_common():
        print(f"  {vt:40s} : {count:>6,d}", flush=True)
    print(flush=True)

    # ── 3. Cells involved in IIS ──
    print(f"[3] CELLS INVOLVED IN IIS", flush=True)
    print("-" * 60, flush=True)

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
    if all_cells:
        print(f"  Cells in constraints : {len(cells_in_constraints):,d}", flush=True)
        print(f"  Cells in bounds      : {len(cells_in_bounds):,d}", flush=True)
        print(f"  Total unique cells   : {len(all_cells):,d}", flush=True)
        print(f"  Cell range           : [{min(all_cells)}, {max(all_cells)}]", flush=True)
    else:
        print("  No cell-level variables found in IIS.", flush=True)
    print(flush=True)

    # ── 4. Global constraints (non cell-level) ──
    print(f"[4] GLOBAL CONSTRAINTS IN IIS (non cell-level)", flush=True)
    print("-" * 60, flush=True)
    print("These are the high-level targets/limits that conflict with cell-level decisions:\n", flush=True)

    global_constraints = []
    for name, body, dec in decoded_constraints:
        if dec["type"] not in ("cell_usage", "ag_mam_usage"):
            global_constraints.append((name, body, dec))

    if not global_constraints:
        print("  No global constraints found in IIS.", flush=True)
        print("  The infeasibility is purely at the cell level.\n", flush=True)
    else:
        for name, body, dec in global_constraints:
            label = _constraint_label(dec, COMMODITY_NAMES)
            print(f"  {label}", flush=True)
            # Extract and show operator + RHS from constraint body
            rhs_m = re.search(r'(>=|<=|=)\s*([\d.e+-]+)\s*$', body)
            if rhs_m:
                print(f"    -> {rhs_m.group(1)} {rhs_m.group(2)} (rescaled)", flush=True)
            print(f"    -> Body (truncated): {body[:200]}...", flush=True)
            print(flush=True)

    # ── 5. Variable bounds analysis ──
    print(f"[5] VARIABLE BOUNDS ANALYSIS", flush=True)
    print("-" * 60, flush=True)
    print("Variables with non-trivial bounds (forced by previous-year lock-in or solver logic):\n", flush=True)

    # Collect variables with effective lower bounds > 0
    # Includes: >= val, = val (where val > 0), range lb..ub (where lb > 0)
    locked_in = []             # effective lb >= 0.5
    weakly_lb = []             # 0 < effective lb < 0.5
    locked_var_names = set()   # raw var names, for cross-referencing with constraint bodies

    for dec in decoded_bounds:
        op = dec.get("operator")
        var_info = dec.get("var", {})
        raw_name = var_info.get("raw", "")

        # Determine effective lower bound
        eff_lb = None
        if op in (">=", "=") and dec.get("value", 0) > 0:
            eff_lb = dec["value"]
        elif op == "range" and dec.get("lb", 0) > 0:
            eff_lb = dec["lb"]

        if eff_lb is not None:
            locked_var_names.add(raw_name)
            if eff_lb >= 0.5:
                locked_in.append((var_info, eff_lb))
            else:
                weakly_lb.append((var_info, eff_lb))

    # Group by descriptive category
    locked_by_category = defaultdict(list)
    for var_info, val in locked_in + weakly_lb:
        cat = _var_category(var_info)
        locked_by_category[cat].append((var_info, val))

    print(f"  Variables locked (lb >= 0.5)        : {len(locked_in):,d}", flush=True)
    print(f"  Variables weakly bounded (0<lb<0.5) : {len(weakly_lb):,d}", flush=True)
    print(flush=True)
    if locked_by_category:
        print("  Breakdown by category:", flush=True)
        for cat, items in sorted(locked_by_category.items(), key=lambda x: -len(x[1])):
            strong = sum(1 for _, v in items if v >= 0.5)
            weak = len(items) - strong
            print(f"    {cat:50s} : {len(items):>6,d}  ({strong} locked, {weak} weak)", flush=True)
    print(flush=True)

    # ── 6. Conflict diagnosis ──
    print(f"[6] CONFLICT DIAGNOSIS", flush=True)
    print("=" * 80, flush=True)
    print(textwrap.dedent("""\
    The IIS identifies the MINIMAL set of constraints + bounds that together
    are infeasible. Removing ANY ONE element would make the rest feasible.
    """), flush=True)

    # ── 6a. Lock-in detail ──
    print("  [6a] LOCK-IN DETAILS", flush=True)
    print("  " + "-" * 60, flush=True)

    if not locked_by_category:
        print("  No variables are locked by bounds (lb > 0).", flush=True)
        print("  The infeasibility is purely between constraint targets.\n", flush=True)
    else:
        for cat, items in sorted(locked_by_category.items(), key=lambda x: -len(x[1])):
            strong = sum(1 for _, v in items if v >= 0.5)
            print(f"  [{cat}]: {len(items)} variables ({strong} locked >= 0.5)", flush=True)
            # Show sample cells (top 5 by bound value)
            sample = sorted(items, key=lambda x: -x[1])[:5]
            for var_info, val in sample:
                cell = var_info.get("cell", "?")
                parts = [
                    var_info.get("land_mgmt", ""),
                    var_info.get("am_name", ""),
                    var_info.get("lu_name", ""),
                ]
                desc = " ".join(p for p in parts if p)
                print(f"    cell {str(cell):>6s}: {desc:<45s} lb={val:.6f}", flush=True)
            if len(items) > 5:
                print(f"    ... and {len(items) - 5} more", flush=True)
        print(flush=True)

    # ── 6b. Global constraint feasibility ──
    print("  [6b] GLOBAL CONSTRAINT FEASIBILITY", flush=True)
    print("  " + "-" * 60, flush=True)
    print("  For each global constraint, how many of its variables are locked vs free:\n", flush=True)

    constraint_pressure = []
    for name, body, dec in global_constraints:
        body_vars = _extract_body_vars(body)
        n_body = len(body_vars)
        locked_in_body = body_vars & locked_var_names
        n_locked = len(locked_in_body)
        n_free = n_body - n_locked

        # Extract operator and RHS
        rhs_m = re.search(r'(>=|<=|=)\s*([\d.e+-]+)\s*$', body)
        op_str = rhs_m.group(1) if rhs_m else "?"
        rhs_str = rhs_m.group(2) if rhs_m else "?"

        constraint_pressure.append((name, dec, n_body, n_locked, n_free, op_str, rhs_str))

        label = _constraint_label(dec, COMMODITY_NAMES)
        pct_locked = (n_locked / n_body * 100) if n_body > 0 else 0
        print(f"  {label}", flush=True)
        print(f"    Variables: {n_body:,d} total | {n_locked:,d} locked ({pct_locked:.0f}%) | {n_free:,d} free", flush=True)
        if rhs_m:
            print(f"    Target: {op_str} {rhs_str} (rescaled)", flush=True)
        if n_body > 0 and pct_locked >= 80:
            print(f"    *** HIGH PRESSURE: {pct_locked:.0f}% of variables locked ***", flush=True)
        print(flush=True)

    # ── 6c. Conflict summary ──
    print("  [6c] CONFLICT SUMMARY", flush=True)
    print("  " + "=" * 60, flush=True)

    # Cell budget analysis
    cell_usage_cells = {dec["cell"] for _, _, dec in decoded_constraints if dec["type"] == "cell_usage"}
    locked_cells = set()
    for var_info, val in locked_in:
        cell = var_info.get("cell")
        if cell is not None:
            locked_cells.add(cell)

    free_cells = cell_usage_cells - locked_cells
    overlap = cell_usage_cells & locked_cells

    if cell_usage_cells:
        print(f"  Cell budget:", flush=True)
        print(f"    Cells with cell_usage constraint : {len(cell_usage_cells):,d}", flush=True)
        print(f"    Cells locked (lb >= 0.5)         : {len(locked_cells):,d}", flush=True)
        print(f"    Overlap (locked + cell_usage)     : {len(overlap):,d}", flush=True)
        print(f"    Free cells (not locked)           : {len(free_cells):,d}", flush=True)

    # Rank constraints by lock-in pressure
    if constraint_pressure:
        print(f"  Constraints ranked by lock-in pressure:", flush=True)
        ranked = sorted(constraint_pressure, key=lambda x: (-(x[3] / (x[2] or 1)), -x[3]))
        for name, dec, n_body, n_locked, n_free, op_str, rhs_str in ranked:
            pct = (n_locked / n_body * 100) if n_body > 0 else 0
            label = _constraint_label(dec, COMMODITY_NAMES)
            print(f"    {pct:5.1f}% locked | {n_free:>6,d} free | {label}", flush=True)

    # Data-driven explanations
    explanations = []

    # Explanation: Lock-in pressure
    if locked_in and overlap:
        cat_cells = defaultdict(set)
        for var_info, val in locked_in:
            cat = _var_category(var_info)
            cell = var_info.get("cell")
            if cell is not None:
                cat_cells[cat].add(cell)
        top_cats = sorted(cat_cells.items(), key=lambda x: -len(x[1]))[:3]
        cat_desc = ", ".join(f"{cat} ({len(cells):,d} cells)" for cat, cells in top_cats)
        explanations.append(
            f"{len(overlap):,d} cells are locked by prior-year decisions, "
            f"leaving only {len(free_cells):,d} free cells.\n"
            f"  Main lock-in categories: {cat_desc}\n"
            f"  These cells cannot be reassigned, reducing feasible space for all global constraints."
        )

    # Explanation: High-pressure constraints
    for name, dec, n_body, n_locked, n_free, op_str, rhs_str in constraint_pressure:
        pct = (n_locked / n_body * 100) if n_body > 0 else 0
        if pct >= 50:
            label = _constraint_label(dec, COMMODITY_NAMES)
            explanations.append(
                f"{label}:\n"
                f"  {pct:.0f}% of its variables are locked, leaving only {n_free:,d} free "
                f"variables to meet the target ({op_str} {rhs_str})."
            )

    # Explanation: Pure constraint competition (no lock-ins)
    if len(global_constraints) > 1 and not locked_in:
        gc_labels = [_constraint_label(dec, COMMODITY_NAMES) for _, _, dec in global_constraints]
        explanations.append(
            f"No variables are locked, but {len(global_constraints)} global constraints "
            f"compete for the same cell allocations:\n"
            + "".join(f"  - {lbl}" for lbl in gc_labels)
            + "  The constraint targets are mutually incompatible."
        )

    if not explanations:
        explanations.append(
            "No dominant conflict pattern detected from automated analysis.\n"
            "  Review the global constraints and variable bounds above for manual diagnosis."
        )

    print(f"  Diagnosis:", flush=True)
    for i, expl in enumerate(explanations, 1):
        print(f"  [{i}] {expl}", flush=True)
    print(flush=True)

    # ── 7. Cell overlap analysis ──
    print(f"[7] CELL OVERLAP ANALYSIS", flush=True)
    print("-" * 60, flush=True)

    # Find cells that appear in multiple constraint types
    cell_to_constraint_types = defaultdict(set)
    for _, _, dec in decoded_constraints:
        if "cell" in dec:
            cell_to_constraint_types[dec["cell"]].add(dec["type"])

    # Cells with the most constraint types
    multi_constrained = {c: types for c, types in cell_to_constraint_types.items() if len(types) > 1}
    if multi_constrained:
        type_combos = Counter()
        for _, types in multi_constrained.items():
            type_combos[frozenset(types)] += 1

        print(f"  Cells appearing in multiple constraint types: {len(multi_constrained):,d}", flush=True)
        print(flush=True)
        print("  Most common constraint-type combinations:", flush=True)
        for combo, count in type_combos.most_common(10):
            print(f"    {count:>5,d} cells: {' + '.join(sorted(combo))}", flush=True)
    else:
        print("  No cells appear in multiple constraint types.", flush=True)
    print(flush=True)

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
        print(f"[8] Detailed cell list exported to: {csv_path}", flush=True)
        print(f"    Total rows: {len(df):,d}", flush=True)

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
        print(f"    Constraint list exported to: {csv_constr_path}", flush=True)
        print(f"    Total constraints: {len(df_c):,d}", flush=True)

    print(f"{'='*80}", flush=True)
    print("ANALYSIS COMPLETE", flush=True)
    print(f"{'='*80}", flush=True)


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
        print(f"ERROR: File not found: {ilp_path}", flush=True)
        sys.exit(1)

    # Load data for standalone usage
    import luto.simulation as sim
    data = sim.load_data()
    analyze_iis(ilp_path, data)
