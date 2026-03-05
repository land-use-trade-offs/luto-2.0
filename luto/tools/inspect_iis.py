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

import re
import sys
import os

import luto.settings as settings

from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from luto.tools import am_name_snake_case, _TeeIO


# ──────────────────────────── lookup tables ───────────────────────────────

def build_lookup_tables(data):
    """Build lookup tables from the data object."""
    AG_LU_NAMES = {k: v for k, v in data.AGLU2DESC.items() if k >= 0}
    NON_AG_LU_NAMES = {i: lu for i, lu in enumerate(data.NON_AGRICULTURAL_LANDUSES)}
    AM_SNAKE_TO_DISPLAY = {
        am_name_snake_case(am): am for am in settings.AG_MANAGEMENTS
    }
    COMMODITY_NAMES = {i: name for i, name in enumerate(data.COMMODITIES)}

    return {
        "AG_LU_NAMES": AG_LU_NAMES,
        "NON_AG_LU_NAMES": NON_AG_LU_NAMES,
        "AM_SNAKE_TO_DISPLAY": AM_SNAKE_TO_DISPLAY,
        "COMMODITY_NAMES": COMMODITY_NAMES,
    }


# ──────────────────────────── parser ──────────────────────────────────────

def parse_ilp(filepath: str):
    """
    Parse a Gurobi ILP (IIS) file and return:
      - constraints: list of (name, body_text) for each constraint
      - bounds: list of bound line strings
    """
    with open(filepath, "r") as f:
        content = f.read()

    subj_match = re.search(r"^Subject To\s*$", content, re.MULTILINE)
    bounds_match = re.search(r"^Bounds\s*$", content, re.MULTILINE)
    end_match = re.search(r"^End\s*$", content, re.MULTILINE)

    # Parse constraints
    constraints = []
    if subj_match:
        subj_end = subj_match.end()
        constr_end = bounds_match.start() if bounds_match else (end_match.start() if end_match else len(content))
        constraints_block = content[subj_end:constr_end]

        constraint_pattern = re.compile(
            r"^ (\S+):\s*(.*?)(?=^ \S+:|\Z)", re.MULTILINE | re.DOTALL
        )
        for m in constraint_pattern.finditer(constraints_block):
            name = m.group(1)
            body = re.sub(r"\s+", " ", m.group(2).strip())
            constraints.append((name, body))

    # Parse bounds
    bounds = []
    if bounds_match:
        bounds_start = bounds_match.end()
        bounds_end = end_match.start() if end_match else len(content)
        for line in content[bounds_start:bounds_end].strip().split("\n"):
            line = line.strip()
            if line and line != "End":
                bounds.append(line)

    return constraints, bounds


# ──────────────────────────── decoders ────────────────────────────────────

def _decode_var(var_name: str, luts: dict) -> str:
    """Decode a variable name into a human-readable string."""
    AG = luts["AG_LU_NAMES"]
    NON_AG = luts["NON_AG_LU_NAMES"]
    AM = luts["AM_SNAKE_TO_DISPLAY"]

    # X_ag_dry_{j}_{r}  or  X_ag_irr_{j}_{r}
    m = re.match(r"X_ag_(dry|irr)_(\d+)_(\d+)", var_name)
    if m:
        lm = "dryland" if m.group(1) == "dry" else "irrigated"
        j, r = int(m.group(2)), int(m.group(3))
        return f"ag {lm} {AG.get(j, f'LU({j})')} @ cell {r}"

    # X_non_ag_{k}_{r}
    m = re.match(r"X_non_ag_(\d+)_(\d+)", var_name)
    if m:
        k, r = int(m.group(1)), int(m.group(2))
        return f"non_ag {NON_AG.get(k, f'LU({k})')} @ cell {r}"

    # X_ag_man_{dry|irr}_{am_snake}_{j}_{r}
    m = re.match(r"X_ag_man_(dry|irr)_(.+?)_(\d+)_(\d+)$", var_name)
    if m:
        lm = "dryland" if m.group(1) == "dry" else "irrigated"
        am_snake = m.group(2)
        j, r = int(m.group(3)), int(m.group(4))
        am_display = AM.get(am_snake, am_snake)
        return f"ag_man {lm} {am_display} on {AG.get(j, f'LU({j})')} @ cell {r}"

    # Penalty variables
    m = re.match(r"V\[(\d+)\]", var_name)
    if m:
        c = int(m.group(1))
        COMM = luts["COMMODITY_NAMES"]
        return f"demand penalty ({COMM.get(c, f'commodity {c}')})"
    if var_name == "E":
        return "GHG penalty"
    m = re.match(r"W\[(\d+)\]", var_name)
    if m:
        return f"water penalty (region {m.group(1)})"

    return var_name


def _decode_constraint(name: str, luts: dict) -> str:
    """Decode a constraint name into a human-readable string."""
    AG = luts["AG_LU_NAMES"]
    AM = luts["AM_SNAKE_TO_DISPLAY"]
    COMM = luts["COMMODITY_NAMES"]

    m = re.match(r"const_cell_usage_(\d+)", name)
    if m:
        return f"Cell usage @ cell {m.group(1)}"

    m = re.match(r"const_ag_mam_(dry|irr)_usage_(.+?)_(\d+)_(\d+)$", name)
    if m:
        lm = "dryland" if m.group(1) == "dry" else "irrigated"
        am_display = AM.get(m.group(2).lower(), m.group(2))
        j = int(m.group(3))
        return f"AM usage: {lm} {am_display} on {AG.get(j, f'LU({j})')} @ cell {m.group(4)}"

    m = re.match(r"const_ag_mam_adoption_limit_(.+?)_(\d+)$", name)
    if m:
        am_display = AM.get(m.group(1).lower(), m.group(1))
        j = int(m.group(2))
        return f"Adoption limit: {am_display} on {AG.get(j, f'LU({j})')}"

    m = re.match(r"demand_soft_bound_lower\[(\d+)\]", name)
    if m:
        c = int(m.group(1))
        return f"Demand: {COMM.get(c, f'commodity({c})')}"

    m = re.match(r"water_yield_limit_(.*)", name)
    if m:
        return f"Water limit: {m.group(1).replace('_', ' ')}"

    m = re.match(r"renewable_(solar|wind)_target_(.*)", name)
    if m:
        return f"Renewable {m.group(1)} target: {m.group(2).replace('_', ' ')}"

    if name.startswith("ghg_emissions"):
        return f"GHG limit: {name}"

    m = re.match(r"bio_(GBF\d+\w*)_limit_(.*)", name)
    if m:
        return f"Biodiversity {m.group(1)}: {m.group(2).replace('_', ' ')}"

    if name.startswith("bio_GBF2"):
        return "Biodiversity GBF2: priority degraded area"

    m = re.match(r"reg_adopt_limit_(ag|non_ag)_(.+?)_(\d+)$", name)
    if m:
        return f"Regional adoption: {m.group(1)} {m.group(2).replace('_', ' ')} (region {m.group(3)})"

    return name


def _is_cell_level(name: str) -> bool:
    """Return True if the constraint is cell-level (not a global target)."""
    return bool(
        re.match(r"const_cell_usage_\d+", name)
        or re.match(r"const_ag_mam_(dry|irr)_usage_", name)
    )


def _parse_bound_var(bline: str) -> str | None:
    """Extract variable name from a bound line. Returns None if unparseable."""
    # Range: "LB <= VAR <= UB"
    m = re.match(r"[\d.e+-]+\s*<=\s*(\S+)\s*<=\s*[\d.e+-]+", bline)
    if m:
        return m.group(1)
    # -infinity range: "-infinity <= VAR <= UB"
    m = re.match(r"-infinity\s*<=\s*(\S+)\s*<=\s*[\d.e+-]+", bline)
    if m:
        return m.group(1)
    # "VAR >= VAL", "VAR <= VAL", "VAR = VAL"
    m = re.match(r"(\S+)\s*(>=|<=|=)\s*[\d.e+-]+", bline)
    if m:
        return m.group(1)
    # "VAR free"
    m = re.match(r"(\S+)\s+free", bline)
    if m:
        return m.group(1)
    return None



# ──────────────────────────── analysis ────────────────────────────────────

def analyze_iis(filepath: str, data):
    """Parse IIS .ilp file and print a concise infeasibility diagnosis.

    Output is also saved to ``iis_analysis_summary.txt`` next to the .ilp file.
    """
    summary_path = os.path.join(os.path.dirname(filepath), "iis_analysis_summary.txt")
    with (
        open(summary_path, "w", encoding="utf-8") as f,
        redirect_stdout(_TeeIO(sys.stdout, f)),
    ):
        _analyze_iis_inner(filepath, data)
    print(f"Summary saved to: {summary_path}", flush=True)


def _analyze_iis_inner(filepath: str, data):
    """Core IIS analysis — concise output."""
    luts = build_lookup_tables(data)
    constraints, bounds = parse_ilp(filepath)

    print(f"{'='*80}", flush=True)
    print(f"IIS ANALYSIS: {Path(filepath).name}", flush=True)
    print(f"  {len(constraints)} constraint(s), {len(bounds)} bound(s)", flush=True)
    print(f"{'='*80}", flush=True)

    # ── Classify constraints ──
    global_constraints = []
    cell_constraints = []
    for name, body in constraints:
        if _is_cell_level(name):
            cell_constraints.append((name, body))
        else:
            global_constraints.append((name, body))

    # ── Infeasible constraints ──
    print(f"\nINFEASIBLE CONSTRAINTS:", flush=True)
    print("-" * 60, flush=True)

    if global_constraints:
        for name, body in global_constraints:
            label = _decode_constraint(name, luts)
            rhs_m = re.search(r'(>=|<=|=)\s*([\d.e+-]+)\s*$', body)
            rhs_str = f" {rhs_m.group(1)} {rhs_m.group(2)}" if rhs_m else ""
            body_vars = re.findall(r'(X_[A-Za-z0-9_]+|V\[\d+\]|E|W\[\d+\])', body)
            print(f"  {label}{rhs_str}  ({len(body_vars)} variables)", flush=True)

    if cell_constraints:
        # Summarize cell-level constraints by type
        cell_types = Counter()
        cell_ids = []
        for name, _ in cell_constraints:
            m = re.match(r"(const_cell_usage|const_ag_mam_\w+_usage)", name)
            ctype = m.group(1) if m else name
            cell_types[ctype] += 1
            # Extract cell id (last number in the constraint name)
            cm = re.search(r"_(\d+)$", name)
            if cm:
                cell_ids.append(int(cm.group(1)))

        for ctype, count in cell_types.most_common():
            label = ctype.replace("const_", "").replace("_", " ")
            print(f"  {count} {label} constraints", end="", flush=True)
            if cell_ids:
                print(f" (cells {min(cell_ids)}-{max(cell_ids)})", flush=True)
            else:
                print(flush=True)

    if not constraints:
        print("  (none)", flush=True)

    # ── Build bound lookup ──
    bound_map = {}  # var_name -> bound string
    for bline in bounds:
        var_name = _parse_bound_var(bline)
        if var_name:
            bound_map[var_name] = bline.strip()

    # ── Diagnosis ──
    print(f"\nDIAGNOSIS:", flush=True)
    print("=" * 60, flush=True)
    print("  The IIS is the MINIMAL set of constraints + bounds that", flush=True)
    print("  together are infeasible. Removing ANY ONE would restore feasibility.", flush=True)

    if not constraints and not bounds:
        print("\n  Empty IIS — no constraints or bounds found.", flush=True)
    elif not constraints:
        print("\n  No constraints in IIS — variable bounds alone are contradictory:", flush=True)
        for bline in bounds[:10]:
            var_name = _parse_bound_var(bline)
            decoded = _decode_var(var_name, luts) if var_name else bline
            print(f"    {decoded}  [{bline.strip()}]", flush=True)
        if len(bounds) > 10:
            print(f"    ... {len(bounds) - 10} more", flush=True)
    elif not bounds:
        print("\n  No variable bounds in IIS — constraint targets are mutually incompatible.", flush=True)
    else:
        # Show equation for each global constraint
        n_global = len(global_constraints)
        for idx, (name, body) in enumerate(global_constraints):
            label = _decode_constraint(name, luts)
            rhs_m = re.search(r'(>=|<=|=)\s*([\d.e+-]+)\s*$', body)
            rhs_op = rhs_m.group(1) if rhs_m else "="
            rhs_val = rhs_m.group(2) if rhs_m else "?"

            # Extract (coefficient, variable) pairs from constraint body
            terms = re.findall(
                r'([+-]?\s*[\d.e+-]+)\s+(X_[A-Za-z0-9_]+|V\[\d+\]|E|W\[\d+\])', body
            )

            locked_terms = []
            free_terms = []
            for coeff_str, var_name in terms:
                try:
                    coeff = float(coeff_str.replace(" ", ""))
                except ValueError:
                    coeff = 0.0
                if var_name in bound_map:
                    locked_terms.append((var_name, coeff, bound_map[var_name]))
                else:
                    free_terms.append((var_name, coeff))

            prefix = f"  [{idx+1}/{n_global}] " if n_global > 1 else "  "
            print(f"  {prefix}Constraint: {label}", flush=True)
            print(f"  Variables: {len(terms)} total, {len(locked_terms)} locked, {len(free_terms)} free", flush=True)
            print(flush=True)

            # Build equation lines
            n_total = len(locked_terms) + len(free_terms)
            show_all = n_total <= 10
            lines = []

            # Locked terms (show up to 10, then summarize)
            for var_name, coeff, bnd in locked_terms[:10]:
                decoded = _decode_var(var_name, luts)
                lines.append(f"[{bnd}] * {coeff}  # {decoded}")
            if len(locked_terms) > 10:
                lines.append(f"... {len(locked_terms) - 10} more locked variables")

            # Free terms
            if show_all:
                for var_name, coeff in free_terms:
                    decoded = _decode_var(var_name, luts)
                    lines.append(f"[{var_name} free] * {coeff}  # {decoded}")
            elif free_terms:
                lines.append(f"... {len(free_terms)} free variables")

            for i, line in enumerate(lines):
                prefix = "    " if i == 0 else "  + "
                print(f"{prefix}{line}", flush=True)
            print(f"  {rhs_op} {rhs_val}  (rescaled target)", flush=True)

        # Summarize cell-level constraints
        if cell_constraints:
            print(f"  Cell-level: {len(cell_constraints)} constraints", end="", flush=True)
            n_cell_locked = sum(
                1 for _, body in cell_constraints
                for var in re.findall(r'(X_[A-Za-z0-9_]+)', body)
                if var in bound_map
            )
            if n_cell_locked:
                print(f" ({n_cell_locked} variables locked by bounds)", flush=True)

        # Standalone bounds not in any constraint
        all_body_vars = set()
        for _, body in constraints:
            all_body_vars.update(re.findall(r'(X_[A-Za-z0-9_]+|V\[\d+\]|E|W\[\d+\])', body))
        standalone = [v for v in bound_map if v not in all_body_vars]
        if standalone:
            print(f"\n  Standalone bounds (not in any constraint): {len(standalone)}", flush=True)
            for var_name in standalone[:5]:
                decoded = _decode_var(var_name, luts)
                print(f"    {decoded}  [{bound_map[var_name]}]", flush=True)
            if len(standalone) > 5:
                print(f"    ... {len(standalone) - 5} more", flush=True)

    print(f"\n{'='*80}", flush=True)


# ──────────────────────────── main ────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1:
        ilp_path = sys.argv[1]
    else:
        ilp_path = r"F:\Users\jinzhu\Documents\luto-2.0\output\2026_02_17__10_10_20_RF13_2010-2050\debug_model_2045_2050.ilp"

    if not os.path.isfile(ilp_path):
        print(f"ERROR: File not found: {ilp_path}", flush=True)
        sys.exit(1)

    import luto.simulation as sim
    data = sim.load_data()
    analyze_iis(ilp_path, data)
