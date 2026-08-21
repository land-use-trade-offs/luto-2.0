# Copyright 2025 Bryan, B.A., Williams, N., Archibald, C.L., de Haan, F., Wang, J., 
# van Schoten, N., Hadjikakou, M., Sanson, J.,  Zyngier, R., Marcos-Martinez, R.,  
# Navarro, J.,  Gao, L., Aghighi, H., Armstrong, T., Bohl, H., Jaffe, P., Khan, M.S., 
# Moallemi, E.A., Nazari, A., Pan, X., Steyl, D., and Thiruvady, D.R.
#
# This file is part of LUTO2 - Version 2 of the Australian Land-Use Trade-Offs model
#
# LUTO2 is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# LUTO2 is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# LUTO2. If not, see <https://www.gnu.org/licenses/>.



"""
Data about transitions costs.
"""

import numpy as np
import luto.tools as tools
import luto.data as Data
import luto.economics.agricultural.ghg as ag_ghg

from functools import lru_cache
from luto import settings
from typing import Dict
from luto.data import Data
from luto.economics.agricultural.water import get_wreq_matrices



@lru_cache(maxsize=1)
def _fold_ag_dvar(data: Data, base_year: int):
    """SOLVER-WORLD base ag dvar fold — shared compute behind ``get_folded_base_ag_dvar`` ([0], the
    folded base) and ``get_ag_dvar_fold_map`` ([1], the sliver map). Returns ``(folded_base, fold_map)``.
    Prefer the two public wrappers; this private, ``lru_cache``-d function exists only so the single fold
    (which most callers need only the folded base of) is computed once per (data, base_year).

    - ``[0]`` folded base dvar — sub-θ slivers merged into the dominant. This is the FOLDED composition,
      used by the TRANSITION machinery (source maps, flow costs, ub/lb bounds, node-balance base). It makes
      the transition problem small, but it is NOT the true composition.
    - ``[1]`` fold map — the sliver bookkeeping that recovers the ACTUAL (unfolded, true) dvar for
      ACCOUNTING. The two-stream accounting model uses it to build X_acct (the true per-land-use fractions)
      so water/GHG/biodiversity/production/profit score the TRUE composition rather than the folded one.

    ``folded_base`` (NLMS, NCELLS, N_AG_LUS): the true base with every sub-θ land-use fraction FOLDED into
    the cell's dominant source (θ = EXACT_REACHABILITY_MIN_FRACTION).

    ★ FOLD-INTO-DOMINANT: θ is a dial between the exact per-source flow model and the old crisp
    dominant-LU model, applied per cell. Worked example, θ = 0.10, one dry cell:

        true base:    Beef 0.55 │ Winter cereals 0.35 │ Hay 0.06 │ Citrus 0.04
        folded base:  Beef 0.65 │ Winter cereals 0.35                            (Hay+Citrus → Beef)

    - Beef and Winter cereals are > θ, so each becomes its own SOURCE: the solver attaches one flow
      delta variable per legal (T_MAT-finite) target, e.g. D[Beef→Sheep], D[Beef→EP],
      D[WC→Barley], ... Their transitions are TRUE and EXACT — a flow out of Winter cereals is
      charged Winter cereals' own from→to cost/water/GHG row, never some cell-average.
    - Hay (0.06) and Citrus (0.04) are ≤ θ: they get NO delta variables of their own. Their 0.10 of
      land is added to Beef (the dominant), stays fully mobile through Beef's delta variables, and
      pays Beef's from→to costs if it moves — the crisp approximation, confined to the sub-θ tail.
    - Beef, the cell's overall-largest land-use, is always exempt from folding (receiver of last
      resort), so every cell keeps at least one source and NO land is ever locked in by θ.

    Receiver choice: the sliver's same-lm largest land-use if that land-use is itself > θ (avoids
    fake dry↔irr cost attribution), else the cell's overall-largest land-use. Cell totals are
    preserved exactly, so ag_mask/Σ-X accounting is unchanged. θ→0: nothing folds (pure exact);
    θ→1: one source per cell carrying the whole cell (pure crisp).

    Everything the solver derives from the base-year ag dvar (source maps, flow costs, ub/base
    consts, ag-man lb) MUST use the folded base (``[0]``, via ``get_folded_base_ag_dvar``) so the solver
    world is self-consistent. The true map (data.ag_dvars) is untouched — reporting sees real allocations;
    solved delta flows attribute folded land's moves to its dominant source (bounded by the folded area).

    ``fold_map`` (``[1]``, via ``get_ag_dvar_fold_map``) is a dict describing every folded sliver — the
    data the two-stream accounting needs to re-express each dominant as its true land-uses (all arrays are
    size 0 when nothing folded, so X_acct == the folded stream, a no-op):

        from_m, cells, from_j : sliver source (lm, cell, land-use) index arrays
        vals                  : the sliver's base-year fraction moved into the dominant (the `slivers` array)
        to_m, to_j            : the receiver dominant's (lm, land-use) index arrays (cell is `cells`)
        folded_dom            : folded_base[to_m, cells, to_j] — the dominant's post-fold mass (denominator)

    Cached (maxsize=1): every consumer calls this for the same (data, base_year) within one step.
    """
    base = data.ag_dvars[base_year].astype(np.float32).copy()
    noise = 10 ** (-settings.ROUND_DECIMALS)
    theta = settings.EXACT_REACHABILITY_MIN_FRACTION

    # Overall dominant (m*, j*) per cell — receiver of last resort, exempt from folding.
    flat = base.transpose(1, 0, 2).reshape(data.NCELLS, data.NLMS * data.N_AG_LUS)
    dom_flat = flat.argmax(axis=1)
    dom_m, dom_j = np.divmod(dom_flat, data.N_AG_LUS)

    # Same-lm dominant per (m, cell).
    dom_j_same = base.argmax(axis=2)                                     # (NLMS, NCELLS)

    sliver = (base > noise) & (base <= theta)
    sliver[dom_m, np.arange(data.NCELLS), dom_j] = False                 # exempt the receiver

    # Fold map — initialised empty, filled below once the slivers are known. Empty (size-0) arrays ⇒
    # nothing folded, so X_acct == the folded stream (the accounting correction is a no-op).
    fold_map = dict(
        from_m=np.array([], dtype=np.intp),
        cells=np.array([], dtype=np.intp),
        from_j=np.array([], dtype=np.intp),
        vals=np.array([], dtype=np.float32),
        to_m=np.array([], dtype=np.intp),
        to_j=np.array([], dtype=np.intp),
        folded_dom=np.array([], dtype=np.float32),
    )
    if not sliver.any():
        return base, fold_map

    from_m, cells, from_j = np.where(sliver)
    vals = base[from_m, cells, from_j].copy()

    # Receiver: same-lm dominant if itself > θ, else the overall dominant. (A sliver that IS its
    # lm's largest land-use fails the > θ test on itself and falls through to the overall dominant.)
    to_j_same = dom_j_same[from_m, cells]
    same_ok = base[from_m, cells, to_j_same] > theta
    to_m = np.where(same_ok, from_m, dom_m[cells])
    to_j = np.where(same_ok, to_j_same, dom_j[cells])

    np.add.at(base, (to_m, cells, to_j), vals)
    base[from_m, cells, from_j] = 0.0

    fold_map.update(
        from_m=from_m.astype(np.intp),
        cells=cells.astype(np.intp),
        from_j=from_j.astype(np.intp),
        vals=vals.astype(np.float32),
        to_m=to_m.astype(np.intp),
        to_j=to_j.astype(np.intp),
        folded_dom=base[to_m, cells, to_j].astype(np.float32),   # post-fold dominant mass (denominator)
    )

    folded_ha = float((vals * data.REAL_AREA[cells]).sum())
    print(
        f"  └── θ fold (θ={theta}): {len(vals):,} sub-θ land-use fractions folded into dominant sources, "
        f"{folded_ha:,.1f} ha re-attributed (max single fraction {vals.max():.4f})",
        flush=True,
    )
    return base, fold_map


def get_folded_base_ag_dvar(data: Data, base_year: int) -> np.ndarray:
    """SOLVER-WORLD folded base ag dvar (NLMS, NCELLS, N_AG_LUS) — sub-θ slivers merged into the cell's
    dominant source. This is what the TRANSITION machinery (source maps, flow costs, ub/lb bounds,
    node-balance base) reads. See ``_fold_ag_dvar`` for the full fold semantics, and
    ``get_ag_dvar_fold_map`` for the sliver bookkeeping the accounting stream needs.
    """
    return _fold_ag_dvar(data, base_year)[0]


def get_ag_dvar_fold_map(data: Data, base_year: int) -> dict:
    """θ-fold sliver map (dict) consumed by the two-stream accounting to build X_acct — the true
    per-land-use composition — from the folded decision vars. Its arrays are empty when nothing folded.
    See ``_fold_ag_dvar`` for the field descriptions.
    """
    return _fold_ag_dvar(data, base_year)[1]


@lru_cache(maxsize=1)
def get_base_dvar_mj_cell_map(data: Data, base_year: int) -> dict:
    """Slice the FOLDED base ag dvar by each (from_m, from_j), returning {(from_m, from_j): cell_idx}
    for every source combo holding land in at least one cell.

    ★ THIS IS THE KEY DESIGN FOR THE DELTA TRANSITION COST. We slice the solver-world base dvar by the
    same source (from_m, from_j) — e.g. dry-Apples. All cells in one slice share the same transition
    costs (the cost of leaving dry-Apples for each target). The solver then creates, for each slice,
    the same number of delta variables (delta >= 0, positive-increment Gurobi vars) — one per sliced
    cell — that represent the TRUE transition flow out of that source on those cells. Transition cost
    is then trans_cost = delta_dvars * cost_cells, and the objective minimises sum(trans_cost).

    Sources come from `get_folded_base_ag_dvar` (sub-θ land-uses already folded into their dominant),
    so EVERY nonzero land-use is a source — the noise cutoff here only drops float dust, and no land
    is ever left without flow-to vars. θ's effect on model size acts through the folding, not through this
    slice. This map is the single source of truth for BOTH (a) the solver's per-source delta/cost
    slices and (b) target eligibility (`get_to_ag_exclude_matrices`), and ghg.py's exact transition-
    emission slicing reuses it, so all stay aligned.

    Cached (maxsize=1): the exclude builder and all exact cost functions call this for the same
    (data, base_year) pair within one solve step, so subsequent calls are free.
    """
    noise = 10 ** (-settings.ROUND_DECIMALS)
    base_dvar_mrj = get_folded_base_ag_dvar(data, base_year)   # folded dvar, for TRANSITION (solver world)
    return {
        (m, j): np.where(base_dvar_mrj[m, :, j] > noise)[0]
        for m in range(data.NLMS)
        for j in range(data.N_AG_LUS)
        if (base_dvar_mrj[m, :, j] > noise).any()
    }


def get_to_ag_exclude_matrices(data: Data, base_year: int) -> np.ndarray:
    """To-ag target-eligibility (exclude) matrix (NLMS, NCELLS, N_AG_LUS) via per-source reachability.

    Covers BOTH ag→ag and non-ag→ag reachability. A cell r is eligible for ag target tj iff:

    - SOME source present at r — an agricultural source from the FOLDED base (every nonzero folded
      land-use; sub-θ land is already absorbed into its dominant) or a non-agricultural source — can
      transition to tj (finite T_MAT entry); AND
    - tj is spatially allowed (EXCLUDE) and not a no-go LU there.

    Every ag land-use vouches for itself via the finite T_MAT diagonal, so (with a reconciled
    x_mrj) every held land-use always gets its own X var — no lb pin needed for var existence.

    Notes:

    - The present sources come from `get_base_dvar_mj_cell_map` (ag) and
      `get_base_nonag_dvar_k_cell_map` (nonag) — the SAME maps that drive the solver's per-source
      flow-out slices, so eligibility and the flow vars share one threshold.
    - `ag_lu2cells` then derives straight from this matrix and cannot diverge from the solver's
      direct `ag_x_mrj[m, r, j]` reads.
    """
    # Lazy import to avoid the agricultural <-> non_agricultural transitions import cycle.
    from luto.economics.non_agricultural.transitions import get_base_nonag_dvar_k_cell_map

    mj_cell_map = get_base_dvar_mj_cell_map(data, base_year)
    k_cell_map  = get_base_nonag_dvar_k_cell_map(data, base_year)

    # Binary T_MAT allow/disallow matrices (finite → True, NaN → False)
    t_ag2ag_jj = ~np.isnan(
        data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu=data.AGRICULTURAL_LANDUSES).values
    )   # (N_AG_LUS_from, N_AG_LUS_to)
    t_nonag2ag_kj = ~np.isnan(
        data.T_MAT.sel(from_lu=data.NON_AGRICULTURAL_LANDUSES, to_lu=data.AGRICULTURAL_LANDUSES).values
    )   # (N_NON_AG_LUS, N_AG_LUS_to)

    # Per-source reachability union: mark every target reachable from any present source at r.
    reach_rj = np.zeros((data.NCELLS, data.N_AG_LUS), dtype=bool)
    for (_fm, fj), cells in mj_cell_map.items():
        if cells.size:
            reach_rj[cells] |= t_ag2ag_jj[fj]
    for k, cells in k_cell_map.items():
        if cells.size:
            reach_rj[cells] |= t_nonag2ag_kj[k]

    t_rj = reach_rj.astype(np.int8)  # (NCELLS, N_AG_LUS)

    # Spatial exclusion and no-go zones
    x_mrj = data.EXCLUDE.copy().astype(np.int8)

    no_go_x_mrj = np.ones_like(data.AG_L_MRJ)
    if settings.EXCLUDE_NO_GO_LU:
        for no_go_x_r, no_go_desc in zip(data.NO_GO_REGION_AG, data.NO_GO_LANDUSE_AG):
            no_go_x_mrj[:, :, data.DESC2AGLU[no_go_desc]] = no_go_x_r

    return (x_mrj * t_rj[np.newaxis, :, :] * no_go_x_mrj).astype(np.int8)


def get_ag2ag_ub(data: Data, base_year: int) -> np.ndarray:
    """ag→ag TARGET upper bound (NLMS, NCELLS, N_AG_LUS), FRACTIONAL.

    `ub[to_m, r, to_j]` is the product of four factors:

    - T_MAT: binary allow/disallow per (from_j → to_j) — `finite → 1`, `NaN → 0`.
    - fraction: reachable land share = `Σ_{from_j : T_MAT[from_j→to_j] finite} Σ_m base_dvar[m, r, from_j]`
      (base-year fractions of every source LU that can reach to_j; overlapping sources summed, not OR-ed).
    - no-go: user-defined LUs banned in specific regions.
    - spatial exclusion (`data.EXCLUDE`): LU never present in the SA2 region in 2010 → banned there.

    Example: cell is 0.2 Apples + 0.8 (reachable LUs); if Apples↛to_j then ub[to_j] = 0.8.
    ag-source component only — nonag2ag adds its own fraction in the combined ag ub (later step).
    Uses the FOLDED base (get_folded_base_ag_dvar): reach follows solver-world source identity.
    """
    ag_dvar = get_folded_base_ag_dvar(data, base_year)                                                  # folded dvar, for TRANSITION reach (NLMS, NCELLS, N_AG_LUS)

    # Transition exclusion (T_MAT): binary allow/disallow per (from_j → to_j).
    t_ag2ag_jj = (~np.isnan(
        data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu=data.AGRICULTURAL_LANDUSES).values
    )).astype(np.float32)                                                                              # (from_j, to_j)
    
    # Reachable land share: sum the base-year fractions of every source LU that can reach to_j.
    ag_frac_rj    = ag_dvar.sum(axis=0)                                                                # (NCELLS, from_j)
    reach_frac_rj = (ag_frac_rj @ t_ag2ag_jj).astype(np.float32)                                       # (NCELLS, to_j)

    # No-go exclusion: user-defined LUs banned in specific regions.
    no_go = np.ones((data.NLMS, data.NCELLS, data.N_AG_LUS), dtype=np.float32)
    if settings.EXCLUDE_NO_GO_LU:
        for no_go_x_r, no_go_desc in zip(data.NO_GO_REGION_AG, data.NO_GO_LANDUSE_AG):
            no_go[:, :, data.DESC2AGLU[no_go_desc]] = no_go_x_r

    # Spatial exclusion (data.EXCLUDE): LU never present in the SA2 region in 2010 → banned there.
    x_mrj = data.EXCLUDE.astype(np.float32)
    return (x_mrj * reach_frac_rj[np.newaxis, :, :] * no_go).astype(np.float32)


def get_ag2ag_lb(data: Data, base_year: int) -> np.ndarray:
    """ag→ag TARGET lower bound (NLMS, NCELLS, N_AG_LUS): all zeros.

    The old sub-θ sliver 'stay' pin is gone — fold-into-dominant (get_folded_base_ag_dvar) absorbs
    every sub-θ land-use into the cell's dominant source BEFORE the solver world is built, so no
    land-use is ever left without flow-to vars or without an X var, and nothing needs to be locked in place.
    Kept as a function (rather than deleted) as the hook for any future genuine ag lower bound.
    """
    return np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS), dtype=np.float32)


def get_transition_matrices_ag2ag(data: Data, yr_idx: int, from_m: int, from_j: int, cells=None, separate=False):
    """Source-parameterised ag2ag transition-cost primitive.

    Answers ONE question: transitioning FROM a single source (from_m, from_j) TO every target
    (to_m, to_j), on `cells` (default all NCELLS), what is the cost/water/GHG per cell? Returns
    (NLMS, len(cells), N_AG_LUS) [to_m, r, to_j] (separate=True → {component: same-shape array}).

    UNMASKED — no exclude / no "not-staying" mask. The diagonal ((to_m,to_j)==(from_m,from_j)) is
    naturally 0 (T_MAT[j→j]=0, zero water-delta, zero GHG), and target eligibility is the solver's job
    (a flow/delta var is created only for a valid transition).

    This is THE per-source cost primitive of the delta transition model. Callers select which cells to
    evaluate per source: the ag2ag/ag2nonag cost builds it on each source's dvar>θ cells
    (`get_base_dvar_mj_cell_map`, θ = EXACT_REACHABILITY_MIN_FRACTION); the nonag→ag levers call it for a
    uniform livestock source (cells=None → all cells). The year-invariant water / T_MAT terms are
    derived internally each call (water via the lru_cached `get_wreq_matrices`, so per-source calls in
    a dict loop stay a single compute per step).

    Components:
      - Establishment  : amortise(T_MAT[from_j → to_j] × mult) × REAL_AREA  (both target lm equal)
      - Water license  : amortised [(target req − source (from_m,from_j) req) × price + irrigation setup/teardown]
      - GHG (× carbon price) : natural→modified / unallocated-natural→livestock-natural release
    """
    yr_cal = data.YR_CAL_BASE + yr_idx
    if cells is None:
        cells = np.arange(data.NCELLS)
    n = len(cells)
    N_AG = data.N_AG_LUS
    area = data.REAL_AREA[cells]                                                    # (n,)

    # ── Establishment: source only enters via the T_MAT[from_j] row ──
    t_ij  = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu=data.AGRICULTURAL_LANDUSES).values * data.TRANS_COST_MULTS[yr_cal]
    t_row = np.nan_to_num(t_ij[from_j]).astype(np.float32)                          # (N_AG,)
    e_rj  = tools.amortise(np.tile(t_row, (n, 1))) * area[:, None]                  # (n, N_AG)
    e_mrj = np.stack([e_rj, e_rj], axis=0).astype(np.float32)                       # (NLMS, n, N_AG)

    # ── Water licence delta (source-parameterised), amortised like Establishment above ──
    w_mrj       = get_wreq_matrices(data, yr_idx)                                   # <ML/cell> (lru_cached)
    w_raw_mrj   = tools.get_ag_to_ag_water_delta_matrix(data, from_m, from_j, cells, w_mrj, yr_idx)
    w_delta_mrj = tools.amortise(w_raw_mrj).astype(np.float32)

    # ── GHG release ($ = carbon price × raw emissions, amortised): source-parameterised ──
    price   = data.get_carbon_price_by_yr_idx(yr_idx)
    ghg_raw = ag_ghg.get_ghg_transition_emissions(data, from_m, from_j, cells, separate=True)   # raw t/cell
    ghg     = {k: tools.amortise(v * price).astype(np.float32) for k, v in ghg_raw.items()}

    if separate:
        return {'Establishment cost': e_mrj, 'Water license cost': w_delta_mrj, **ghg}
    return (e_mrj + w_delta_mrj + sum(ghg.values())).astype(np.float32)


def get_asparagopsis_effect_t_mrj(data: Data):
    """
    Gets the transition costs of asparagopsis taxiformis, which are none.
    Transition/establishment costs are handled in the costs matrix.
    """
    land_uses = settings.AG_MANAGEMENTS_TO_LAND_USES["Asparagopsis taxiformis"]
    return np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)


def get_precision_agriculture_effect_t_mrj(data: Data):
    """
    Gets the effects on transition costs of precision agriculture, which are none.
    Transition/establishment costs are handled in the costs matrix.
    """
    land_uses = settings.AG_MANAGEMENTS_TO_LAND_USES['Precision Agriculture']
    return np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)


def get_ecological_grazing_effect_t_mrj(data: Data):
    """
    Gets the effects on transition costs of ecological grazing, which are none.
    Transition/establishment costs are handled in the costs matrix.
    """
    land_uses = settings.AG_MANAGEMENTS_TO_LAND_USES['Ecological Grazing']
    return np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)


def get_savanna_burning_effect_t_mrj(data: Data):
    """
    Gets the effects on transition costs of savanna burning, which are none.
    Transition/establishment costs are handled in the costs matrix.
    """
    land_uses = settings.AG_MANAGEMENTS_TO_LAND_USES['Savanna Burning']
    return np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)


def get_agtech_ei_effect_t_mrj(data: Data):
    """
    Gets the effects on transition costs of AgTech EI, which are none.
    Transition/establishment costs are handled in the costs matrix.
    """
    land_uses = settings.AG_MANAGEMENTS_TO_LAND_USES['AgTech EI']
    return np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)


def get_biochar_effect_t_mrj(data: Data):
    """
    Gets the effects on transition costs of Biochar, which are none.
    Transition/establishment costs are handled in the costs matrix.
    """
    land_uses = settings.AG_MANAGEMENTS_TO_LAND_USES['Biochar']
    return np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)


def get_beef_hir_effect_t_mrj(data: Data):
    land_uses = settings.AG_MANAGEMENTS_TO_LAND_USES['HIR - Beef']
    return np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)


def get_sheep_hir_effect_t_mrj(data: Data):
    land_uses = settings.AG_MANAGEMENTS_TO_LAND_USES['HIR - Sheep']
    return np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)


def get_utility_solar_pv_effect_t_mrj(data: Data, yr_idx):
    """
    Returns zeros — CAPEX for Utility Solar PV has been moved to
    get_utility_solar_pv_effect_c_mrj (cost.py) as an amortised annual cost.
    """
    solar_lus = settings.AG_MANAGEMENTS_TO_LAND_USES['Utility Solar PV']
    return np.zeros((data.NLMS, data.NCELLS, len(solar_lus)), dtype=np.float32)


def get_onshore_wind_effect_t_mrj(data: Data, yr_idx):
    """
    Returns zeros — CAPEX for Onshore Wind has been moved to
    get_onshore_wind_effect_c_mrj (cost.py) as an amortised annual cost.
    """
    wind_lus = settings.AG_MANAGEMENTS_TO_LAND_USES['Onshore Wind']
    return np.zeros((data.NLMS, data.NCELLS, len(wind_lus)), dtype=np.float32)


def get_agricultural_management_transition_matrices(data: Data, yr_idx) -> Dict[str, np.ndarray]:
    
    asparagopsis_data = get_asparagopsis_effect_t_mrj(data)                     
    precision_agriculture_data = get_precision_agriculture_effect_t_mrj(data)   
    eco_grazing_data = get_ecological_grazing_effect_t_mrj(data)                
    sav_burning_data = get_savanna_burning_effect_t_mrj(data)                   
    agtech_ei_data = get_agtech_ei_effect_t_mrj(data)                           
    biochar_data = get_biochar_effect_t_mrj(data)                               
    beef_hir_data = get_beef_hir_effect_t_mrj(data)                             
    sheep_hir_data = get_sheep_hir_effect_t_mrj(data)
    utility_solar_pv_data = get_utility_solar_pv_effect_t_mrj(data, yr_idx)
    onshore_wind_data = get_onshore_wind_effect_t_mrj(data, yr_idx)
                  
    return {
        'Asparagopsis taxiformis': asparagopsis_data,
        'Precision Agriculture': precision_agriculture_data,
        'Ecological Grazing': eco_grazing_data,
        'Savanna Burning': sav_burning_data,
        'AgTech EI': agtech_ei_data,
        'Biochar': biochar_data,
        'HIR - Beef': beef_hir_data,
        'HIR - Sheep': sheep_hir_data,
        'Utility Solar PV': utility_solar_pv_data,
        'Onshore Wind': onshore_wind_data
    }


def get_asparagopsis_adoption_limits(data: Data, yr_idx):
    """
    Gets the adoption limit of Asparagopsis taxiformis for each possible land use.
    """
    if not settings.AG_MANAGEMENTS['Asparagopsis taxiformis']:
        return {data.DESC2AGLU[lu]: 0 for lu in settings.AG_MANAGEMENTS_TO_LAND_USES['Asparagopsis taxiformis']}
    
    asparagopsis_limits = {}
    yr_cal = data.YR_CAL_BASE + yr_idx
    for lu in settings.AG_MANAGEMENTS_TO_LAND_USES['Asparagopsis taxiformis']:
        j = data.DESC2AGLU[lu]
        asparagopsis_limits[j] = min(data.ASPARAGOPSIS_DATA[lu].loc[yr_cal, 'Technical_Adoption'] * settings.TECH_ADOPT_MULT, 1)

    return asparagopsis_limits


def get_precision_agriculture_adoption_limit(data: Data, yr_idx):
    """
    Gets the adoption limit of precision agriculture for each possible land use.
    """
    if not settings.AG_MANAGEMENTS['Precision Agriculture']:
        return {data.DESC2AGLU[lu]: 0 for lu in settings.AG_MANAGEMENTS_TO_LAND_USES['Precision Agriculture']}
    
    prec_agr_limits = {}
    yr_cal = data.YR_CAL_BASE + yr_idx
    for lu in settings.AG_MANAGEMENTS_TO_LAND_USES['Precision Agriculture']:
        j = data.DESC2AGLU[lu]
        prec_agr_limits[j] = min(data.PRECISION_AGRICULTURE_DATA[settings.LU2TYPE[lu]].loc[yr_cal, 'Technical_Adoption'] * settings.TECH_ADOPT_MULT, 1)

    return prec_agr_limits


def get_ecological_grazing_adoption_limit(data: Data, yr_idx):
    """
    Gets the adoption limit of ecological grazing for each possible land use.
    """
    if not settings.AG_MANAGEMENTS['Ecological Grazing']:
        return {data.DESC2AGLU[lu]: 0 for lu in settings.AG_MANAGEMENTS_TO_LAND_USES['Ecological Grazing']}
    
    eco_grazing_limits = {}
    yr_cal = data.YR_CAL_BASE + yr_idx
    for lu in settings.AG_MANAGEMENTS_TO_LAND_USES['Ecological Grazing']:
        j = data.DESC2AGLU[lu]
        eco_grazing_limits[j] = data.ECOLOGICAL_GRAZING_DATA[lu].loc[yr_cal, 'Feasible Adoption (%)']

    return eco_grazing_limits


def get_savanna_burning_adoption_limit(data: Data):
    """
    Gets the adoption limit of Savanna Burning for each possible land use
    """
    if not settings.AG_MANAGEMENTS['Savanna Burning']:
        return {data.DESC2AGLU[lu]: 0 for lu in settings.AG_MANAGEMENTS_TO_LAND_USES['Savanna Burning']}
    
    sav_burning_limits = {}
    for lu in settings.AG_MANAGEMENTS_TO_LAND_USES['Savanna Burning']:
        j = data.DESC2AGLU[lu]
        sav_burning_limits[j] = 1

    return sav_burning_limits


def get_agtech_ei_adoption_limit(data: Data, yr_idx):
    """
    Gets the adoption limit of AgTech EI for each possible land use.
    """
    if not settings.AG_MANAGEMENTS['AgTech EI']:
        return {data.DESC2AGLU[lu]: 0 for lu in settings.AG_MANAGEMENTS_TO_LAND_USES['AgTech EI']}
    
    agtech_ei_limits = {}
    yr_cal = data.YR_CAL_BASE + yr_idx
    for lu in settings.AG_MANAGEMENTS_TO_LAND_USES['AgTech EI']:
        j = data.DESC2AGLU[lu]
        agtech_ei_limits[j] = min(data.AGTECH_EI_DATA[settings.LU2TYPE[lu]].loc[yr_cal, 'Technical_Adoption'] * settings.TECH_ADOPT_MULT, 1)

    return agtech_ei_limits


def get_biochar_adoption_limit(data: Data, yr_idx):
    """
    Gets the adoption limit of Biochar for each possible land use.
    """
    if not settings.AG_MANAGEMENTS['Biochar']:
        return {data.DESC2AGLU[lu]: 0 for lu in settings.AG_MANAGEMENTS_TO_LAND_USES['Biochar']}
    
    biochar_limits = {}
    yr_cal = data.YR_CAL_BASE + yr_idx
    for lu in settings.AG_MANAGEMENTS_TO_LAND_USES['Biochar']:
        j = data.DESC2AGLU[lu]
        biochar_limits[j] = min(data.BIOCHAR_DATA[settings.LU2TYPE[lu]].loc[yr_cal, 'Technical_Adoption'] * settings.TECH_ADOPT_MULT, 1)

    return biochar_limits


def get_beef_hir_adoption_limit(data: Data):
    """
    Gets the adoption limit of HIR - Beef for each possible land use.
    """
    if not settings.AG_MANAGEMENTS['HIR - Beef']:
        return {data.DESC2AGLU[lu]: 0 for lu in settings.AG_MANAGEMENTS_TO_LAND_USES['HIR - Beef']}
    hir_limits = {}
    for lu in settings.AG_MANAGEMENTS_TO_LAND_USES['HIR - Beef']:
        j = data.DESC2AGLU[lu]
        hir_limits[j] = 1

    return hir_limits


def get_sheep_hir_adoption_limit(data: Data):
    """
    Gets the adoption limit of HIR - Sheep for each possible land use.
    """
    if not settings.AG_MANAGEMENTS['HIR - Sheep']:
        return {data.DESC2AGLU[lu]: 0 for lu in settings.AG_MANAGEMENTS_TO_LAND_USES['HIR - Sheep']}
    hir_limits = {}
    for lu in settings.AG_MANAGEMENTS_TO_LAND_USES['HIR - Sheep']:
        j = data.DESC2AGLU[lu]
        hir_limits[j] = 1

    return hir_limits

def get_utility_solar_pv_adoption_limit(data: Data):
    """
    Gets the adoption limit of Utility Solar PV for each possible land use.
    """
    if not settings.AG_MANAGEMENTS['Utility Solar PV']:
        return {data.DESC2AGLU[lu]: 0 for lu in settings.AG_MANAGEMENTS_TO_LAND_USES['Utility Solar PV']}
    solar_pv_limits = {}
    for lu in settings.AG_MANAGEMENTS_TO_LAND_USES['Utility Solar PV']:
        j = data.DESC2AGLU[lu]
        solar_pv_limits[j] = settings.RENEWABLES_ADOPTION_LIMITS['Utility Solar PV']

    return solar_pv_limits

def get_onshore_wind_adoption_limit(data: Data):
    """
    Gets the adoption limit of Onshore Wind for each possible land use.
    """
    if not settings.AG_MANAGEMENTS['Onshore Wind']:
        return {data.DESC2AGLU[lu]: 0 for lu in settings.AG_MANAGEMENTS_TO_LAND_USES['Onshore Wind']}
    
    wind_limits = {}
    for lu in settings.AG_MANAGEMENTS_TO_LAND_USES['Onshore Wind']:
        j = data.DESC2AGLU[lu]
        wind_limits[j] = settings.RENEWABLES_ADOPTION_LIMITS['Onshore Wind']

    return wind_limits

def get_agricultural_management_adoption_limits(data: Data, yr_idx) -> Dict[str, dict]:
    """
    An adoption limit represents the maximum percentage of cells (for each land use) that can utilise
    each agricultural management option.
    """
    ag_management_data = {}

    ag_management_data['Asparagopsis taxiformis'] = get_asparagopsis_adoption_limits(data, yr_idx)
    ag_management_data['Precision Agriculture'] = get_precision_agriculture_adoption_limit(data, yr_idx)
    ag_management_data['Ecological Grazing'] = get_ecological_grazing_adoption_limit(data, yr_idx)
    ag_management_data['Savanna Burning'] = get_savanna_burning_adoption_limit(data)
    ag_management_data['AgTech EI'] = get_agtech_ei_adoption_limit(data, yr_idx)
    ag_management_data['Biochar'] = get_biochar_adoption_limit(data, yr_idx)
    ag_management_data['HIR - Beef'] = get_beef_hir_adoption_limit(data)
    ag_management_data['HIR - Sheep'] = get_sheep_hir_adoption_limit(data)
    ag_management_data['Utility Solar PV'] = get_utility_solar_pv_adoption_limit(data)
    ag_management_data['Onshore Wind'] = get_onshore_wind_adoption_limit(data)
   
    return ag_management_data


def get_lower_bound_agricultural_management_matrices(data: Data, base_year) -> dict[str, dict]:
    """
    Returns per-am lower bounds (shape: NLMS × NCELLS × N_AG_LUS) for the next solve.

    Each am_lb[am][m, r, j] is the floor-truncated min(am_dvar, ag_dvar).  The clamp
    against ag_dvar corrects for FeasibilityTol: the solver enforces am ≤ ag as a linear
    constraint (not a variable bound), so the reported am_dvar can exceed ag_dvar by up to
    FeasibilityTol.  The correct "true" am value is min(am_dvar, ag_dvar) — equivalent to
    what a variable upper-bound on ag_dvar would have enforced exactly.
    """

    if base_year == data.YR_CAL_BASE or base_year not in data.ag_man_dvars:
        return {
            am: np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS), dtype=np.float32)
            for am in settings.AG_MANAGEMENTS_TO_LAND_USES
            if settings.AG_MANAGEMENTS[am]
        }

    # FOLDED base: am lb must not exceed the SOLVER-WORLD host ag land-use (an am fraction whose host
    # sliver was folded away is clamped to 0 — its land now lives under the dominant source).
    ag_dvar = get_folded_base_ag_dvar(data, base_year)     # folded dvar, for TRANSITION (NLMS, NCELLS, N_AG_LUS)

    result = {}
    for am in settings.AG_MANAGEMENTS_TO_LAND_USES:
        if not settings.AG_MANAGEMENTS[am]:
            continue
        am_dvar = data.ag_man_dvars[base_year][am].astype(np.float32)
        am_dvar_true = tools.clamp_dvar_bound(am_dvar, 0.0, ag_dvar, f'Ag man lb clamped [{am}]')   # am cannot exceed its host ag
        am_lb = np.divide(
            np.floor(am_dvar_true * 10 ** settings.ROUND_DECIMALS),
            10 ** settings.ROUND_DECIMALS,
        )
        result[am] = am_lb
    return result


def get_regional_adoption_limits(data: Data, yr_cal: int):
    """
    Build per-region adoption caps for the solver.

    Returns
    -------
    ag_reg_adoption_constrs : list[[reg_id, lu_code, lu_name, reg_ind, area_limit_ha]]
        Per-(region, ag-landuse) caps from regional_adoption_zones.xlsx. 'on' mode only.
    non_ag_reg_adoption_constrs : list[[reg_id, lu_code, lu_name, reg_ind, area_limit_ha]]
        Per-(region, non-ag-landuse) caps from regional_adoption_zones.xlsx. 'on' mode only.
    non_ag_reg_adoption_sum_constrs : list[[reg_id, reg_ind, area_limit_ha]]
        Per-region SUM-of-all-non-ag caps. 'NON_AG_CAP' mode only.
    """
    if settings.REGIONAL_ADOPTION_CONSTRAINTS == "off":
        return [], [], []

    ag_reg_adoption_constrs = []
    non_ag_reg_adoption_constrs = []

    # Per-LU (ag + non-ag) caps from xlsx — only populated in 'on' mode
    for reg_id, lu_name, area_limit_ha in data.get_regional_adoption_limit_ha_by_year(yr_cal):
        reg_ind = np.where(data.REGIONAL_ADOPTION_ZONES == reg_id)[0]

        if lu_name in data.DESC2AGLU:
            lu_code = data.DESC2AGLU[lu_name]
            ag_reg_adoption_constrs.append([reg_id, lu_code, lu_name, reg_ind, area_limit_ha])
        elif lu_name in data.DESC2NONAGLU:
            lu_code = data.DESC2NONAGLU[lu_name] - settings.NON_AGRICULTURAL_LU_BASE_CODE
            non_ag_reg_adoption_constrs.append([reg_id, lu_code, lu_name, reg_ind, area_limit_ha])
        else:
            raise ValueError(f"Regional adoption constraint exists for unrecognised land use: {lu_name}")

    # SUM-of-non-ag per-region cap — only populated in 'NON_AG_CAP' mode
    non_ag_reg_adoption_sum_constrs = [
        [reg_id, reg_ind, area_limit_ha]
        for reg_id, reg_ind, area_limit_ha
        in data.get_regional_adoption_non_ag_sum_limit_ha_by_year(yr_cal)
    ]

    return ag_reg_adoption_constrs, non_ag_reg_adoption_constrs, non_ag_reg_adoption_sum_constrs
