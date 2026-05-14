# GBF2 — Priority Degraded Areas Restoration

Global Biodiversity Framework Target 2 requires that by 2030 at least 30% of degraded terrestrial ecosystems be under effective restoration. LUTO implements this as a spatially-explicit constraint over cells identified as **priority degraded areas**.

---

## 1. Core Concepts

### 1.1 Pre-1750 baseline

The GBF framework measures degradation relative to a pre-1750 biodiversity baseline — the state before European land clearing. A cell at full (pre-1750) condition has a biodiversity score of 1.0; a cell converted to intensive cropping might retain only 0.27 (27% of pre-1750 condition). The remaining 0.73 is the **degradation**.

### 1.2 Priority degraded areas (the GBF2 mask)

Not all degraded cells are targeted. GBF2 focuses on **priority** cells: those that matter most to overall biodiversity benefit if restored. LUTO identifies them using a Zonation-derived **conservation performance curve**.

The curve maps the cumulative area percentage (ranked from highest to lowest biodiversity quality) to the cumulative biodiversity benefit percentage. By choosing a cut at `GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT = 20` (the top 20% of biodiversity-weighted area), LUTO selects the cells that contribute disproportionately high biodiversity returns per hectare restored.

**Mask construction** (`data.py`):

```python
# Raw Zonation-derived biodiversity quality score per cell
bio_quality_raw = ...   # shape [NCELLS], range ~0-1

# Threshold from the conservation performance curve at the chosen cut
threshold = conservation_performance_curve[GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT]

# Boolean mask: True for cells in the top-priority zone
BIO_GBF2_MASK = bio_quality_raw >= threshold   # shape [NCELLS]
```

`GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT = 20` — cells in the top 20% of the performance curve.  
Setting it to 0 disables the mask entirely; 100 applies it to all LUTO cells.

### 1.3 Biodiversity contribution per land use

Each agricultural land use has a **bio contribution** score (0–1) from the HCAS (Habitat Condition Assessment System) dataset, representing the fraction of pre-1750 biodiversity retained under that land use. Non-agricultural land uses (Environmental Plantings, etc.) and agricultural management options have their own contribution values.

```python
BIO_HABITAT_CONTRIBUTION_LOOK_UP = {j: bio_contribution_j, ...}
# e.g. Apples: 0.27, Beef - natural land: 0.72, Unallocated - natural land: 1.0
```

### 1.4 Savanna burning (LDS correction)

Cells eligible for savanna burning (`SAVBURN_ELIGIBLE`) operate under a **late dry season (LDS)** fire regime by default. LDS fires suppress biodiversity relative to the ideal early dry season (EDS) fire regime. The `BIO_CONTRIBUTION_LDS = 0.8` factor means a savburn cell only contributes 80% of its raw biodiversity value under the default regime. When EDS savanna burning is applied as an agricultural management, the full value is recovered.

This correction is baked into `BIO_GBF2_BASE_YR`:

```python
BIO_GBF2_BASE_YR[r] = (
    sum_j sum_m bio_contr[j] * AG_L_MRJ[m,r,j] * BIO_GBF2_MASK[r] * REAL_AREA[r]
    - SAVBURN_ELIGIBLE[r] * BIO_GBF2_MASK[r] * (1 - BIO_CONTRIBUTION_LDS) * REAL_AREA[r] * AG_MASK_PROPORTION_R[r]
)
```

The second term subtracts the LDS penalty from cells where savanna burning applies at the base year — lowering the effective base-year score, which in turn raises the restoration target.

---

## 2. Target Interpolation

`data.get_GBF2_target_for_yr_cal(yr_cal)` returns the absolute biodiversity score the solver must reach at `yr_cal`.

### Quantities

| Symbol | Definition | Code |
|--------|-----------|------|
| **baseline** | Maximum achievable score (all priority cells at pre-1750 condition) | `(BIO_GBF2_MASK * REAL_AREA * AG_MASK_PROPORTION_R).sum()` |
| **base_yr** | Actual score in 2010 under observed land use, including LDS correction | `BIO_GBF2_BASE_YR.sum()` |
| **degradation** | Gap between baseline and base year | `baseline − base_yr` |

### Target schedule

`GBF2_TARGETS_DICT` maps the scenario name to key-year restoration fractions:

```python
GBF2_TARGETS_DICT = {
    'off':    None,
    'low':    {2030: 0,    2050: 0,    2100: 0},
    'medium': {2030: 0.30, 2050: 0.30, 2100: 0.30},
    'high':   {2030: 0.30, 2050: 0.50, 2100: 0.50},
}
```

For `BIODIVERSITY_TARGET_GBF_2 = 'high'` the model must restore 30% of the degradation by 2030 and 50% by 2050 and 2100.

### Interpolation formula

```
base_yr_proportion  = base_yr / baseline

target_proportion(t) = base_yr_proportion
                      + (1 − base_yr_proportion) × restoration_fraction(t)

target(yr_cal) = baseline × interp(yr_cal, key_years, target_proportions)
```

This ensures:
- At 2010 the target equals `base_yr` (no immediate restoration required).
- At 2030 the target equals `base_yr + restoration_fraction × degradation`.
- Intermediate years are linearly interpolated.

---

## 3. Solver Constraint

**File:** `luto/solvers/solver.py` — `_add_GBF2_constraints()`

The constraint forces the total area-weighted biodiversity score across all priority cells to meet or exceed the interpolated target.

### Score expression

```
score = Σ_r Σ_j  GBF2_area[r] × bio_contr[j] × X_ag[j,r]   (agricultural)
      + Σ_r Σ_am Σ_j  GBF2_area[r] × bio_contr_am[am,j,r] × X_am[am,j,r]   (ag management)
      + Σ_r Σ_k  GBF2_area[r] × bio_contr_nonag[k] × X_nonag[k,r]   (non-agricultural)
```

where `GBF2_area[r] = BIO_GBF2_MASK[r] × REAL_AREA[r]` (hectares inside the priority mask).

The constraint is:

```
score ≥ target(yr_cal)          [hard constraint]
```

or, when `GBF2_CONSTRAINT_TYPE = 'soft'`, expressed as a soft goal-programming slack.

Cells are pre-filtered to `GBF2_mask_idx = where(BIO_GBF2_MASK_LDS)` — only cells within the priority mask (with the LDS adjustment applied) enter the constraint. This keeps the LP matrix sparse.

---

## 4. Reporting: `Relative_Contribution_Percentage`

**File:** `luto/tools/write.py` — `write_biodiversity_GBF2_scores()`

The function decomposes the solver's aggregate target into per–(land-use, land-management, NRM region) contributions and expresses each as a percentage of the **total degradation**. The sum of non-ALL rows for Australia equals the fraction of degradation that has been restored at `yr_cal`.

### Denominator — total degradation at base year

```python
degreded_area_weighted_bio_contr = (
    (data.BIO_GBF2_MASK * data.REAL_AREA * data.AG_MASK_PROPORTION_R).sum()   # baseline
    - data.BIO_GBF2_BASE_YR.sum()                                              # base_yr
)
```

This is identical to the `degradation` used in `get_GBF2_target_for_yr_cal`, ensuring the denominator is consistent with the solver's target calculation. (An earlier formulation used `(BASEYEAR_dvar × GBF2_area × (1 − bio_contr)).sum()`, which was ~3% smaller because it excluded the savanna-burning LDS correction.)

### Numerator — change in score from 2010

**Agricultural land uses (ag):**

```python
xr_gbf2_ag = priority_degraded_area_score_r * (
    ag_impact_j * (ag_dvar_mrj − BASEYEAR_dvar)          # change in ag score
    + savburn_eligible_r * (1 − BIO_CONTRIBUTION_LDS) * BASEYEAR_dvar   # LDS correction
)
```

- `priority_degraded_area_score_r = BIO_GBF2_MASK × REAL_AREA` per cell.
- `ag_dvar_mrj` is the solver's land-use allocation at `yr_cal`; `BASEYEAR_dvar` is the 2010 allocation. Both are masked at cells where the total allocation is < 1% (threshold 0.01), preventing masking-induced bias in the subtraction.
- The **savanna-burning correction** adds back the LDS penalty embedded in `BIO_GBF2_BASE_YR`. Without it, cells where savburn applies would appear to have been "degraded" more than the solver acknowledges, causing the reported sum to fall ~3 pp short of the restoration target.

**Non-agricultural land uses (non-ag):**

```python
xr_gbf2_non_ag = priority_degraded_area_score_r * non_ag_impact_k * non_ag_dvar_rk
```

No base-year subtraction because there is no non-ag allocation in 2010.

**Agricultural management (am):**

```python
xr_gbf2_am = priority_degraded_area_score_r * am_impact_ajr * am_dvar_amrj
```

Same reasoning — am options are absent at 2010.

### Relative_Contribution_Percentage

```python
Relative_Contribution_Percentage = (Area_Weighted_Score / degreded_area_weighted_bio_contr) * 100
```

For a hard constraint that is tight at `yr_cal`, the sum of non-ALL rows for the AUSTRALIA region across all three types equals the restoration target fraction. For `BIODIVERSITY_TARGET_GBF_2 = 'high'` at 2030 this should be ≈ 30%.

### Exclusions when summing

To avoid double-counting the `add_all`-generated aggregate rows:

| Type | Exclude rows where |
|------|--------------------|
| Agricultural land-use | `lm = 'ALL'` **or** `lu = 'ALL'` |
| Non-agricultural land-use | `lu = 'ALL'` |
| Agricultural management | `am = 'ALL'` **or** `lm = 'ALL'` **or** `lu = 'ALL'` |

---

## 5. Settings Reference

| Setting | Default | Effect |
|---------|---------|--------|
| `BIODIVERSITY_TARGET_GBF_2` | `'high'` | Scenario: `'off'`, `'low'`, `'medium'`, `'high'` |
| `GBF2_CONSTRAINT_TYPE` | `'hard'` | Hard or soft solver constraint |
| `GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT` | `20` | Top-N% of Zonation performance curve used as the priority mask |
| `GBF2_TARGETS_DICT` | see above | Restoration fractions at key years per scenario |
| `BIO_QUALITY_LAYER` | `'MNES_likely'` | Zonation quality layer used for both priority masking and bio-contribution scores |
| `CONTRIBUTION_PERCENTILE` | `'USER_DEFINED'` | HCAS percentile or mode for bio-contribution lookup |
| `BIO_CONTRIBUTION_LDS` | `0.8` | Biodiversity value retained under default LDS savanna fire regime |
| `CONNECTIVITY_SOURCE` | `'NCI'` | Source of spatial connectivity weights applied to quality scores |
| `CONNECTIVITY_LB` | `0.7` | Lower bound of the rescaled connectivity multiplier |

---

## 6. Data Flow Summary

```
settings.py
  GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT
  GBF2_TARGETS_DICT / BIODIVERSITY_TARGET_GBF_2
  BIO_CONTRIBUTION_LDS
          │
          ▼
data.py   (load phase)
  BIO_GBF2_MASK          ← bio_quality_raw ≥ conservation_performance_curve[cut]
  BIO_GBF2_BASE_YR       ← einsum(bio_contr, AG_L_MRJ, MASK, REAL_AREA) − LDS_correction
  BIO_HABITAT_CONTRIBUTION_LOOK_UP  ← HCAS lookup per land-use code
          │
          ├──► get_GBF2_target_for_yr_cal(yr_cal)
          │      baseline = (MASK × AREA × AG_MASK_PROP).sum()
          │      target   = BASE_YR + frac(yr_cal) × (baseline − BASE_YR)
          │
          ▼
input_data.py (pre-solve)
  GBF2_mask_area_r  = BIO_GBF2_MASK × REAL_AREA    (no AG_MASK_PROPORTION_R)
  GBF2_mask_idx     = where(BIO_GBF2_MASK_LDS)
  limits["GBF2"]    = get_GBF2_target_for_yr_cal(yr_cal)
          │
          ▼
solver.py  _add_GBF2_constraints()
  score = Σ GBF2_area[r] × bio_contr[j] × X[j,r]   (ag + am + non-ag)
  score ≥ limits["GBF2"] / scale_factor              (hard constraint)
          │
          ▼
write.py  write_biodiversity_GBF2_scores()
  denom  = baseline − BIO_GBF2_BASE_YR.sum()
  ag     = MASK×AREA × (bio_contr×(dvar_yr − base_dvar) + savburn×(1−LDS)×base_dvar)
  non_ag = MASK×AREA × non_ag_bio_contr × non_ag_dvar
  am     = MASK×AREA × am_bio_contr × am_dvar
  Relative_Contribution_Percentage = component / denom × 100
  → CSV: biodiversity_GBF2_priority_scores_{yr_cal}.csv
```
