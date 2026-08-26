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



""" LUTO model settings. """



# ---------------------------------------------------------------------------- #
# LUTO model version.                                                          #
# ---------------------------------------------------------------------------- #

VERSION = '2.4'


# ---------------------------------------------------------------------------- #
# Directories.                                                                 #
# ---------------------------------------------------------------------------- #

INPUT_DIR = 'input'
OUTPUT_DIR = 'output'
RAW_DATA = '../raw_data'


# ---------------------------------------------------------------------------- #
# Scenario parameters.                                                         #
# ---------------------------------------------------------------------------- #

# Climate change assumptions. Options include '126', '245', '370', '585'
SSP = '245'
RCP = 'rcp' + SSP[1] + 'p' + SSP[2] # Representative Concentration Pathway string identifier e.g., 'rcp4p5'.

# Set demand parameters which define requirements for Australian production of agricultural commodities
SCENARIO        = 'SSP' + SSP[0]     # SSP1, SSP2, SSP3, SSP4, SSP5
DIET_DOM        = 'BAU'              # 'BAU', 'FLX', 'VEG', 'VGN' - domestic diets in Australia
DIET_GLOB       = 'BAU'              # 'BAU', 'FLX', 'VEG', 'VGN' - global diets
CONVERGENCE     = 2050               # 2050 or 2100 - date at which dietary transformation is completed (velocity of transformation)
IMPORT_TREND    = 'Trend'            # 'Static' (assumes 2010 shares of imports for each commodity) or 'Trend' (follows historical rate of change in shares of imports for each commodity)
WASTE           = 1                  # 1 for full waste, 0.5 for half waste
FEED_EFFICIENCY = 'BAU'              # 'BAU' or 'High'

# Set the demand and supply multipliers
APPLY_DEMAND_MULTIPLIERS = True     # True or False. Whether to apply demand multipliers from AusTIME model.

# Productivity trend; 
PRODUCTIVITY_TREND = 'BAU'           # 'BAU', 'LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH'


# Add CO2 fertilisation effects on agricultural production from GAEZ v4
CO2_FERT = 'off'   # 'on' or 'off'

# Number of years over which to spread (average) soil carbon accumulation (from Mosnier et al. 2022 and Johnson et al. 2021)
CARBON_EFFECTS_WINDOW = 60 # 50, 60, 70, 80, or 90
'''
Available options are  50, 60, 70, 80, 90 years. This is the number of years over which to spread (average) 
soil carbon accumulation. 

The logic is that carbon accumulation after tree planting over time follows an S-shaped curve,
with rapid accumulation in the first few decades, then slowing down as it approaches a new equilibrium.

For example, by setting the CARBON_EFFECTS_WINDOW to 50 years, LUTO will take the total co2 sequestration
for the first 50 years after planting and then use the average as the annual sequestration rate in the model.
'''


# Fire impacts on carbon sequestration
RISK_OF_REVERSAL = 0  # Risk of reversal buffer under ERF (reasonable values range from 0.05 [100 years] to 0.25 [25 years]) https://www.cleanenergyregulator.gov.au/ERF/Choosing-a-project-type/Opportunities-for-the-land-sector/Risk-of-reversal-buffer
'''
As of 20260318, RISK_OF_REVERSAL is set to 0 and just use the 5% ERF risk of reversal, as per Brett's comment in 
the LUF 2026 scenario runs document: "I suggested that we drop the fire risk and just use the 5% ERF risk of reversal. 
Doubling up is very conservative and leads to more area reqd to meet targets. You still want both on?"
'''


FIRE_RISK = 'med'   # Options are 'low', 'med', 'high'. Determines whether to take the 5th, 50th, or 95th percentile of modelled fire impacts.
""" 
Mean FIRE_RISK cell values (%)
- FD_RISK_PERC_5TH    80.3967
- FD_RISK_MEDIAN      89.2485
- FD_RISK_PERC_95TH   93.2735 

"""


# ---------------------------------------------------------------------------- #
# Economic parameters
# ---------------------------------------------------------------------------- #

# Amortise upfront (i.e., establishment and transitions) costs
AMORTISE_UPFRONT_COSTS = False


# θ — the EXACT ↔ CRISP dial of the transition flow model (fold-into-dominant, per cell).
# Essentially: COLLAPSE each cell's small dvar fractions (≤ θ) into its dominant fraction, so the
# solver sets up delta variables only for the collapsed base — every (source, cell) pair removed
# saves one full row of delta vars (~28 targets × 2 lms), while the collapsed land stays in the
# model (mobile, conserved) under the dominant's identity.
# Example, θ = 0.10, one dry cell:
#
#     true base:    Beef 0.55 | Winter cereals 0.35 | Hay 0.06 | Citrus 0.04
#     folded base:  Beef 0.65 | Winter cereals 0.35                            (Hay+Citrus -> Beef)
# θ→0: pure exact per-source model (at RESFACTOR=5 nothing folds below 0.04, the min block fraction).
# θ→1: one source per cell carrying the whole cell = the old crisp dominant-LU model.
# θ only applies to AG land-uses; non-ag sources are always exact (noise-floor cutoff, no folding).
EXACT_REACHABILITY_MIN_FRACTION = 0.01

# Discount rate for amortisation
DISCOUNT_RATE = 0.07     # 0.05 = 5% pa.

# Set amortisation period
AMORTISATION_PERIOD = 30 # years

# Scenario multiplier for land-use transition costs (establishment + water license + carbon penalty).
# 1 = baseline; <1 cheaper switching; >1 higher barrier. Applied on top of data.TRANS_COST_MULTS.
TRANSITION_COST_MULT = 1

# Scenario multiplier for technical-adoption ceilings (Asparagopsis, Precision Ag, AgTech EI, Biochar).
# 1 = baseline; <1 tighter ceilings; >1 relaxed ceilings (capped at 1.0 to stay a valid proportion).
TECH_ADOPT_MULT = 1

# Set whether to use demand elasticity when calculating commodity prices
DYNAMIC_PRICE = True



# ---------------------------------------------------------------------------- #
# Model parameters
# ---------------------------------------------------------------------------- #

# Optionally coarse-grain spatial domain (faster runs useful for testing). E.g. RESFACTOR 5 selects the middle cell in every 5 x 5 cell block
RESFACTOR = 5        # set to 1 to run at full spatial resolution, > 1 to run at reduced resolution.

# The step size for the temporal domain (years)
SIM_YEARS = list(range(2020, 2051, 5))

# Define the objective function
OBJECTIVE = 'maxprofit'   # maximise profit (revenue - costs)  **** Requires soft demand constraints otherwise agriculture over-produces
# OBJECTIVE = 'mincost'  # minimise cost (transitions costs + annual production costs)



DEMAND_CONSTRAINT_TYPE = 'hard'
'''
Options are 'soft', or 'hard'. This determines the type of demand constraint to apply in the model.
- 'soft': commodity can be produced under/over the target, but the under/over part will pay a penalty that
  equals the deviation amount multiplied by the corresponding prices. 
- 'hard': commodity must be produced at the target amount, with a relaxation factor (DEMAND_BOUNDS) 
  that allows for a certain percentage above the target to be produced (e.g., 1.05 allows for 5% overproduction).
'''                      

DEMAND_BOUNDS = {
    # Commodities need relaxation
    'sheep lexp':               [1.0, 1.0],     # Sheep live exports can be met exactly because its not co-produced with sheep (some sheep just not exported). 
    'sheep meat':               [1.0, 1.0],     # Meat and wool are co-produced in biologically fixed ratios, so either overproduce meat (~2.5 times), or
    'sheep wool':               [0.1, 1.5],     # underproduce wool (0.8 times).
    
    # Commodities with no relaxation (one-to-one land-use to commodity)
    'apples':                   [1.0, 1.0],
    'beef lexp':                [1.0, 1.0],
    'beef meat':                [1.0, 1.0],
    'citrus':                   [1.0, 1.0],
    'cotton':                   [1.0, 1.0],
    'dairy':                    [1.0, 1.0],
    'grapes':                   [1.0, 1.0],
    'hay':                      [1.0, 1.0],
    'nuts':                     [1.0, 1.0],
    'other non-cereal crops':   [1.0, 1.0],
    'pears':                    [1.0, 1.0],
    'plantation fruit':         [1.0, 1.0],
    'rice':                     [1.0, 1.0],
    'stone fruit':              [1.0, 1.0],
    'sugar':                    [1.0, 1.0],
    'summer cereals':           [1.0, 1.0],
    'summer legumes':           [1.0, 1.0],
    'summer oilseeds':          [1.0, 1.0],
    'tropical stone fruit':     [1.0, 1.0],
    'vegetables':               [1.0, 1.0],
    'winter cereals':           [1.0, 1.0],
    'winter legumes':           [1.0, 1.0],
    'winter oilseeds':          [1.0, 1.0],
}
'''
Dictionary of [lb, ub] bound multipliers for hard demand constraints. [lb, ub] = [1.0, 1.0] means the model must
hit the demand target exactly. Values > 1.0 allow overproduction; values < 1.0 allow underproduction.

Livestock co-production note: sheep (and beef) land-use cells simultaneously produce multiple commodities
(sheep: meat + wool + live exports) in biologically fixed ratios that differ from demand ratios.
The anchor commodity is wool (tight [1.0, 1.0]); meat must be given a wide UB because:
  - biological median meat/wool ratio = 1.856, but demand meat/wool = 1.60-1.75 (and declining)
  - soft-demand run shows sheep meat overshoots up to 2.23x by 2050 when wool demand is met
  - setting UB=2.34 (= 2.23 * 1.05 safety margin) avoids infeasibility across all years 2010-2050
Crops are one-to-one land-use to commodity — keep at [1.0, 1.0].
'''

RESCALE_FACTOR = 1e3
'''
All input data before feeding into the solver is rescaled in the range between 0 and this factor.
This is to avoid numerical issues with the solver when dealing with very small/large numbers. 
E.g., the water yield for some cells is 10t but the Biodiversity-score is 1e-7, making the 
the model sensitive to variations in input data. 
'''

REDUCE_FORCED_ZERO_ROWS = False
'''
Apply the forced-zero row reduction to the PRODUCTION model, not just the diagnosis probes.

An equality row `sum(a_i x_i) = 0` whose coefficients are all positive and whose variables all have
lb >= 0 has exactly one solution: every variable in it is zero. Deleting the row and fixing those
variables is therefore EXACT -- no feasible solution is removed, the optimum is unchanged.

LUTO emits an enormous number of them from the transition-flow node balances, one per (source, cell)
with no land to move. Measured on R2_SNES_T1525_cap15's 2045 model: 1,393,366 of 1,760,948 flow_in
rows (79 %), pinning 4,369,481 variables -- 30 % of ALL rows in the model, and the most degenerate
30 %. Gurobi's presolve would normally clear them, but the barrier runs with Presolve=0 by design
(presolve + homogeneous barrier once produced false infeasibility), so they go straight into the
factorisation of A.D.A'.

Default False pending full-trajectory validation: the mathematics is not in doubt, but a solve is
allowed to differ in which OPTIMAL vertex it reports, so the claim to check is that the objective
and the reported dvars are unchanged. The diagnosis probes apply the reduction unconditionally
(solvers/tools._feasibility_copy) -- there it is not an optimisation but a precondition, because
without it a probe including the flow system returns NUMERIC and the caller cannot tell that from
feasible.
'''

SOLVER_COEFF_MIN = 1e-4
'''
Minimum absolute coefficient threshold applied by ``_qsum()`` in solver.py before
adding a term to any Gurobi expression (constraints and objective alike).

After rescaling, cross-products of two rescaled values can still be tiny (e.g.
val_vector[r]=1e-3 × coeff[j]=1e-5 = 1e-8), stretching the matrix/objective
coefficient range far below Gurobi's recommended [1e-3, 1e6] band and causing
barrier divergence ("Numerical trouble encountered"). This threshold filters such
terms before they enter Gurobi.

Applied to ALL constraint / objective builders:
  Economy, Biodiversity-quality, GHG, Water, Renewable, GBF2/3/4/8,
  Demand/Quantity, and Regional Adoption limits.

1e-4 was chosen empirically: 1e-3 caused ~3% economic loss by filtering meaningful
small production coefficients; 1e-4 retains those while keeping the matrix range
ratio at 1e8 (well within Gurobi's safe zone).
'''





# ---------------------------------------------------------------------------- #
# Geographical raster writing parameters
# ---------------------------------------------------------------------------- #
WRITE_OUTPUTS = True                        # Whether to write outputs (e.g., GeoTIFFs) at the end of the run. Set to False to skip output writing (e.g. when doing a quick test run or debugging IIS infeasibility).

WRITE_REPORT_MAX_MEM_MB = 64 * 1024         # The maximum memory (in MB) to use for writing report layers.
                                            #   Estimated based on the 0.5 GB MEM usage when RESFACTOR = 13
                                            #   (for example, for RESFACTOR = 5, the MEM usage will be 0.5 * (13/5)^2 = 3.4 GB).

WRITE_CHUNK_SIZE = 4096                     # The processing size of each chunk during writeing process.
                                            #   E.g., layer of ~200 k cells (under chunk size of 1024) will create ~200 chunks.
                                            #   This makes memory usage to be ~1/200 of the original size.

                                            # ---- Biodiversity score/layer writing -------------------------------
                                            # 'off'      : skip entirely.
                                            # 'all'      : every species/community/vegetation group in the input CSV.
                                            # 'selected' : only the entries the model actually constrained, i.e. the
                                            #              same set the solver uses (region mode, selected regions,
                                            #              exclude lists and target dicts all applied). Independent of
                                            #              whether the matching GBF*_TARGET is on.
                                            # 'on' is accepted as a legacy alias for 'all'.
WRITE_GBF3_NVIS  = 'off'                    # 'off' | 'all' | 'selected'. Biodiversity GBF3 NVIS scores and map layers.
WRITE_GBF4_SNES  = 'off'                    # 'off' | 'all' | 'selected'. 'all' writes ~2000 species and takes ~5 hours.
WRITE_GBF4_ECNES = 'off'                    # 'off' | 'all' | 'selected'. Biodiversity GBF4 ECNES scores and map layers.


# ---------------------------------------------------------------------------- #
# Gurobi parameters
# ---------------------------------------------------------------------------- #

# Print detailed output to screen
VERBOSE = 1

# Number of threads to use in parallel algorithms (e.g., barrier). PBS_NCPUS is the requested CPUs on GADI hpc.
THREADS = 32

# Primal feasibility tolerance — defines the solver precision granule.
# ROUND_DECIMALS is derived from this: floor-truncation keeps digits down to
# this precision, so lb values are exact multiples of FEASIBILITY_TOLERANCE.
# Snap threshold for near-zero bounds and near-degenerate windows = FEASIBILITY_TOLERANCE * 10.
FEASIBILITY_TOLERANCE = 1e-6
''' Primal feasibility tolerance - Default: 1e-6, Min: 1e-9, Max: 1e-2'''

OPTIMALITY_TOLERANCE = 1e-2               
''' Dual feasility tolerance - Default: 1e-6, Min: 1e-9, Max: 1e-2'''

BARRIER_CONVERGENCE_TOLERANCE = 1e-5
'''
Barrier stops when the RELATIVE duality gap falls below this tolerance:
  |dual_obj - primal_obj| / (1 + |primal_obj|) < BarConvTol
 - Primal obj: objective at the current interior feasible point — a LOWER bound on the true
   maximum (the model has achieved at least this much; the true optimum could be higher).
 - Dual obj: derived from dual variables — an UPPER bound on the true maximum (the model
   cannot exceed this; ideally converges down toward the primal as the gap closes).
Range from 1e-2 (fast, loose) to 1e-8 (slow, tight; Gurobi default).
 - 20260606: relaxed from 1e-5 to 1e-3 — more constraint targets cause dual blow-up that
   prevents the barrier from closing the gap to 1e-5; crossover polishes the result.
'''

DROP_UNREACHABLE_CONSTRAINTS = []
'''
Which constraint groups may be SACRIFICED to keep a year solvable — ORDERED, least-valued first.

DEFAULT [] (since v2.4): the diagnose-and-drop machinery (solvers/tools.py) is OFF — nothing is
probed or removed, and an infeasible year fails as-is. To enable it, set an ordered list such as
['bio_snes', 'bio_ecnes', 'bio_nvis'] together with INFEASIBILITY_DIAGNOSIS_GROUPS below.

This list is the drop policy for BOTH halves of the infeasibility flow in simulation.py:

  1. Pre-solve, each group in INFEASIBILITY_DIAGNOSIS_GROUPS is feasibility-tested alone; a row
     that cannot hold even by itself (e.g. one ECNES community whose target exceeds anything its
     cells could reach) is dropped before the real solve.
  2. After a failed solve, the IIS names a set of rows that cannot all hold together; the earliest
     group in this list that appears in the IIS gives up one row, and the solve is retried.

Order matters: when a conflict could be resolved from several of these groups, the FIRST group
listed loses its row. Groups not listed are never dropped — the adoption caps, GHG, water and
demand are deliberately absent, so a conflict entirely among those ends the year and is REPORTED
instead of silently relaxing a scenario-defining constraint.

Group names come from CONSTRAINT_GROUPS in solvers/tools.py. Every drop is recorded in
out_<year>/dropped_constraints_<year>.csv, never silent.

Set to [] to drop nothing (e.g. when reproducing a historical run): the pre-solve test is then
skipped, and a failed year is still diagnosed — the IIS is printed — but nothing is removed, so
the year fails as-is.
'''

KNIFE_EDGE_DROP_BELOW = 1e-4
'''
Drop threshold for the pre-solve knife-edge census (see solvers/tools.knife_edge_rows).

The census names rows that are SATISFIABLE but only by a hair — tightening every diagnosis-group
RHS by 1%/1e-4/1e-6 of its magnitude and reading the IIS at each level. Rows whose relative
headroom is below THIS value, and whose group is in DROP_UNREACHABLE_CONSTRAINTS, are dropped
before the solve and recorded as action='DROPPED_KNIFE_EDGE' with their headroom tier
(`headroom_lt`) in out_<year>/dropped_constraints_<year>.csv. Thin rows above the threshold (and
ALL rows of non-droppable groups, e.g. the non-ag cap) are recorded as 'KNIFE_EDGE' but kept.

Why drop them at all: a row inside this margin is numerically indistinguishable from infeasible to
the production solve (FeasibilityTol 1e-6) — measured on R2_SNES_T1525_cap10, whose 2045 stalled
DETERMINISTICALLY (twice, to the digit) on rows every probe certified feasible. Dropping trades a
met-by-a-hair target for guaranteed termination, and the record says exactly how thin the margin
was. 1e-4 is the saturation threshold from the Phase-1 analysis: the GB cap sat at 5e-6 relative
slack when its runs returned status 4; healthy rows sit above 8.5e-2.

Set to 0 to disable knife-edge dropping (the census still records). Inert while
DROP_UNREACHABLE_CONSTRAINTS / INFEASIBILITY_DIAGNOSIS_GROUPS are empty — the census itself
only runs as part of the pre-solve spectrum.
'''

INFEASIBILITY_DIAGNOSIS_GROUPS = []
'''
Which constraint groups the infeasibility diagnosis works on.

DEFAULT [] (since v2.4): diagnosis is OFF — the pre-solve feasibility spectrum, knife-edge census
and post-failure restricted IIS are all skipped (simulation.py bypasses them when this is empty).
The recommended set when diagnosing infeasible runs is
['bio_nvis', 'bio_snes', 'bio_ecnes', 'nonag_cap', 'flow_in', 'flow_out']. Two roles in simulation.py:

  * pre-solve: each of these groups is feasibility-tested ALONE (plus the structural rows);
  * post-failure: the diagnosis probe keeps ONLY these groups (plus structural), and the IIS is
    computed on that restricted copy.

Restriction is what makes the IIS affordable, not just faster: one measured feasibility solve went
from 417 s (full model) to 13 s (restricted), and the full model's computeIIS did not finish in
2.7 h. It is not free — discarding constraints is a relaxation, so:

    restricted model INFEASIBLE  ->  the full model is infeasible too, and the conflict lies
                                     entirely within the groups kept. Sound.
    restricted model FEASIBLE    ->  says NOTHING about the full model. A conflict involving a
                                     discarded group is invisible, and the year will still fail.

So only exclude a group when you are confident it does not bind. The recommended set keeps the
biodiversity families, the non-ag adoption cap AND the transition-flow system (`flow_in` /
`flow_out`), and discards demand, GHG, water and renewables. (Water was tested: a restriction that
still contained it returned the identical IIS, so it does not participate.)

The flow system is in the recommended set because excluding it manufactures false FEASIBLE verdicts
(2026-08-10): a species target can be satisfiable in an unconstrained-land sense yet unreachable
given how land is permitted to MOVE — T_MAT reachability and source availability live in the flow
rows. Five capped runs stalled for days on exactly this: their flow-blind probes reported feasible,
the ladder ran into `Numerical trouble`, and Gurobi's internal simplex fallback never returned.
With flow in scope the same probe answers INFEASIBLE in ~75 s and the IIS names the rows
(Macquaria australasica, E. alligatrix, ...); dropping them let every stalled year solve by
ordinary barrier in ~8-14 min. Diagnosis cost: ~25-75 s per IIS round.

`bio_gbf2` is deliberately NOT in the default: when GBF2 is on ('high', hard), its severely-scaled
national row (dual ~500x the largest SNES dual) makes every deletion-filter LP inside computeIIS
glacial — 111 min for 5 rounds, measured on R4_GBF2_T3050_cap25. Excluding it only blinds the
probe, never the solve (GBF2 stays a hard constraint in the real model); the accepted risk is that
a genuinely GBF2-involving conflict surfaces as a real-solve failure instead of a pre-solve drop.

Group names come from CONSTRAINT_GROUPS in solvers/tools.py. `cell_usage` and `ag_mgt_link` are
always kept regardless: without them land is not scarce and almost anything looks feasible.

Set to [] (or None) to turn the diagnosis machinery OFF entirely — both the pre-solve test and the
post-failure IIS are skipped, and a failed year just fails. NOTE there is currently no value that
diagnoses against the FULL model; the closest is listing every group from CONSTRAINT_GROUPS.
'''

SCALE_FLAG = 0
''' 
Scales the rows and columns of the model to improve the numerical properties of 
the constraint matrix. -1: Auto, 0: No scaling, 1: equilibrium scaling (First scale each 
row to make its largest nonzero entry to be magnitude one, then scale each column to 
max-norm 1), 2: geometric scaling, 3: multi-pass equilibrium scaling. Testing revealed 
that 1 tripled solve time, 3 led to numerical problems.
'''

RETRY_PARAMS = [
    (0, 2, -1, -1, -1),   # NF, Method, Crossover, Presolve, BarHomogeneous
    (0, 1,  0, -1, 0 ),
]
'''
List of solve attempts to try in order, per year. Each entry MUST be a
(NumericFocus, Method, Crossover, Presolve, BarHomogeneous) tuple.

NumericFocus:
    0 = automatic (slight preference for speed); 1-3 = increasingly careful.
    NF=0 is safe for all attempts since geometry mean rescaling (2026-05)
    keeps LHS/RHS coefficients well within Gurobi's recommended range.

Method:
    -1 = automatic, 0 = primal simplex, 1 = dual simplex, 2 = barrier,
     3 = concurrent, 4 = deterministic concurrent, 5 = det. concurrent simplex.

Crossover:
    -1 = automatic, 0 = off, 1/2/3 = forced variants.
    Converts a barrier interior-point solution to a vertex; can be slow on
    large models. Use -1 (auto) as a fallback if barrier stagnates.

Presolve:
    -1 = automatic, 0 = off, 1 = conservative, 2 = aggressive.
    Keep OFF (0) for barrier (Method=2) — observed to introduce numerical
    errors that cause the homogeneous barrier to declare false infeasibility.
    Safe to enable for simplex (Method=0 or 1).

BarHomogeneous:
    -1 = automatic, 0 = off, 1 = on.
    Keep OFF (0): the homogeneous algorithm's tau parameter drifts toward zero
    in highly degenerate problems, triggering false INFEASIBLE (status 3) even
    with NumericFocus=3. With 0, the barrier reports NUMERIC (12) or SUBOPTIMAL
    (13) instead — both handled by the retry loop. Set to 1 only when debugging
    to avoid ambiguous INF_OR_UNBD status.

Default sequence:
  (0, 2, -1, -1, -1) barrier, auto crossover, presolve off, homogeneous off  — fast first pass
  (0, 1,  0, -1, 0)  dual simplex, presolve auto, homogeneous off            — fallback; simplex
                     walks the boundary so it cannot misdiagnose feasibility
                     from an interior-point argument; presolve safe with simplex
'''





# ---------------------------------------------------------------------------- #
# No-Go areas; Regional adoption constraints
# ---------------------------------------------------------------------------- #

EXCLUDE_NO_GO_LU = False        # True or False
'''
The exclude no-go land-uses option.
- True: exclude land-uses from no-go areas. User must provide the `NO_GO_VECTORS` dictionary, with land-use names as keys and shapefile paths as values.
- False: do not exclude land-uses from no-go areas.
'''
NO_GO_VECTORS = {
    'Winter cereals':           'no_go_areas/no_go_Winter_cereals.shp',
    'Environmental Plantings':  'no_go_areas/no_go_Enviornmental_Plantings.shp'
}
'''
Land-use and vector file pairs to exclude land-use from being utilised in that area. 
- The key is the land-use name. 
- The value is the path to the ESRI shapefile.
'''

REGIONAL_ADOPTION_CONSTRAINTS = 'off'            
'''
Adoption mode for non-ag land uses.
- 'off': no regional adoption constraints
- 'on', user needs to set the percentage targets in 'input/regional_adoption_zones.xlsx'
- 'NON_AG_CAP', the SUM of all non-ag land uses can not exceed a certain percentage (REGIONAL_ADOPTION_NON_AG_CAP) in every region
'''

REGIONAL_ADOPTION_ZONE = 'NRM_CODE'              # 'ABARES_AAGIS', 'LGA_CODE', 'NRM_CODE', 'IBRA_ID', 'SLA_5DIGIT'
'''
The regional adoption zone is the spatial unit used to enforce regional adoption constraints.
The options are:
  - 'ABARES_AAGIS': Australian Bureau of Agricultural and Resource Economics and Sciences (ABARES) Agricultural and Agribusiness Geographic Information System (AAGIS) regions.
  - 'LGA_CODE': Local Government Area code.
  - 'NRM_CODE': Natural Resource Management code.
  - 'IBRA_ID': Interim Biogeographic Regionalisation of Australia (IBRA) region code.
  - 'SLA_5DIGIT': Statistical Local Area (SLA) 5-digit code.
'''


REGIONAL_ADOPTION_NON_AG_REGION = 'NRM'
'''
The regional adoption zone for non-agricultural land uses when using the 'NON_AG_CAP' mode.
The options are the same as REGIONAL_ADOPTION_ZONE:
- 'NRM': Natural Resource Management code.
- 'State': Australian state regions.
'''


REGIONAL_ADOPTION_NON_AG_CAP = 15
'''
None or numbers between 0-100 (both inclusive);
 Only work under 'REGIONAL_ADOPTION_CONSTRAINTS = NON_AG_CAP'.
 E.g., 15 means the combined area of all non-ag land uses can not exceed 15% of each region's area.
'''

REGIONAL_ADOPTION_NON_AG_CAP_REGIONS = []
'''
Scope of the NON_AG_CAP: which regions (names from REGIONAL_ADOPTION_NON_AG_REGION, e.g. NRM names like
'North East', 'Goulburn Broken') the SUM-of-non-ag cap is applied to.
- []  (empty)  -> cap ALL regions (default / original behaviour).
- ['North East', 'Goulburn Broken'] -> cap ONLY these regions; everywhere else is uncapped.
Only used under 'REGIONAL_ADOPTION_CONSTRAINTS = NON_AG_CAP'.
'''

REGIONAL_ADOPTION_NON_AG_CAP_OVERRIDE = {}
'''
Per-region cap percentages that OVERRIDE the uniform REGIONAL_ADOPTION_NON_AG_CAP for named regions.
Maps region name -> percentage (0-100), e.g. {'Goulburn Broken': 10, 'North East': 20}. A region not in
this dict uses REGIONAL_ADOPTION_NON_AG_CAP. Applied within the REGIONAL_ADOPTION_NON_AG_CAP_REGIONS scope
(or all regions if that list is empty). Only used under 'REGIONAL_ADOPTION_CONSTRAINTS = NON_AG_CAP'.
'''
                                        



# ---------------------------------------------------------------------------- #
# Non-agricultural land usage parameters
# ---------------------------------------------------------------------------- #

NON_AG_LAND_USES = {
    'Environmental Plantings': True,
    'Riparian Plantings': True,
    'Sheep Agroforestry': True,    
    'Beef Agroforestry': True,
    'Carbon Plantings (Block)': True,
    'Sheep Carbon Plantings (Belt)': True,
    'Beef Carbon Plantings (Belt)': True,
    'BECCS': False,
    'Destocked - natural land': True,
}
"""
The dictionary here is the master list of all of the non agricultural land uses
and whether they are currently enabled in the solver (True/False).

To disable a non-agricultural land use, change the correpsonding value of the
NON_AG_LAND_USES dictionary to false.
"""


NON_AG_LAND_USES_REVERSIBLE = {
    'Environmental Plantings': False,
    'Riparian Plantings': False,
    'Sheep Agroforestry': False,
    'Beef Agroforestry': False,
    'Carbon Plantings (Block)': False,
    'Sheep Carbon Plantings (Belt)': False,
    'Beef Carbon Plantings (Belt)': False,
    'BECCS': False,
    'Destocked - natural land': True,
}
"""
The values of the below dictionary determine whether the model is allowed to abandon non-agr.
land uses on cells in the years after it chooses to utilise them. For example, if a cell has is using 'Environmental Plantings'
and the corresponding value in this dictionary is False, all cells using EP must also utilise this land use in all subsequent
years.

CAUTION: Setting reversibility == True can cause infeasibility issues in timeseries runs due to not being able to meet the water constraints.
With the net water yield limit set to say 80%, some catchments could be close to that yield then they experience some land use change to meet
GHG and biodiversity targets. This pushes the catchment close to the net yield constraint. Over time, climate change may reduce the amount of
water yield and if non-ag land uses are not reversible then a catchment may not be able to meet the net yield constraint.
This is expected behaviour and the user must choose how to deal with it.
"""

# Cost of fencing per linear metre
FENCING_COST_PER_M = 2

# Environmental Plantings Parameters
EP_ANNUAL_MAINTENANCE_COST_PER_HA_PER_YEAR = 100
EP_ANNUAL_ECOSYSTEM_SERVICES_BENEFIT_PER_HA_PER_YEAR = 0

# Carbon Plantings Block Parameters
CP_BLOCK_ANNUAL_MAINTENANCE_COST_PER_HA_PER_YEAR = 100
CP_BLOCK_ANNUAL_ECOSYSTEM_SERVICES_BENEFIT_PER_HA_PER_YEAR = 0

# Carbon Plantings Belt Parameters
CP_BELT_ANNUAL_MAINTENANCE_COST_PER_HA_PER_YEAR = 100
CP_BELT_ANNUAL_ECOSYSTEM_SERVICES_BENEFIT_PER_HA_PER_YEAR = 0

CP_BELT_ROW_WIDTH = 20
CP_BELT_ROW_SPACING = 40
CP_BELT_PROPORTION = CP_BELT_ROW_WIDTH / (CP_BELT_ROW_WIDTH + CP_BELT_ROW_SPACING)
cp_no_alleys_per_ha = 100 / (CP_BELT_ROW_WIDTH + CP_BELT_ROW_SPACING)
CP_BELT_FENCING_LENGTH = 100 * cp_no_alleys_per_ha * 2     # Length (average) of fencing required per ha in metres

# Riparian Planting Parameters
RP_ANNUAL_MAINTENANCE_COST_PER_HA_PER_YEAR = 100
RP_ANNUAL_ECOSYSTEM_SERVICES_BENEFIT_PER_HA_PER_YEAR = 0

RIPARIAN_PLANTING_BUFFER_WIDTH = 30
RIPARIAN_PLANTING_TORTUOSITY_FACTOR = 0.5

# Agroforestry Parameters
AF_ANNUAL_MAINTENANCE_COST_PER_HA_PER_YEAR = 100
AF_ANNUAL_ECOSYSTEM_SERVICES_BENEFIT_PER_HA_PER_YEAR = 0

AGROFORESTRY_ROW_WIDTH = 20
AGROFORESTRY_ROW_SPACING = 40
AF_PROPORTION = AGROFORESTRY_ROW_WIDTH / (AGROFORESTRY_ROW_WIDTH + AGROFORESTRY_ROW_SPACING)
no_belts_per_ha = 100 / (AGROFORESTRY_ROW_WIDTH + AGROFORESTRY_ROW_SPACING)
AF_FENCING_LENGTH_HA = 100 * no_belts_per_ha * 2 # Length of fencing required per ha in metres


# ---------------------------------------------------------------------------- #
# Renewable energy parameters
# ---------------------------------------------------------------------------- #
RENEWABLES_OPTIONS = {
    'Utility Solar PV': False,
    'Onshore Wind': False,
}


EXCLUDE_RENEWABLES_IN_GBF2_MASKED_CELLS = True
'''
Whether to exclude renewable energy installation on cells inside the GBF2 masked layer (i.e., cells with very high biodiversity value).
 - True: The model cannot install renewable energy on GBF2-masked cells.
 - False: The model can install renewable energy on GBF2-masked cells.
'''

RENEWABLE_GBF2_CUT_WIND = 20
RENEWABLE_GBF2_CUT_SOLAR = 20
'''
Independent biodiversity area coverage percentage thresholds (same scale as GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT)
for determining which cells to exclude from renewable energy installation.
Cells with biodiversity quality >= the conservation performance curve value at this cut are excluded.
Lower values = fewer cells excluded, higher values = more cells excluded.
'''

EXCLUDE_RENEWABLES_IN_EPBC_MNES_MASK = True
'''
Whether to exclude renewable energy installation on cells inside the EPBC MNES prioritization layer
(i.e., cells with high MNES priority rank).
 - True: The model cannot install renewable energy on MNES high-priority cells.
 - False: The model can install renewable energy on MNES high-priority cells.
'''

RENEWABLE_EPBC_MNES_CUT_SOLAR = 10
RENEWABLE_EPBC_MNES_CUT_WIND = 10
'''
Independent MNES area coverage percentage thresholds for determining which cells to exclude from
renewable energy installation. Cells with MNES priority rank >= the performance curve value at this
cut are excluded.
Lower values = fewer cells excluded, higher values = more cells excluded.
'''


RENEWABLE_TARGET_SCENARIO_TARGETS = 'Gladstone - Core'
'''
The renewable energy target scenario to use when `RENEWABLES_OPTIONS` is set to True. One of
 - 'AEMO 2026 ISP - Accelerated Transition'
 - 'AEMO 2026 ISP - Slower Growth'
 - 'AEMO 2026 ISP - Step Change'
 - 'Gladstone - BESS Sensitivity'
 - 'Gladstone - Core'
'''


RENEWABLE_TARGET_SCENARIO_INPUT_LAYERS = 'step_change'
'''
The renewable energy target scenario for input spatial layersto use when `RENEWABLES_OPTIONS`
is set to True. One of
 - 'step_change',
 - 'accelerated_transition',
 - 'ANU_transmission_T3',
 - 'ANU_transmission_T5',
 - 'ANU_transmission_T10'.
'''

RE_TARGET_LEVEL = "STATE"  # options: "STATE", "NRM"; TODO: currently (20260205) only support STATE, will add NRM in the future.
'''
The spatial level at which to apply the renewable energy targets when `RENEWABLES_OPTIONS` is set to True.
Options include "STATE" or "NRM". Currently (20260205) only support STATE.
'''

INSTALL_CAPACITY_MW_HA = {
    "Utility Solar PV": 0.45,
    "Onshore Wind": 0.04,
}
'''
The per/ha capacity (Mw/ha) for each renewable energy management type.
'''


RENEWABLES_ADOPTION_LIMITS = {
    'Utility Solar PV': 1.0,        # Maximum proportion of land that can be used for Utility Solar PV
    'Onshore Wind': 1.0,            # Maximum proportion of land that can be used for Onshore Wind
}
'''
The maximum proportion of land that can be used for each renewable energy management type.
For example, if RENEWABLES_ADOPTION_LIMITS['Utility Solar PV'] = 0.5, then at most 50% of
the land can be used for Utility Solar PV.
'''



# ---------------------------------------------------------------------------- #
# Land use type groupings (used by AgTech bundles)
# ---------------------------------------------------------------------------- #

LUS_CROPPING     = ['Hay', 'Summer cereals', 'Summer legumes', 'Summer oilseeds', 'Winter cereals', 'Winter legumes', 'Winter oilseeds']
LUS_INT_CROPPING = ['Cotton', 'Other non-cereal crops', 'Rice', 'Sugar', 'Vegetables']
LUS_HORTICULTURE = ['Apples', 'Citrus', 'Grapes', 'Nuts', 'Pears', 'Plantation fruit', 'Stone fruit', 'Tropical stone fruit']

LU2TYPE = (
    {lu: "cropping"     for lu in LUS_CROPPING}
  | {lu: "int_cropping" for lu in LUS_INT_CROPPING}
  | {lu: "horticulture" for lu in LUS_HORTICULTURE}
)


# ---------------------------------------------------------------------------- #
# Agricultural Management parameters
# ---------------------------------------------------------------------------- #

AG_MANAGEMENTS_TO_LAND_USES = {
    'Asparagopsis taxiformis':  ['Beef - modified land', 'Sheep - modified land', 'Dairy - natural land', 'Dairy - modified land'],
    'Precision Agriculture':    LUS_CROPPING + LUS_INT_CROPPING + LUS_HORTICULTURE,
    'Ecological Grazing':       ['Beef - modified land', 'Sheep - modified land', 'Dairy - modified land'],
    'Savanna Burning':          ['Beef - natural land', 'Dairy - natural land', 'Sheep - natural land', 'Unallocated - natural land'],
    'AgTech EI':                LUS_CROPPING + LUS_INT_CROPPING + LUS_HORTICULTURE,
    'Biochar':                  LUS_CROPPING + LUS_HORTICULTURE,
    'HIR - Beef':               ['Beef - natural land'],
    'HIR - Sheep':              ['Sheep - natural land'],
    'Utility Solar PV':         ['Unallocated - modified land',
                                 'Beef - modified land', 'Sheep - modified land', 'Dairy - modified land',
                                 *[lu for lu in LUS_CROPPING if lu != 'Hay']],  # 'Hay' is missing in the PV raw bundle data.
    'Onshore Wind':             ['Unallocated - modified land',
                                 'Beef - modified land', 'Sheep - modified land', 'Dairy - modified land',
                                 *LUS_CROPPING,
                                 *LUS_INT_CROPPING]
}                                


AG_MANAGEMENTS = {
    'Asparagopsis taxiformis': True,
    'Precision Agriculture': True,
    'Ecological Grazing': False,
    'Savanna Burning': True,
    'AgTech EI': True,
    'Biochar': True,
    'HIR - Beef': True,
    'HIR - Sheep': True,
    'Utility Solar PV': RENEWABLES_OPTIONS['Utility Solar PV'],
    'Onshore Wind': RENEWABLES_OPTIONS['Onshore Wind'],
}
"""
The dictionary below contains a master list of all agricultural management options and
which land uses they correspond to.

To disable an ag-mangement option, change the corresponding value in the AG_MANAGEMENTS dictionary to False.
"""

AG_MANAGEMENTS_REVERSIBLE = {
    'Asparagopsis taxiformis': True,
    'Precision Agriculture': True,
    'Ecological Grazing': True,
    'Savanna Burning': True,
    'AgTech EI': True,
    'Biochar': True,
    'HIR - Beef': False,        # Can not abandon HIR - Beef once adopted (irreversible)
    'HIR - Sheep': False,       # Can not abandon HIR - Sheep once adopted (irreversible)
    'Utility Solar PV': False,  # Can not abandon Utility Solar PV once adopted due to the long lifespan and high transition costs
    'Onshore Wind': False,      # Can not abandon Onshore Wind once adopted due to the long lifespan and high transition costs
}
"""
The values of the below dictionary determine whether the model is allowed to abandon agricultural
management options on cells in the years after it chooses to utilise them. For example, if a cell has is using 'Asparagopsis taxiformis',
and the corresponding value in this dictionary is False, all cells using Asparagopsis taxiformis must also utilise this land use
and agricultural management combination in all subsequent years.

WARNING: changing to False will result in 'locking in' land uses on cells that utilise the agricultural management option for
the rest of the simulation. This may be an unintended side effect.
"""


# The cost for removing and establishing irrigation infrastructure ($ per hectare)
REMOVE_IRRIG_COST = 5000
NEW_IRRIG_COST = 10000

# Savanna burning cost per hectare per year ($/ha/yr)
SAVBURN_COST_HA_YR = 10

# The minimum value an agricultural management variable must take for the write_output function to consider it being used on a cell
AGRICULTURAL_MANAGEMENT_USE_THRESHOLD = 0.1

# Productivity contribution of HIR compared to not implementing HIR
HIR_PRODUCTIVITY_CONTRIBUTION = 0.5

# HIR celling factor, assuming HIR achienves x% of bio/GHG benefits of the Destocked - natural land land use
HIR_CEILING_PERCENTAGE = 0.8

# Maintainace cost for HIR
BEEF_HIR_MAINTENANCE_COST_PER_HA_PER_YEAR = 100
SHEEP_HIR_MAINTENANCE_COST_PER_HA_PER_YEAR = 100







# ---------------------------------------------------------------------------- #
# Off-land commodity parameters
# ---------------------------------------------------------------------------- #

OFF_LAND_COMMODITIES = ['pork', 'chicken', 'eggs', 'aquaculture']
EGGS_AVG_WEIGHT = 60  # Average weight of an egg in grams


# ---------------------------------------------------------------------------- #
# Environmental parameters
# ---------------------------------------------------------------------------- #

# Take data from 'GHG_targets.xlsx', 
GHG_TARGETS_DICT = {
    'off':     None,
    'low':    '1.8C 67%',
    'high':   '1.5C 50%',
}

# Greenhouse gas emissions limits and parameters *******************************
GHG_EMISSIONS_LIMITS = 'low'         # 'off', 'low', 'medium', or 'high'
'''
`GHG_EMISSIONS_LIMITS` options include: 
- (deprecated) Assuming agriculture is responsible to sequester 100% of the carbon emissions
    - '1.5C (67%)', '1.5C (50%)', or '1.8C (67%)' 
- (deprecated) Assuming agriculture is responsible to sequester carbon emissions not including electricity emissions and  off-land emissions 
    - '1.5C (67%) excl. avoided emis', '1.5C (50%) excl. avoided emis', or '1.8C (67%) excl. avoided emis'
- (deprecated) Assuming agriculture is responsible to sequester carbon emissions only in the scope 1 emissions (i.e., direct emissions From-land-use and livestock types)
    - '1.5C (67%) excl. avoided emis SCOPE1', '1.5C (50%) excl. avoided emis SCOPE1', or '1.8C (67%) excl. avoided emis SCOPE1'
- Assuming agriculture is responsible to sequester carbon emissions only in the scope 1 emissions (i.e., direct emissions From-land-use and livestock types):
    - '1.5C 50%', '1.8C 67%'
'''
  	  	  

# Carbon price scenario: either 'AS_GHG', 'Default', '100', or 'CONSTANT', or NONE.
# Setting to None falls back to the 'Default' scenario.
CARBON_PRICES_FIELD = 'CONSTANT'

# Automatically update the carbon price field if it is set to 'AS_GHG'
if CARBON_PRICES_FIELD == 'AS_GHG':
    if GHG_TARGETS_DICT[GHG_EMISSIONS_LIMITS] is None:
        raise ValueError(
            "CARBON_PRICES_FIELD='AS_GHG' requires GHG_EMISSIONS_LIMITS to name a GHG target "
            f"(got '{GHG_EMISSIONS_LIMITS}'); set an explicit carbon price scenario instead."
        )
    CARBON_PRICES_FIELD = GHG_TARGETS_DICT[GHG_EMISSIONS_LIMITS][:9].replace('(','')  # '1.5C (67%) excl. avoided emis' -> '1.5C 67%'

if CARBON_PRICES_FIELD == 'CONSTANT':
    CARBON_PRICE_COSTANT = 0.0  # The constant value to add to the carbon price (e.g., $10/tonne CO2e).
'''
Only works when CARBON_PRICES_FIELD is set to 'CONSTANT'.
'''


USE_GHG_SCOPE_1 = True
'''
Controls whether the solver uses only scope 1 (direct, on-farm) GHG emissions — as defined
by the Australian NGGI (National Greenhouse Gas Inventory) — or the full profile that also
includes scope 2 electricity and scope 3 indirect emissions (fertiliser/pesticide production).

Rationale (Sanson, van Schoten et al., 2025):
  AusTIMES agriculture sector scope 1 baseline (2022, excl. off-land): ~81.3 MtCO2e
  LUTO2 on-land ag baseline (2022, full profile):                       ~95.1 MtCO2e
  LUTO2 on-land ag baseline (2022, after all scope 1 exclusions):       ~70-73 MtCO2e

  Because AusTIMES already models energy-system decarbonisation externally, including
  scope 2/3 emissions in LUTO2's solve causes double-counting of those reductions.
  The ~70-73 Mt revised LUTO2 baseline aligns with the NGGI scope 1 column used in
  the AusTIMES pathway targets (1.5C 50% excl. avoided emissions).

When True, the solver excludes:
  Crops     — energy-related CO2 from: chemical application (non-liming), crop management,
              cultivation, fodder production, harvesting, irrigation, pasture seed production,
              and sowing; plus fertiliser production and pesticide production.
              Only CROP_GHG_SCOPE_1 sources (soil N2O) enter the constraint.
  Livestock — electricity use (CO2E_KG_HEAD_ELEC), fuel use (CO2E_KG_HEAD_FUEL),
              fodder production (CO2E_KG_HEAD_FODDER), and seed (CO2E_KG_HEAD_SEED).
              Only LVSTK_GHG_SCOPE_1 sources (enteric, dung/urine, manure, leach/runoff)
              enter the constraint.

Note: BECCS should also be disabled when using AusTIMES pathway targets to maintain
inter-model consistency (energy-related CO2 displacement is handled by AusTIMES).
'''

CROP_GHG_SCOPE_1 = ['CO2E_KG_HA_SOIL']
LVSTK_GHG_SCOPE_1 = ['CO2E_KG_HEAD_DUNG_URINE', 'CO2E_KG_HEAD_ENTERIC', 'CO2E_KG_HEAD_IND_LEACH_RUNOFF', 'CO2E_KG_HEAD_MANURE_MGT']


GHG_CONSTRAINT_TYPE = 'hard'  # Adds GHG limits as a constraint in the solver (linear programming approach)
# GHG_CONSTRAINT_TYPE = 'soft'  # Adds GHG usage as a type of slack variable in the solver (goal programming approach)
# NOTE: 'soft' mode is planned to be decommissioned so the objective only considers economy.

SOLVE_WEIGHT_BETA = 0.5
'''
The weight of the deviations from target in the objective function.
 - if approaching 0, the model will ignore the deviations from target.
 - if approaching 1, the model will try harder to meet the target.
'''


# Water use yield and parameters *******************************
WATER_LIMITS = 'on'                     # 'on' or 'off'. 'off' will turn off water net yield limit constraints in the solver.
WATER_CLIMATE_CHANGE_IMPACT = 'on'      # 'on' or 'off'. 'off' will turn off climate change impact on water yields.
'''
    When 'on', model will consider water yield change driven by climate change.

    Note the cxtreme CCI target relaxation (see water.py:get_water_target_inside_LUTO_by_CCI):
    1. Compute extreme CCI delta per region: the minimum water yield change across all SIM_YEARS
       (via get_water_delta_by_extreme_CCI_for_whole_region), assuming land use stays constant,
       combining inside-LUTO Ag-land and outside-LUTO CCI deltas.
    2. Compute extreme scenario water availability per region:
           wny_extreme_CCI = wny_inside_LUTO + wny_outside_LUTO - wreq_domestic + CCI_extreme_stress
       where wreq_domestic is domestic/industrial water demand (positive), and CCI_extreme_stress
       is the worst-case yield reduction from step 1 (negative).
    3. Compare against historical target: wny_hist_target = hist_level * WATER_STRESS
    4. If wny_extreme_CCI < wny_hist_target, the target is relaxed to wny_extreme_CCI to avoid
       solver infeasibility. The inside-LUTO target = relaxed_target - wny_outside_LUTO.
       Otherwise, the standard target applies: wny_hist_target - wny_outside_LUTO.
'''

WATER_CONSTRAINT_TYPE = 'hard'  # Adds water limits as a constraint in the solver (linear programming approach)
# WATER_CONSTRAINT_TYPE = 'soft'  # Adds water usage as a type of slack variable in the solver (goal programming approach)
# NOTE: 'soft' mode is planned to be decommissioned so the objective only considers economy.


# Regionalisation to enforce water use limits by
WATER_REGION_DEF = 'Drainage Division'         # 'River Region' or 'Drainage Division' Bureau of Meteorology GeoFabric definition
"""
    Water net yield targets: the value represents the proportion of the historical water yields
    that the net yield must exceed in a given year. Base year (2010) uses base year net yields as targets.
    Everything past the latest year specified uses the target figure for the latest year.
    
    Safe and just Earth system boundaries suggests a water stress of 0.2 (yield of 0.8). This is inclusive of
    domestic/industrial: https://www.nature.com/articles/s41586-023-06083-8, Approximately 70% of the total water use
    is used for agricultural purposes. This includes water used for irrigation, livestock, and domestic purposes on farms,
    with the rest used for domestic/industrial  https://soe.dcceew.gov.au/inland-water/pressures/population
    Hence, assuming that this proportion is uniform over all catchments and remains constant over time then if water
    stress is 0.2 then agriculture can use up 70% of this, leaving 30% for domestic/industrial. The water yield target for ag
    should then be historical net yield * (1 - water stress * agricultural share)
    
    Aqueduct water stress levels:
    Low stress < 10% of the water available is withdrawn annually
    Low to medium stress 10-20% of the water available is withdrawn annually
    Medium to high stress 20-40% 10% of the water available is withdrawn annually
    High stress 40-80% of the water available is withdrawn annually
    Extremely high stress > 80% of the water available is withdrawn annually
    
    https://chinawaterrisk.org/resources/analysis-reviews/aqueduct-global-water-stress-rankings/ 
"""

WATER_STRESS = 0.6                                      
'''
    Aqueduct limit catchments, 0.6 means the water yield in a region must be >= 60% of the historical water yield.
    The safe and just Earth system boundaries suggests a water stress of. We tried but it would lead to infeasibility
    issues in the model.

    There are two notes for calculating the water yield targets at watershed regions level:
     - The domestic/industrial water use is subtracted from total available water when computing
       the extreme climate scenario yield: wny_extreme_CCI = inside + outside - domestic + CCI_extreme_stress
     - If the extreme climate change scenario makes a region's water yield fall below the historical
       target (hist_level * WATER_STRESS), the target is relaxed to the extreme scenario level to
       avoid infeasibility. The inside-LUTO target is then derived by subtracting the outside-LUTO
       contribution: wny_inside_LUTO_target = raw_target - wny_outside_LUTO
'''

# Consider livestock drinking water (0 [off] or 1 [on]) ***** Livestock drinking water can cause infeasibility issues with water constraint in Pilbara
LIVESTOCK_DRINKING_WATER = 1

# Consider water license costs (0 [off] or 1 [on]) of land-use transition ***** If on then there is a noticeable water sell-off by irrigators in the MDB when maximising profit
INCLUDE_WATER_LICENSE_COSTS = 1



# Biodiversity limits and parameters *******************************


# ------------------- Agricultural biodiversity parameters -------------------

# Global Biodiversity Framework Target 2: Restore 30% of all Degraded Ecosystems
GBF2_TARGET = 'high'              # 'off', 'low', 'medium', or 'high'
'''
Kunming-Montreal Global Biodiversity Framework Target 2: Restore 30% of all Degraded Ecosystems
Ensure that by 2030 at least 30 per cent of areas of degraded terrestrial, inland water, and coastal and marine ecosystems are under effective restoration,
in order to enhance biodiversity and ecosystem functions and services, ecological integrity and connectivity.
 - 'off' will turn off the GBF-3 target. 
 - 'low' is the low level of biodiversity target (i.e., restore 0% of degreaded biodiversity socore in the 'priority degraded land').
 - 'medium' is the medium level of biodiversity target (i.e., restore 15% of degreaded biodiversity socore in the 'priority degraded land').
 - 'high' is the high level of biodiversity target (i.e., restore 25% of degreaded biodiversity socore in the 'priority degraded land').
'''


# Set biodiversity target (0 - 1 e.g., 0.3 = 30% of total achievable Zonation biodiversity benefit)
GBF2_TARGETS_DICT = {
    'off':     None,
    'low':    {2030: 0,    2050: 0,    2100: 0},
    'medium': {2030: 0.30, 2050: 0.30, 2100: 0.30},
    'high':   {2030: 0.30, 2050: 0.50, 2100: 0.50},
}


GBF2_CONSTRAINT_TYPE = 'hard' # Adds biodiversity limits as a constraint in the solver (linear programming approach)
# GBF2_CONSTRAINT_TYPE = 'soft'  # Adds biodiversity usage as a type of slack variable in the solver (goal programming approach)
'''
The constraint type for the biodiversity target.
- 'hard' adds biodiversity limits as a constraint in the solver (linear programming approach)
- 'soft' adds biodiversity usage as a type of slack variable in the solver (goal programming approach)
'''


GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT = 15
'''
Based on Zonation alogrithm, the biodiversity feature coverage (an indicator of overall biodiversity benifits) is 
more attached to high rank cells (rank is an indicator of importance/priority in biodiversity conservation). 
For example, cells with rank between 0.9-1.0 only cover 20% of the areas but contribute to 40% of the biodiversity benefits.

By sorting the rank values from high to low and plot the cumulative area and cumulative biodiversity benefits,
we can get the a curve that shows the relationship between the area and the biodiversity benefits. In LUTO, we normalise
the area and biodiversity benefits between 0-100, and use the `GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT` as the threshold
to identify the priority degraded areas that should be conserved to achieve the biodiversity target.

If set to 0, no cells will be considered as priority degraded areas, equal to not setting any GBF2 target.
If set to 100, all cells will be considered as priority degraded areas, equal to setting GBF2 target covering the whole LUTO study area.
'''


# Biodiversity quality options
BIO_QUALITY_LAYERS = ['Suitability', 'ECNES_likely_may', 'ECNES_likely', 'SNES_likely_may', 'SNES_likely', 'MNES_likely_may', 'MNES_likely', 'RHI']
BIO_QUALITY_LAYER = 'MNES_likely'
'''
One of 'Suitability', 'ECNES_likely_may', 'ECNES_likely', 'SNES_likely_may', 'SNES_likely', 'MNES_likely_may', 'MNES_likely', 'RHI'.
    - 'Suitability': use the Zonation algorith to compute quanlity score over 10k species.
    - '*NES_likely|may': use the Zonation algorith to compute quanlity score over the SNES/ECNES species community.
    - 'RHI': DCCEEW's published Relative Habitat Importance layer.

Essentially, the biodiversity quality layer determines how important (0-100) a cell is to the overall biodiversity value. 
    - By choosing 'Suitability' layer, you assume that the overal biodiversity is determined by considering all species (plants, 
      mamals, amphibians, birds, reptiles, etc). 
    - If choosing one of the 'SNES_likely|may' layers, you assume that the overal biodiversity is determined by species 
      related to the Environment Protection and Biodiversity Conservation Act 1999 (EPBC Act). 
    - If choosing one of the 'ECNES_likely|may' layers, you assume that the overal biodiversity is determined by ecological
      communities related to the Environment Protection and Biodiversity Conservation Act 1999 (EPBC Act).
    - If choosing one of the 'MNES_likely|may' layers, you assume that the overal biodiversity is determined by both SNES
      and ECNES species communities, where each community is treated as a species, and the Zonation algorith sees each
      community and species equally important.
    - If choosing 'RHI', you assume the overal biodiversity is determined by EPBC-listed threatened and migratory species,
      as ranked by DCCEEW themselves. Unlike the layers above this one is not produced in-house: DCCEEW ran Zonation 5
      (CAZMAX) over the SNES 'likely to occur' distributions and published the finished 0-100 raster, which LUTO uses as
      is. Note it carries those native 0-100 units while the layers above run 0-1, so RHI's absolute bio-quality scores
      read ~100x theirs in the outputs. Nothing in the model is affected: the GBF2 mask comes from the 'RHI' sheet of
      Biodiversity_conserve_performance.xlsx, which is on the same 0-100 scale, and the solver rescales the bio-quality
      matrices by their own maximum before use.

To understand the 'Suitability' layer, refer to
    https://academic.oup.com/gigascience/article/doi/10.1093/gigascience/giae002/7619364
To understand the 'SNES_likely|may' and 'ECNES_likely|may' layers, refer to
    https://www.dcceew.gov.au/environment/environmental-information-data/databases-applications/snes
    https://www.dcceew.gov.au/environment/environmental-information-data/databases-applications/ecnes
To understand the 'RHI' layer, refer to
    https://fed.dcceew.gov.au/datasets/0732222780f84f9387975f26e0bc5af6/about

'''


# Connectivity source source
CONNECTIVITY_SOURCE = 'NCI'                 # 'DCCEEW_NCI', 'NATURAL_AREA_CONNECTIVITY' or 'NONE'
'''
The connectivity source is the source of the connectivity score used to weigh the raw biodiversity priority score.
This score is normalised between 0 (fartherst) and 1 (closest).
Can be either 'NCI' or 'DWI'.
- if 'NCI' is selected, the connectivity score is sourced from the DCCEEW's National Connectivity Index (v3.0).
- if 'DWI' is selected, the connectivity score is calculated as distance to the nearest area of natural land as mapped
        by the National Land Use Map of Australia.
- if 'NONE' is selected, the connectivity score is not used in the biodiversity calculation.
'''

# Connectivity score importance
CONNECTIVITY_LB = 0.7                       # Avaliable values are [0.5, 0.6, 0.7, 0.8, 0.9]
'''
The relative importance of the connectivity score in the biodiversity calculation. Used to scale the raw biodiversity score.
I.e., the lower bound of the connectivity score for weighting the raw biodiversity priority score is CONNECTIVITY_LB.
'''


# Habitat condition data source
HCAS_CONTRIBUTION_PERCENTILE = 'CSV_DEFINED'                  # One of ['10', '25', '50', '75', '90'], 'CSV_DEFINED', or 'AG_UNIFORM'
HCAS_AG_UNIFORM_CONTRIBUTION = 0.0                             # Only under 'AG_UNIFORM': the one habitat contribution (0-1) given to EVERY
                                                               # modified / cropped agricultural land use (data.LU_MODIFIED_LAND). Natural land
                                                               # (Beef/Dairy/Sheep - natural land, Unallocated - natural land) keeps the CSV's
                                                               # AG_UNIFORM column values (0.7 / 1.0). 0.0 = only non-ag and native land count.
'''
Different land-use types have different biodiversity degradation impacts. We calculated the percentiles values of HCAS (indicating the
suitability for wild animals ranging between 0-1) for each land-use type.Avaliable percentiles is one of ['10', '25', '50', '75', '90'].

For example, the 50th percentile for 'Beef - Modified land' is 0.22, meaning this land retains 22% biodiversity score compared
to undisturbed natural land.
'''


# Biodiversity value under default late dry season savanna fire regime
BIO_CONTRIBUTION_LDS = 0.75
''' For example, 0.8 means that all areas in the area eligible for savanna burning have a biodiversity value of 0.8 * the raw biodiv value
    (due to hot fires etc). When EDS sav burning is implemented the area is attributed the full biodiversity value (i.e., 1.0).
'''

# Non-agricultural biodiversity parameters 
BIO_CONTRIBUTION_ENV_PLANTING = 0.7
BIO_CONTRIBUTION_CARBON_PLANTING_BLOCK = 0.12
BIO_CONTRIBUTION_CARBON_PLANTING_BELT = 0.12
BIO_CONTRIBUTION_RIPARIAN_PLANTING = 1.0
BIO_CONTRIBUTION_AGROFORESTRY = 0.7
BIO_CONTRIBUTION_BECCS = 0
BIO_CONTRIBUTION_DESTOCKING = 0.75  # If 'GAP', uses BIO_HABITAT_CONTRIBUTION_LOOK_UP difference; if set to a number (e.g. 0.75), overrides with a fixed scalar
'''
The benefit of each non-agricultural land use to biodiversity is set as a proportion to the raw biodiversity priority value.
For example, if the raw biodiversity priority value is 0.6 and the benefit is 0.8, then the biodiversity value
will be 0.6 * 0.8 = 0.48.
'''




# ---------------------- GBF3 parameters ----------------------

GBF3_NVIS_TARGET = 'off'           # 'off', 'medium', 'high', 'SPECIFIED', or 'CSV_DEFINED'
'''
Target 3 of the Kunming-Montreal Global Biodiversity Framework (NVIS):
protect and manage vegetation groups using the National Vegetation Information System.

- if 'off' is selected, turn off the GBF-3 NVIS target for biodiversity.
- if 'medium' is selected, the conservation target is set to 30% by 2030 and 30% by 2050 for each vegetation group.
- if 'high' is selected, the conservation target is set to 30% by 2030 and 50% by 2050 for each vegetation group.
- if 'CSV_DEFINED' is selected, targets are kept from the input CSV; only groups with all year targets > 0 are constrained.
- if 'SPECIFIED' is selected, GBF3_NVIS_SEL_REGION_TARGETS is a dict {region: {year: pct}} and every group in a region
  gets that region's level (region-specific uniform targets; in AUSTRALIA mode key it 'AUSTRALIA').
Level presets apply to ALL vegetation groups at the configured region mode (no CSV-target filter).
(No 'low' level — a 0% target only makes sense for GBF2's degraded-areas logic.)
'''


GBF3_TARGETS_DICT = {
    'off':     None,
    'medium':  {2030: 30, 2050: 30},
    'high':    {2030: 30, 2050: 50},
    'SPECIFIED':    None,                    # levels come from GBF3_NVIS_SEL_REGION_TARGETS (a dict in this mode)
    'CSV_DEFINED': None
}


# Per-(region, vegetation group) GBF3 NVIS target overrides — the NVIS counterpart of
# GBF4_SNES_TARGETS_OVERRIDE / GBF4_ECNES_TARGETS_OVERRIDE. Applied AFTER the uniform
# GBF3_TARGETS_DICT and independent of GBF3_NVIS_TARGET mode, so it works with 'medium', 'high' and
# 'CSV_DEFINED' alike. Maps (region, group) -> {year: pct}. Empty = no override (a byte-identical
# no-op, so every run that does not set it is unaffected).
#
# Lets the vegetation floor carry a target level that GBF3_TARGETS_DICT has no preset for — there is
# no 15/25 option, so halving NVIS alongside SNES/ECNES is only expressible this way short of the
# CSV_DEFINED Excel route.
#
# NOTE ON KEYS: in AUSTRALIA region mode the rows are relabelled 'AUSTRALIA' BEFORE this is applied,
# so keys must use ('AUSTRALIA', group) there, not the NRM region name. In NRM mode use the NRM
# region name, exactly as GBF3_NVIS_SEL_REGION_TARGETS does.
GBF3_NVIS_TARGETS_OVERRIDE = {}


GBF3_NVIS_TARGET_CLASS  = 'NVIS_MVG'             # 'NVIS_MVG', 'NVIS_MVS'
'''
The National Vegetation Information System (NVIS) provides the 100m resolution information on
the distribution of vegetation (~30 primary group layers, or ~90 subgroup layers) across Australia.
Also used as the class selector for IBRA bioregion layers when GBF3_NVIS_REGION_MODE = 'IBRA_REG'.
'''


GBF3_NVIS_REGION_MODE = 'NRM'                    # 'AUSTRALIA', 'NRM', or 'IBRA_REG'
'''
Controls the spatial resolution of GBF3 NVIS constraints.
 - 'AUSTRALIA' → nationwide NVIS vegetation-group targets (existing behaviour, default)
 - 'NRM'       → per-NRM-region NVIS targets (masked to selected NRM regions)
 - 'IBRA_REG'  → IBRA bioregion targets (bio_GBF3_NVIS_MVG/MVS.nc + IBRA Excel file)
'''

GBF3_NVIS_SEL_REGION_TARGETS = {
    'North East':      {2030: 30, 2050: 50, 2100: 50},
    'Goulburn Broken': {2030: 30, 2050: 50, 2100: 50},
}
'''
The NRM regions to enforce GBF3 NVIS constraints for (keys must match REGION_NRM_NAME; used when
GBF3_NVIS_REGION_MODE = 'NRM') and, under GBF3_NVIS_TARGET = 'SPECIFIED', each region's uniform target
{year: pct} (all three years required). In the preset / CSV_DEFINED modes only the keys are used, so a
plain list of region names is also accepted there. In AUSTRALIA region mode key it 'AUSTRALIA'.
'''

GBF3_NVIS_MIN_AREA_HA = 100
'''
Drop every (region, group) whose IN_LUTO_HA (restorable habitat inside the LUTO study area) is below this
many hectares: a constraint with LHS ≈ 0 is structurally infeasible. Applies in every mode and region mode.
'''



# ------------------------------- GBF4 Parameters -------------------------------
'''
Target 4 of the Kunming-Montreal Global Biodiversity Framework (GBF) aims to 
halt the extinction of known threatened species, protect genetic diversity, 
and manage human-wildlife interactions
'''


GBF4_TARGET_SNES  = 'off'           # 'off', 'medium', 'high', 'SPECIFIED', or 'CSV_DEFINED'
GBF4_TARGET_ECNES = 'off'           # 'off', 'medium', 'high', 'SPECIFIED', or 'CSV_DEFINED'
'''
'off'               — GBF4 SNES/ECNES constraints disabled.
'CSV_DEFINED'      — targets read from the input CSV as-is; only species/communities with a
                      defined TARGET_LEVEL_2030 > 0 in the CSV are selected.
'SPECIFIED'              — same species/communities as 'CSV_DEFINED', with REGION-SPECIFIC uniform levels:
                      GBF4_{SNES,ECNES}_SEL_REGION_TARGETS is then a dict {region: {year: pct}} whose keys
                      are the selected regions (in AUSTRALIA mode: {'AUSTRALIA': {...}}). Makes automated
                      task runs easy, e.g. a sensitivity sweep over target levels per CMA.
'medium'/'high'     — ALL species/communities at the configured presence/region are selected
                      (no CSV-target filter), and every TARGET_LEVEL_{year} column is set to the
                      uniform level preset in GBF4_{SNES,ECNES}_TARGETS_DICT
                      (same convention as GBF2_TARGETS_DICT; no 'low' level — a 0% target
                      only makes sense for GBF2's degraded-areas logic).
'''

GBF4_SNES_PRESENCE_CLASS  = 'LIKELY'  # 'LIKELY', 'LIKELY_AND_MAYBE'
GBF4_ECNES_PRESENCE_CLASS = 'LIKELY'  # 'LIKELY', 'LIKELY_AND_MAYBE'

# Uniform target presets (percent) keyed by GBF4_TARGET_{SNES,ECNES} mode.
# Only consulted when the target setting is 'medium'/'high' ('SPECIFIED' levels live in *_SEL_REGION_TARGETS).
GBF4_SNES_TARGETS_DICT  = {
    'medium': {2030: 30, 2050: 30, 2100: 30},
    'high':   {2030: 30, 2050: 50, 2100: 50},
}
GBF4_ECNES_TARGETS_DICT = {
    'medium': {2030: 30, 2050: 30, 2100: 30},
    'high':   {2030: 30, 2050: 50, 2100: 50},
}

# Per-(region, SCIENTIFIC_NAME) SNES target overrides, applied AFTER the uniform dict
# (and independent of GBF4_TARGET_SNES mode). Maps (region, species) -> {year: pct}.
# Lets a few species carry a different target from the rest. Empty = no override.
GBF4_SNES_TARGETS_OVERRIDE = {}

# Per-(region, COMMUNITY) ECNES target overrides — the ECNES counterpart of
# GBF4_SNES_TARGETS_OVERRIDE, applied AFTER the uniform dict and independent of
# GBF4_TARGET_ECNES mode. Maps (region, community) -> {year: pct}. Empty = no override.
# Targets above a community's ATTAINABLE_LEVEL are clamped to it in
# get_GBF4_ECNES_target_inside_LUTO_by_year() (note ECNES clamps at attainable exactly,
# with no safety margin — there is no ECNES analogue of GBF4_SNES_CAP_MARGIN).
GBF4_ECNES_TARGETS_OVERRIDE = {}

# Safety margin (percentage points) subtracted from each species' ATTAINABLE_LEVEL when the
# interpolated SNES target is clamped in data.get_GBF4_SNES_target_inside_LUTO_by_year(). A target
# pinned exactly at attainable gives zero slack (razor-thin feasible space); this keeps a small
# buffer. Effective cap per species = ATTAINABLE_LEVEL - GBF4_SNES_CAP_MARGIN.
GBF4_SNES_CAP_MARGIN = 2.0

GBF4_SNES_REGION_MODE       = 'AUSTRALIA'                    # 'AUSTRALIA' or 'NRM'
GBF4_SNES_SEL_REGION_TARGETS  = {
    'North East':      {2030: 30, 2050: 50, 2100: 50},
    'Goulburn Broken': {2030: 30, 2050: 50, 2100: 50},
}
'''
Controls the spatial resolution of GBF4 SNES constraints.
 - 'AUSTRALIA' → nationwide targets (existing behaviour, default)
 - 'NRM'       → per-NRM-region targets from NRM target files
GBF4_SNES_SEL_REGION_TARGETS: {region: {year: pct}} — the keys select the NRM regions (mode = 'NRM') and,
under GBF4_TARGET_SNES = 'SPECIFIED', the values are each region's uniform level (all three years required).
The preset / CSV_DEFINED modes use only the keys, so a plain list of names is also accepted there.
AUSTRALIA mode: {'AUSTRALIA': {...}}.
'''
GBF4_SNES_MIN_AREA_HA = 100     # drop (region, species) pairs with IN_LUTO_HA below this (LHS ≈ 0 → infeasible)

GBF4_ECNES_REGION_MODE      = 'AUSTRALIA'                   # 'AUSTRALIA' or 'NRM'
GBF4_ECNES_SEL_REGION_TARGETS = {
    'North East':      {2030: 30, 2050: 50, 2100: 50},
    'Goulburn Broken': {2030: 30, 2050: 50, 2100: 50},
}
'''
Controls the spatial resolution of GBF4 ECNES constraints.
 - 'AUSTRALIA' → nationwide targets (existing behaviour, default)
 - 'NRM'       → per-NRM-region targets from NRM target files
GBF4_ECNES_SEL_REGION_TARGETS: {region: {year: pct}} — the keys select the NRM regions (mode = 'NRM') and,
under GBF4_TARGET_ECNES = 'SPECIFIED', the values are each region's uniform level (all three years required).
The preset / CSV_DEFINED modes use only the keys, so a plain list of names is also accepted there.
AUSTRALIA mode: {'AUSTRALIA': {...}}.
'''
GBF4_ECNES_MIN_AREA_HA = 100    # drop (region, community) pairs with IN_LUTO_HA below this (LHS ≈ 0 → infeasible)



# -------------------------------- Climate change impacts on biodiversity -------------------------------
GBF8_TARGET = 'off'           # 'off', 'medium', 'high', or 'CSV_DEFINED'
'''
Target 8 of the Kunming-Montreal Global Biodiversity Framework (GBF) aims to
reduce the impacts of climate change on biodiversity and ecosystems.

'off'               — GBF8 constraints disabled.
'CSV_DEFINED'       — targets read from the hand-filled USER_DEFINED_TARGET_PERCENT_{year}
                      columns of BIODIVERSITY_GBF8_TARGET.csv; only species with all three
                      year targets defined and > 0 are selected.
'medium'/'high'     — ALL species in the CSV are selected and given the uniform level preset
                      from GBF8_TARGETS_DICT (same convention as GBF2_TARGETS_DICT; no 'low'
                      level — a 0% target only makes sense for GBF2's degraded-areas logic).
                      NOTE: that is every species in the file (~10,600) — a far larger
                      constraint set than any historical GBF8 run.
'''

# Uniform target presets (percent) keyed by GBF8_TARGET level.
# Only consulted when GBF8_TARGET is 'medium'/'high'.
GBF8_TARGETS_DICT = {
    'medium': {2030: 30, 2050: 30, 2100: 30},
    'high':   {2030: 30, 2050: 50, 2100: 50},
}




# ---------------------------------------------------------------------------- #
# Other parameters
# ---------------------------------------------------------------------------- #

# Non-ag output coding. Non-agricultural land uses will appear on the land use map offset by this amount (e.g. land use 0 will appear as 100)
NON_AGRICULTURAL_LU_BASE_CODE = 100

# Number of decimals to round any value; designed to remove insignificant decimals
ROUND_DECIMALS = 6


""" NON-AGRICULTURAL LAND USES (indexed by k)
0: 'Environmental Plantings'
1: 'Riparian Plantings'
2: 'Sheep Agroforestry'
3: 'Beef Agroforestry'
4: 'Carbon Plantings (Block)'
5: 'Sheep Carbon Plantings (Belt)'
6: 'Beef Carbon Plantings (Belt)'
7: 'BECCS'
8: 'Destocked - natural land'


DRAINAGE DIVISIONS
 1: 'Tanami-Timor Sea Coast',
 2: 'South Western Plateau',
 3: 'South West Coast',
 4: 'Tasmania',
 5: 'South East Coast (Victoria)',
 6: 'South Australian Gulf',
 7: 'Murray-Darling Basin',
 8: 'Pilbara-Gascoyne',
 9: 'North Western Plateau',
 10: 'South East Coast (NSW)',
 11: 'Carpentaria Coast',
 12: 'Lake Eyre Basin',
 13: 'North East Coast'


RIVER REGIONS
 1: 'ADELAIDE RIVER',
 2: 'ALBANY COAST',
 3: 'ARCHER-WATSON RIVERS',
 4: 'ARTHUR RIVER',
 5: 'ASHBURTON RIVER',
 6: 'AVOCA RIVER',
 7: 'AVON RIVER-TYRELL LAKE',
 8: 'BAFFLE CREEK',
 9: 'BARRON RIVER',
 10: 'BARWON RIVER-LAKE CORANGAMITE',
 11: 'BATHURST-MELVILLE ISLANDS',
 12: 'BEGA RIVER',
 13: 'BELLINGER RIVER',
 14: 'BENANEE-WILLANDRA CREEK',
 15: 'BILLABONG-YANCO CREEKS',
 16: 'BLACK RIVER',
 17: 'BLACKWOOD RIVER',
 18: 'BLYTH RIVER',
 19: 'BORDER RIVERS',
 20: 'BOYNE RIVER',
 21: 'BRISBANE RIVER',
 22: 'BROKEN RIVER',
 23: 'BROUGHTON RIVER',
 24: 'BRUNSWICK RIVER',
 25: 'BUCKINGHAM RIVER',
 26: 'BULLO RIVER-LAKE BANCANNIA',
 27: 'BUNYIP RIVER',
 28: 'BURDEKIN RIVER',
 29: 'BURNETT RIVER',
 30: 'BURRUM RIVER',
 31: 'BUSSELTON COAST',
 32: 'CALLIOPE RIVER',
 33: 'CALVERT RIVER',
 34: 'CAMPASPE RIVER',
 35: 'CAPE LEVEQUE COAST',
 36: 'CARDWELL COAST',
 37: 'CASTLEREAGH RIVER',
 38: 'CLARENCE RIVER',
 39: 'CLYDE RIVER-JERVIS BAY',
 40: 'COAL RIVER',
 41: 'COLLIE-PRESTON RIVERS',
 42: 'CONDAMINE-CULGOA RIVERS',
 43: 'COOPER CREEK',
 44: 'CURTIS ISLAND',
 45: 'DAINTREE RIVER',
 46: 'DALY RIVER',
 47: 'DARLING RIVER',
 48: 'DE GREY RIVER',
 49: 'DENMARK RIVER',
 50: 'DERWENT RIVER',
 51: 'DIAMANTINA-GEORGINA RIVERS',
 52: 'DON RIVER',
 53: 'DONNELLY RIVER',
 54: 'DRYSDALE RIVER',
 55: 'DUCIE RIVER',
 56: 'EAST ALLIGATOR RIVER',
 57: 'EAST COAST',
 58: 'EAST GIPPSLAND',
 59: 'EMBLEY RIVER',
 60: 'ENDEAVOUR RIVER',
 61: 'ESPERANCE COAST',
 62: 'EYRE PENINSULA',
 63: 'FINNISS RIVER',
 64: 'FITZMAURICE RIVER',
 65: 'FITZROY RIVER (QLD)',
 66: 'FITZROY RIVER (WA)',
 67: 'FLEURIEU PENINSULA',
 68: 'FLINDERS-CAPE BARREN ISLANDS',
 69: 'FLINDERS-NORMAN RIVERS',
 70: 'FORTESCUE RIVER',
 71: 'FORTH RIVER',
 72: 'FRANKLAND-DEEP RIVERS',
 73: 'FRASER ISLAND',
 74: 'GAIRDNER',
 75: 'GASCOYNE RIVER',
 76: 'GAWLER RIVER',
 77: 'GLENELG RIVER',
 78: 'GOOMADEER RIVER',
 79: 'GORDON RIVER',
 80: 'GOULBURN RIVER',
 81: 'GOYDER RIVER',
 82: 'GREENOUGH RIVER',
 83: 'GROOTE EYLANDT',
 84: 'GWYDIR RIVER',
 85: 'HASTINGS RIVER',
 86: 'HAUGHTON RIVER',
 87: 'HAWKESBURY RIVER',
 88: 'HERBERT RIVER',
 89: 'HINCHINBROOK ISLAND',
 90: 'HOLROYD RIVER',
 91: 'HOPKINS RIVER',
 92: 'HUNTER RIVER',
 93: 'HUON RIVER',
 94: 'ISDELL RIVER',
 95: 'JARDINE RIVER',
 96: 'JEANNIE RIVER',
 97: 'JOHNSTONE RIVER',
 98: 'KANGAROO ISLAND',
 99: 'KARUAH RIVER',
 100: 'KEEP RIVER',
 101: 'KENT RIVER',
 102: 'KIEWA RIVER',
 103: 'KING EDWARD RIVER',
 104: 'KING ISAND',
 105: 'KING-HENTY RIVERS',
 106: 'KINGSTON COAST',
 107: 'KOLAN RIVER',
 108: 'KOOLATONG RIVER',
 109: 'LACHLAN RIVER',
 110: 'LAKE EYRE',
 111: 'LAKE TORRENS-MAMBRAY COAST',
 112: 'LENNARD RIVER',
 113: 'LIMMEN BIGHT RIVER',
 114: 'LITTLE RIVER',
 115: 'LIVERPOOL RIVER',
 116: 'LOCKHART RIVER',
 117: 'LODDON RIVER',
 118: 'LOGAN-ALBERT RIVERS',
 119: 'LOWER MALLEE',
 120: 'LOWER MURRAY RIVER',
 121: 'MACLEAY RIVER',
 122: 'MACQUARIE-BOGAN RIVERS',
 123: 'MACQUARIE-TUGGERAH LAKES',
 124: 'MANNING RIVER',
 125: 'MAROOCHY RIVER',
 126: 'MARY RIVER (NT)',
 127: 'MARY RIVER (QLD)',
 128: 'MERSEY RIVER',
 129: 'MILLICENT COAST',
 130: 'MITCHELL-COLEMAN RIVERS (QLD)',
 131: 'MITCHELL-THOMSON RIVERS',
 132: 'MOONIE RIVER',
 133: 'MOORE-HILL RIVERS',
 134: 'MORNING INLET',
 135: 'MORNINGTON ISLAND',
 136: 'MORUYA RIVER',
 137: 'MOSSMAN RIVER',
 138: 'MOYLE RIVER',
 139: 'MULGRAVE-RUSSELL RIVERS',
 140: 'MURCHISON RIVER',
 141: 'MURRAY RIVER (WA)',
 142: 'MURRAY RIVERINA',
 143: 'MURRUMBIDGEE RIVER',
 144: 'MYPONGA RIVER',
 145: 'McARTHUR RIVER',
 146: 'NAMOI RIVER',
 147: 'NICHOLSON-LEICHHARDT RIVERS',
 148: 'NOOSA RIVER',
 149: 'NORMANBY RIVER',
 150: 'NULLARBOR',
 151: "O'CONNELL RIVER",
 152: 'OLIVE-PASCOE RIVERS',
 153: 'ONKAPARINGA RIVER',
 154: 'ONSLOW COAST',
 155: 'ORD-PENTECOST RIVERS',
 156: 'OTWAY COAST',
 157: 'OVENS RIVER',
 158: 'PAROO RIVER',
 159: 'PIEMAN RIVER',
 160: 'PINE RIVER',
 161: 'PIONEER RIVER',
 162: 'PIPER-RINGAROOMA RIVERS',
 163: 'PLANE CREEK',
 164: 'PORT HEDLAND COAST',
 165: 'PORTLAND COAST',
 166: 'PRINCE REGENT RIVER',
 167: 'PROSERPINE RIVER',
 168: 'RICHMOND RIVER',
 169: 'ROBINSON RIVER',
 170: 'ROPER RIVER',
 171: 'ROSIE RIVER',
 172: 'ROSS RIVER',
 173: 'RUBICON RIVER',
 174: 'SALT LAKE',
 175: 'SANDY CAPE COAST',
 176: 'SANDY DESERT',
 177: 'SETTLEMENT CREEK',
 178: 'SHANNON RIVER',
 179: 'SHOALHAVEN RIVER',
 180: 'SHOALWATER CREEK',
 181: 'SMITHTON-BURNIE COAST',
 182: 'SNOWY RIVER',
 183: 'SOUTH ALLIGATOR RIVER',
 184: 'SOUTH COAST',
 185: 'SOUTH GIPPSLAND',
 186: 'SOUTH-WEST COAST',
 187: 'SPENCER GULF',
 188: 'STEWART RIVER',
 189: 'STRADBROKE ISLAND',
 190: 'STYX RIVER',
 191: 'SWAN COAST-AVON RIVER',
 192: 'SYDNEY COAST-GEORGES RIVER',
 193: 'TAMAR RIVER',
 194: 'TORRENS RIVER',
 195: 'TORRES STRAIT ISLANDS',
 196: 'TOWAMBA RIVER',
 197: 'TOWNS RIVER',
 198: 'TULLY-MURRAY RIVERS',
 199: 'TUROSS RIVER',
 200: 'TWEED RIVER',
 201: 'UPPER MALLEE',
 202: 'UPPER MURRAY RIVER',
 203: 'VICTORIA RIVER-WISO',
 204: 'WAKEFIELD RIVER',
 205: 'WALKER RIVER',
 206: 'WARD RIVER',
 207: 'WARREGO RIVER',
 208: 'WARREN RIVER',
 209: 'WATER PARK CREEK',
 210: 'WENLOCK RIVER',
 211: 'WERRIBEE RIVER',
 212: 'WHITSUNDAY ISLANDS',
 213: 'WILDMAN RIVER',
 214: 'WIMMERA RIVER',
 215: 'WOLLONGONG COAST',
 216: 'WOORAMEL RIVER',
 217: 'YANNARIE RIVER',
 218: 'YARRA RIVER'}
"""