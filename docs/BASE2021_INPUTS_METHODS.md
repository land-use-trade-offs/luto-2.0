# 2021-base input layers: methods note

This note documents how the 2021-base input layers are built from public sources, why each rule was chosen, what was measured against it, and how each layer changes against the 2010 layer it is to replace. It accompanies the build scripts in `input/Nick_build/scripts/`, and it is written for reviewers of those changes. The comparison tables quoted here are produced by `input/Nick_build/scripts/compare_inputs.py`, which reruns from the built inputs, so the evidence updates whenever a layer changes.

LUTO continues to run on the 2010 base until the base-year switch; the layers described here are written to `input/Nick_build/`, one folder per script, and are read only after that switch. Monetary values are in FY2020-21 dollars; emissions are in t CO2e under AR5 GWP100; 'FY2021' means the financial year ending June 2021. Figures the comparison tool computes from the built inputs cite the table they come from; ranges such as 2016-21 are written with a hyphen.

## 1. How the inputs are built

Each layer is built by one standalone script, run from the repository root in the order given in `input/Nick_build/scripts/README.md`; LUTO's dataprep does not call them. Rather than copying prepared files, every tool downloads its sources from their public URLs, checks each against a pinned sha256 and writes a provenance file recording the source pins, the hash of every existing input file it reads, the script hash, the package versions and the hash of each output.

| Tool | Builds | Main public sources |
|---|---|---|
| `nlum_map.py` | 2010-11, 2015-16 and 2020-21 class allocation, SA2 key, definition crosswalk | ABARES NLUM v7.1 rasters and probability surfaces; ABS ASGS Edition 3 SA2 2021 |
| `pasture_productivity.py` | `pasture_kg_dm_ha.h5` | AussieGRASS total pasture growth, FY2016-25 |
| `livestock_map.py` | `lumap.h5`, `lmmap.h5`, `stubble_dse_ha.h5`, `lvstk_k.h5` | ABS 7121.0 2020-21; NIR 2024 livestock activity table and national enteric CH4 |
| `eligibility.py` | `x_mrj.npy`, `xmrj_zone_layers.h5` | NLUM v7.1 2020-21 (irrigation rule: 2010-11, 2015-16, 2020-21); ABS SA2 2021; SILO rainfall 1961-2020; Australian Irrigation Areas v1A |
| `ghg_factors.py` | `agGHG_lvstk.h5`, `agGHG_irrpast.h5`, `agGHG_lvstk_off_land.csv`; in `agGHG_crops.h5` the SOIL fields and the crop residue (`CO2E_KG_HA_RESIDUES`), field burning (`CO2E_KG_HA_BURNING`) and rice CH4 (`CO2E_KG_HA_RICE_CH4`) fields | NIR 2024 by state (National Greenhouse Accounts API); NIR activity tables; NIR 2024 Volume 2 Table A5.5.8.1 |
| `hir_layer.py` | `hir_rate_layer_gross.h5` | CER Australian Carbon Credit Unit (ACCU) project register; Carbon Estimation Area (CEA) files via the caps tool |
| `clearing_layer.py` | `clearing_layer.h5` | NIR 2024 LULUCF activity table |
| `crop_economics.py` | `agec_crops.h5`, final `agGHG_crops.h5` | ABS 7121.0, 7503.0, Australian Agriculture, 4618.0, CPI; ABARES Crop Report and Farm Data Portal |
| `livestock_economics.py` | `agec_lvstk.h5`, `water_delivery_price.h5`, `yieldincreases_bau2022.csv` (flat wool trend) | ABS 7215.0, 7503.0, 4618.0; ABARES Agricultural Commodities and Farm Data Portal; DAFF live animal exports; AWI; Dairy Australia |
| `cci_fill.py` | eight `climate_change_impacts_*.h5` files | none (a fill of the 2010-build files) |
| `water_licence_price.py` | `water_licence_price.h5` | Ricardo Water Markets Report 2020-21; Queensland permanent water trading report 2020-21; NSW 71Q entitlement trade table; Tasmanian Irrigation scheme offers; CPI |
| `water_domestic.py` | `Water_Use_Domestic.csv` | ABS Water Account, Australia, 2020-21 (4610.0) |

+ **Pins.** Every source is pinned to the bytes retrieved in September and October 2026. A download that differs from its pin stops the run and is set aside for review; pins are never changed automatically. Sources that change without notice are handled explicitly: the CER project register and the National Greenhouse Accounts API are pinned response by response, and a live check compares them with their pins; the AussieGRASS grids are pinned on their decoded pixel values, grid layout and data type, because LongPaddock rebuilds recent issues with new metadata and identical values.
+ **Cache.** Raw downloads are kept in one content-addressed cache, so a rebuild runs offline from the cache and gives the same outputs.
+ **NetCDF.** Every tool that writes a LUTO input array also writes it as NetCDF in the layout of the `jinzhu` branch's input files (`input/Nick_build/scripts/nc_layout.py`): one `cell` dimension in the `.h5` row order; single layers and 1-D columns as one variable each, named as the column; wide tables as one 2-D variable named after the file in capitals, dims `(column, cell)`, with a text coordinate for each column level; `x_mrj` as `int8` marked boolean over `(lm, cell, lu)`; NETCDF4, zlib 9, one chunk per column. Each file is read back and compared with the arrays written, and recorded by a content hash over decoded values and writer attributes rather than file bytes. The `.h5`, `.npy` and `.csv` outputs are written as before.
+ **Inherited inputs.** The tools also read 38 old-format files of the 2010 build, kept as frozen copies in `input/Nick_build/inherited_2010/`, so the 2021-base layers are built from public sources together with those files, not from public sources alone. Section 9 lists each with what it supplies and what replaces it.
+ **Hand-entered values.** Values that come from documents rather than machine-readable files (the NIR Volume 2 crop areas, wool and milk by state, drinking water per head, rule constants) are held in one marked reference table per tool, each with its source, page and file hash.
+ **Reproducibility.** Every layer was rebuilt from the cache and again from live downloads and compared with the evidence it replaces: `.npy` and CSV outputs byte for byte, HDF5 outputs by the content hash of every dataset together with its keys, storage format and cell order (HDF5 files carry a write timestamp, so their bytes differ between runs).

## 2. Land-use map

The 2021-base map takes its land use from the NLUM v7.1 2020-21 release and keeps LUTO's 2010 study area, so that change against the 2010 map can be read as change in land use rather than change in the cell set. A cell is in the study area in every year if, and only if, it is in the 2010 map (`lumap` not -1).

+ **Definition crosswalk.** Each ALUM v8 tertiary class maps to a LUTO (land use, irrigation) pair by its definition in the ALUM v8 handbook and the NLUM v7.1 metadata, not by overlaying the 2010 map. The crosswalk is a reference table of 146 classes in `nlum_map.py`, each row with its handbook page, confidence and any policy flag, and is written out as `xwalk_definition.csv`. The earlier cell-level overlay had sent cotton and hay to winter cereals; the definition crosswalk gives 2020-21 cotton 437 kha and hay 2,482 kha where the overlay gave none.
+ **One-to-many classes.** Cereals, oilseeds, legumes and tree fruit split over their LUTO candidates in proportion to the NLUM probability surfaces at each pixel centre, normalised over the candidates; pixels with no probability information take their class's split in their SA2, else state, else nationally.
+ **Area-preserving allocation.** Every 250 m pixel is assigned to the 0.01 degree LUTO cell containing it, and whole cells are then allocated to classes within each SA2 (ASGS Edition 3, 2021) so that each class's area in the SA2 is kept to within one cell: a floor quota in descending share order, then the largest remainders. Rather than giving each cell its majority class, this keeps minority classes on the map.
+ **Held cells.** Footprint cells with no v7.1 data, or whose class is outside LUTO's agricultural land uses, keep their 2010 land use, except 2010 crop cells, which are retyped as unallocated natural land (natural classes, including lakes, rivers, marshes and estuaries) or unallocated modified land (modified and intensive classes, including reservoirs, dams and channels). Classes 1.2.5 and 1.3.3 are held outside agriculture.
+ **Grazing.** A grazing cell that was grazing in the 2010-11 allocation keeps its 2010 livestock land use (natural or modified, as the class requires); new grazing cells are typed in the livestock step (section 3). Unassigned modified pasture keeps the class's irrigation flag, so irrigated pasture area is preserved.

**Measured alternatives.** The same allocation with the state as the group keeps national class totals closer (2.2 kha against 35.5 kha of 465 Mha) but leaves SA2 totals off by up to 2,113 cells, which would move crop area between SA2s with different economics and livestock targets. Majority rule matches the pixels at cell level best (pixel agreement 0.914 against 0.909) but erases minority classes: 21.7 Mha of national class area moves, and dry cotton falls to 64.6 kha against a native 177.8 kha. The SA2 rule loses 66.1 kha to rounding in 5,222 (SA2, class) rows with a target under one cell, which falls hardest on thin classes (dry apples -69 per cent, dry pears -49 per cent, dry citrus -35 per cent nationally).

**Change against the 2010 map.** The control is the same method applied to NLUM v7.1 2010-11, which separates the change due to method from the change in land use between 2010-11 and 2020-21:

| Group (Mha) | 2010 map | 2010-11 control | 2020-21 | 2021 map | Method change | Land-use change |
|---|---:|---:|---:|---:|---:|---:|
| Crops, dry | 27.83 | 31.16 | 40.70 | 40.70 | +3.33 | +9.54 |
| Crops, irrigated | 1.47 | 1.21 | 2.11 | 2.11 | -0.26 | +0.90 |
| Grazing, modified land, dry | 58.41 | 72.71 | 64.07 | 63.97 | +14.30 | -8.65 |
| Grazing, modified land, irrigated | 0.63 | 0.61 | 0.56 | 0.56 | -0.02 | -0.05 |
| Grazing, natural land | 97.88 | 93.76 | 99.54 | 99.48 | -4.12 | +5.78 |
| Unallocated land | 278.86 | 265.62 | 258.11 | 258.26 | -13.23 | -7.51 |
| Non-agricultural land (study area held) | 303.76 | 303.76 | 303.76 | 303.76 | 0.00 | 0.00 |

Method change is the 2010-11 control less the 2010 map; land-use change is 2020-21 less the 2010-11 control; the 2021 map differs from 2020-21 only by the livestock typing (section 3), which leaves 0.15 Mha of grazing unallocated. Source: `t2_landuse_area_state_class.csv`. Most of the rise in cropping land (+12.9 Mha against the 2010 map) is real change between the two NLUM years (+10.4 Mha) rather than method (+3.1 Mha); the method moves 14.3 Mha into modified grazing, broadly matched by 13.2 Mha out of unallocated land and 4.1 Mha out of natural grazing, which is consistent with NLUM v7.1 mapping modified pasture where the 2010 map had unallocated or natural land (the cell-level transitions are not decomposed here).

## 3. Livestock head, stubble and stocking calibration

The livestock step types each grazing cell as beef, sheep or dairy and calibrates a stocking factor `k` so that head on the map matches the NIR FY2021 state totals, with every head not placed on a cell held in a named row.

+ **Feed ratings.** Head converts to dry-sheep equivalents at national FY2021 intake per head, derived from NIR enteric methane per head with the NIR equations (Charmley et al. 2015 for cattle; Howden et al. 1994 for sheep, flock-weighted over six classes): beef 6.670, sheep 0.887 and dairy 12.678 kg DM per head per day. These replace the 2010 ratings of 8, 1.5 and 17.
+ **Targets.** Each SA2's head by type is the ABS 7121.0 2020-21 SA2 count scaled by the state's NIR-to-ABS ratio. The typing uses these targets. NIR's beef total is pasture cattle only while ABS meat cattle include feedlot cattle, so `k` is calibrated on beef scaled from pasture head: each SA2 first loses an estimate of its feedlot head (estimated: the NIR state feedlot head, about 1.06 M nationally, spread over SA2s by their share of the state's NLUM 2020-21 feedlot area). The subtraction stops at the SA2's ABS head and, where the SA2 has beef cells, where `k` on them would fall below the lower edge of the band; 0.98 M is subtracted. The map's livestock typing is unchanged by it.
+ **Capacity.** Pasture capacity is pasture growth times safe utilisation (natural or modified land) times area, doubled on irrigated cells, per day. Cereal stubble counts toward sheep capacity at 0.411 DSE-years per hectare (about 150 sheep grazing days per hectare, MLA 2020), on every cell of an SA2 with a sheep target.
+ **Typing.** Persistent grazing cells keep their type; other grazing cells take the type furthest below its share of the SA2's DSE (dairy first, on irrigated then higher-capacity modified cells). Where `k` exceeds three times the national median, unallocated natural cells in the SA2 are added to grazing. Persistent cells of a type with no target are retyped or left unallocated.
+ **`k`.** `k` is target over capacity by SA2 and type. Pairs with no base-year stock take the state-and-type head-weighted median; in pairs outside the national band (one-third to three times the national median by type), cells that carry the type keep the SA2's `k` and every other cell takes the state-and-type median.
+ **Town SA2s.** A pair above the band in an SA2 under 20,000 ha is calibrated with its SA3: it takes the SA3's pooled `k` for the type, and the head its cells cannot carry at that `k` moves to the SA3's other SA2s of the type in proportion to their target. ABS counts a holding's stock at its location, and an SA2 that small is a settlement, so a pastoral town's census head would otherwise sit on its few grazing cells (Newman, WA: 18,000 cattle on 1,472 ha, `k` 4,180). Dairy and the other pairs above the band, where the census head is on mapped pasture that the pasture layer rates low, are left as calibrated.

**Measured alternatives.** A state-and-type band flagged 525 rows and 6.96 M head against the national band's 530 rows and 13.51 M head, but left most of Western Australia's sheep above three times the national median unflagged; the national band is used everywhere. Counting cereal stubble as sheep capacity (rather than as a fixed exogenous crop-land flock) moves 6.67 M sheep to cereal land and lifts the share of head in band from 86.7 to 91.7 per cent. Stubble rates of 0.231, 0.411 and 0.493 DSE-years per hectare flag 9.12, 8.47 and 7.77 M head; 0.411 is the published practice value.

**Sensitivity of the typing.** The typing fills each SA2's grazing cells in order of carrying capacity, so a small change in one cell's capacity can reorder the fill across its SA2. Taking another pixel of the same exact-distance tie set in the pasture layer (section 4) changes pasture growth on 168 cells that carry livestock; that alone changes the type of 9,319 cells (beef to sheep 4,656, sheep to beef 4,661, two others), only 10 of them tie cells themselves, with no net change in head in any state (beef and dairy within 0.02 head; sheep moves between pasture and stubble, net zero). Cell-level livestock type is therefore a known sensitivity of the typing; SA2 and state head are not.

**Head against NIR and ABS.** Every head in the NIR state totals is either on a cell or in a named row; the closure is within two head nationally (float32 `k`). Source: `t1_livestock_head_state_type.csv`.

| Head, Australia | Beef | Dairy | Sheep |
|---|---:|---:|---:|
| NIR FY2021 | 25,281,557 | 2,450,979 | 74,329,201 |
| ABS 2020-21 | 22,048,333 | 2,382,840 | 68,047,402 |
| Pasture head on the map | 25,248,755 | 2,444,927 | 61,551,040 |
| Stubble head on cereal cells | 0 | 0 | 12,677,040 |
| Named row: ABS SA2 rounding and suppression | 5,470 | 0 | 0 |
| Named row: SA2 has no grazing cell in scope | 23,274 | 6,052 | 38,665 |
| Named row: no cell of this type in the SA2 | 4,058 | 0 | 62,458 |
| Map head as a share of NIR | 0.9987 | 0.9975 | 0.9986 |

NIR counts 14.7 per cent more beef cattle than ABS nationally (dairy 2.9, sheep 9.2 per cent), so the targets follow ABS's spatial pattern at NIR's level.

## 4. Pasture productivity

The pasture layer is the mean annual total pasture growth from AussieGRASS for FY2016-25, sampled at LUTO cell centres, and it replaces the 2010 layer of the same name.

+ **Invalid pixels.** A pixel that is a lake in any year, or zero in every year, is invalid, on land or not.
+ **Sampling.** Bilinear sampling uses only valid corners, with weights snapped to 0 or 1 within 1e-9 and renormalised; where every weighted corner is invalid the cell takes the nearest valid pixel, and exact-distance ties go to the lowest pixel index so that the result does not depend on the build of the nearest-neighbour library.

**Measured alternatives.** Treating zero-in-every-year pixels as valid on land (a land mask) left 93 stocked cells at zero pasture and gave Murwillumbah sheep a `k` of 26.5; the rule used leaves none and gives 2.1. FY2016-25 is about 4 per cent below the earlier layer nationally and 22 per cent below it in the arid zone.

**Change against the 2010 layer.** Area-weighted pasture growth falls 4.2 per cent nationally, and more in the arid states, while Victoria and New South Wales rise. Source: `t6a_pasture_productivity_state.csv`.

| kg DM/ha/yr, area-weighted | NSW | VIC | QLD | SA | WA | TAS | NT | AUS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2010 layer | 2,099 | 4,546 | 1,775 | 587 | 711 | 2,494 | 1,271 | 1,307 |
| 2021-base layer | 2,134 | 4,838 | 1,718 | 542 | 634 | 2,506 | 1,141 | 1,252 |
| Ratio | 1.017 | 1.064 | 0.968 | 0.922 | 0.891 | 1.005 | 0.897 | 0.958 |

7,511 of the 43,988 nearest-filled cells have an exact-distance tie. Which tied pixel is taken matters little in aggregate: taking another pixel of the same tie set changed 4,396 cells, national pasture growth by +0.0005 per cent and the livestock type of 9,319 cells (beef to sheep 4,656, sheep to beef 4,661), with every state total unchanged. The lowest-index rule makes the choice reproducible.

## 5. Land-use eligibility

Eligibility (`x_mrj`) says which (irrigation, land use) each cell may take; it follows the 2010 recipe with its SA2 pivot rebuilt from the native NLUM 2020-21 pixels.

+ **Occurrence.** A land use is possible in an SA2 if NLUM 2020-21 maps it there on at least 4 pixels (25 ha), counting only pixels with probability information for split classes; grazing is possible for the livestock types present in the SA2 on the 2021-base map. Agricultural pixels outside every SA2 polygon take the nearest SA2.
+ **Gates.** Dry crops need mean April-November SILO rainfall of at least 175 mm over 1991-2020. Irrigation needs an Australian Irrigation Areas v1A polygon at the cell centre or an irrigated 2021-base cell within the 3 x 3 neighbourhood. Irrigated crops also need the cell to be irrigated in at least one NLUM epoch (2010-11, 2015-16 or 2020-21, each read from the class allocation as section 2 defines it): new irrigated cropping is not placed on land NLUM never records as irrigated. This irrigation infrastructure rule is a change to the default behaviour; irrigated pasture is not affected. Unallocated land is always eligible dry. Every cell is eligible for its own land use.

**Measured alternatives.** Thresholds of 1, 2, 4, 8 and 16 pixels keep 10,875, 9,609, 8,521, 7,618 and 6,756 (SA2, land use) pairs and force 2,445, 2,720, 3,020, 3,320 and 3,733 cells to their own land use; every class keeps some eligibility at 4. The 1991-2020 normal removes dry-crop eligibility from 220,723 cells (24.6 Mha), 98.4 per cent of them never cropped in any NLUM year, and adds 23,387 cells.

**Change against the 2010 layer.** Eligibility widens for every group; the table sums eligible area over land uses (a cell eligible for three crops counts three times). Source: `t6b_eligibility_state_lm_lu.csv`.

| Mha, summed over land uses | Eligible 2010 | Eligible 2021 | Gained | Lost |
|---|---:|---:|---:|---:|
| Dry crops | 1,127.1 | 1,402.0 | 529.3 | 254.3 |
| Dry livestock | 2,086.9 | 2,873.8 | 908.2 | 121.3 |
| Irrigated crops | 63.0 | 99.4 | 60.7 | 24.3 |
| Irrigated livestock | 13.1 | 27.8 | 20.7 | 5.9 |

The table predates the irrigation infrastructure rule, which takes irrigated-crop eligibility from 99.4 to 31.6 Mha summed over land uses, and the area of cells eligible for at least one irrigated crop from 13.2 to 3.8 Mha (a third of it in New South Wales).

The gains are broadly indicative of the native 250 m pixels recording land uses in more SA2s than the 2010 aggregated map did, and of livestock types present in more SA2s on the 2021-base map; the dry-crop losses include the move to the 1991-2020 rainfall normal. The gains and losses are not decomposed by cause here.

## 6. Agricultural GHG factors

The GHG factors put NIR FY2021 emissions on LUTO's activity by state, so that base-year agricultural emissions on the 2021-base map reconcile with the inventory.

+ **Livestock.** Enteric, manure management, dung and urine and indirect leaching per head are NIR FY2021 state emissions over NIR state populations (national values where NIR has no population).
+ **Crop soils.** Fertiliser N2O (direct and indirect) per hectare of LUTO cropping land on the 2021-base map, by NIR production system (irrigated and non-irrigated crops, sugar, cotton, horticulture) and state, filled on every eligible cell. NIR reports the ACT inside New South Wales, so NSW is NSW and ACT counted once and ACT cells take the NSW rate.
+ **Crop residues, field burning and rice CH4.** Crop residue N2O (3.D.a.4 direct and 3.D.b.2.v leaching, crop leaves only), field burning (3.F) and rice CH4 (3.C) per hectare of LUTO cropping land, by crop group and state (NSW and ACT counted once), filled on every eligible cell; crops with no NIR leaf take 0, and pasture residues, urea and liming stay exogenous. Placed on the 2021-base map they equal NIR FY2021 by field and state: 1,581,188 t residues, 474,229 t burning and 180,418 t rice CH4.
+ **Irrigated pasture.** One national factor per LUTO hectare of irrigated pasture: NIR FY2021 national irrigated-pasture fertiliser N2O of 133,126 t over 559,989 ha, 237.73 kg CO2e per hectare.
+ **NIR areas.** The NIR crop areas by state and system come from NIR 2024 Volume 2 Table A5.5.8.1 (FY2021 row, pages 200 to 203), which neither the activity tables nor the API carry; the 49 values are a reference table with page and file hash, and two printed typos in the 2019 row (Tasmania, page 202: `279,0844` and `17,4623`) are recorded with their corrections.
+ **Energy and lifecycle fields.** The eight crop energy and lifecycle fields are filled on eligible blank cells from state means at this step and rebuilt in the crop economics step; they are reported, not on a constraint row.
+ **Off-land and exogenous.** Off-land emission intensities come from NIR FY2021 and the 2010 demand projections; the exogenous agriculture series LUTO reads is the GHG targets tool's, not a 2021-base product.

**Base-year emissions against NIR.** On the 2021-base map with the 2021-base factors, modelled agricultural emissions are within 0.14 per cent of NIR FY2021. The difference is split into activity (LUTO's head or area against NIR's, at the state factor) and the factor residual. Source: `t3b_ghg_fy2021_category_state.csv`.

| t CO2e, FY2021, Australia | LUTO | NIR | Difference | Activity | Factor |
|---|---:|---:|---:|---:|---:|
| Enteric fermentation (3.A), beef, dairy, sheep | 56,041,893 | 56,123,180 | -81,287 | -81,287 | 0 |
| Manure management (3.B; dairy with 3.D.a.2.a), all types | 2,020,471 | 2,024,384 | -3,913 | -3,913 | 0 |
| Dung and urine (3.D.a.3) and indirect N2O (3.D.b), all types | 3,369,680 | 3,375,126 | -5,446 | -5,446 | 0 |
| Crop and irrigated-pasture fertiliser N2O | 2,863,952 | 2,863,952 | 0 | +1,731,670 | -1,731,670 |
| Off-land commodities | 2,421,596 | 2,421,596 | 0 | 0 | 0 |
| All rows | 66,717,592 | 66,808,238 | -90,647 | +1,641,023 | -1,731,669 |

The livestock differences are the head held in named rows (section 3). The fertiliser row closes exactly but splits into a large activity term and an equal factor term, because LUTO's cropping-land area exceeds NIR's sown area while the factor is per hectare of cropping land (section 10). Irrigated-pasture energy and lifecycle emissions (0.28 Mt) and off-land energy (1.12 Mt) are reported but are not NIR sector 3.

On the 2010 map with the 2010 factors, modelled emissions are 68.25 Mt against NIR FY2010-11 of 70.56 Mt (AR5), a difference of -2.31 Mt. The 2010 factors do not state their GWP basis, so the split depends on how they are read: read as AR5, activity +3.25 Mt, factor -5.77 Mt, a joint term -0.92 Mt and energy scope +1.13 Mt; read as the second assessment report [SAR] values they appear to use, a GWP term of -17.19 Mt, a factor term of +11.05 Mt and a joint term of -0.55 Mt, with activity and scope unchanged. Source: `t3c_ghg_fy2011.csv`, full resolution.

## 7. Carbon layers: HIR and clearing

The two carbon layers give LUTO observed per-hectare rates for human-induced regeneration (HIR) and for clearing, both on LUTO's grid and independent of the land-use map.

+ **HIR.** Mean FY2020-24 issuance per CEA hectare of the 310 mapped HIR projects, grossed up to net abatement by NRM (natural resource management) region. The grossing follows the unit entitlement of the Carbon Credits (Carbon Farming Initiative) Act 2011, s 16(2): the 5 per cent risk of reversal buffer and, for 25-year projects, the 20 per cent permanence period discount are both percentages of net abatement and are subtracted from it, so issuance is 0.75 of net abatement for a 25-year project and 0.95 for a 100-year project, each project taking its own nominated permanence period from the CER register (365 of the 529 HIR projects nominate 25 years). Regions with fewer than three projects take their state's pooled rate, and regions with no project have a rate of zero; the FY2020-24 window and the fewer-than-three pooling rule are analysis choices, no published source. The layer carries the gross rate (the GHG row) and the issuance rate (carbon revenue). National gross 1.084 and issuance 0.893 t CO2e per hectare per year (issuance 0.824 of gross).
+ **Clearing.** NIR direct clearing emissions per hectare cleared by state, pooled over FY2017-22 and split by forest and sparse NVIS groups per cell; ACT takes NSW rates. Victorian and Tasmanian rates rest on about 5 kha cleared over six years each and are flagged.

**Measured alternatives.** An HIR rate scaled by site potential gives 0.82 of the CER gross rate on project cells; it is not used, and the site-potential layer is not needed for the rates used. Pooling FY2017-22 gives a national mixed clearing rate of 56.6 t CO2e per hectare against 78 from FY2021-22 alone. Charging each clearing its direct emission only leaves NIR's decay tail on previously cleared land uncharged: a geometric-lag fit to NIR FY1990 to FY2022 gives a national tail of 42.2 t CO2e per cleared hectare, 0.49 of the direct emission (86.1 t), R2 0.57. The fit is best at the lower bound of its decay rate, so it cannot identify the half-life, and the tail's timing is unknown.

## 8. Economics

The economics layers express revenue, costs and water per hectare of cropping land and per head, calibrated so that base-year production on the 2021-base map matches ABS.

+ **Crop yields.** Yield per hectare of cropping land is calibrated so that production on the map equals the ABS window mean by crop and state, with each state's calibration factor held within one-third to three times the national factor; rows at the band are named (section 10). On cells without the crop on the base map, the SA2 yield index is capped at 1, so a crop placed there yields at most its state calibration factor (times the irrigated ratio on irrigated land); the calibration and every allocated cell are unchanged. Irrigation water per hectare is calibrated the same way against ABS 4618.0.
+ **Prices and costs.** Farm-gate prices from ABS value of production; costs from the ABARES Farm Data Portal for broadacre groups and, for horticulture, cotton, rice, sugar and other non-cereal crops, the 2010 files' cost-to-revenue ratio applied to the new revenue (interim, section 9). 'Other non-cereal crops' takes the ABS value of 'all other crops n.e.c.' at the state grain-group farm-gate price.
+ **Livestock.** Per-head outputs (meat, live export, wool, milk) by state from ABS, AWI and Dairy Australia; national prices; per-head costs from the Farm Data Portal scaled by the share of farm livestock receipts the modelled products cover (beef and sheep 0.843, dairy 0.860 nationally; NT takes the national share, cause unverified).
+ **Wool trend.** The productivity input's wool per head is held flat over the projection (it fell by 0.77 per cent a year dryland and 1.53 per cent irrigated from 2010 in the 2010-build file, source not recorded), because AWI's greasy cut per head is flat over two decades: 4.53 kg in 2003-04 and 4.44 kg in 2023-24, a trend of -0.04 per cent a year. This is a change to the default behaviour; the other productivity trends are carried unchanged.
+ **Water licence price.** The price charged on the change in irrigation water requirement in a transition is the FY2020-21 entitlement price in each state's main class: NSW general security, Victorian high reliability, SA high security, Queensland supplemented medium priority, and Tasmanian Irrigation's primary-sale prices (converted to FY2021 dollars by the CPI). It is a single base year, unlike the FY2017-25 window of the other economics. Each river region and state pair takes the price of the trading zone or scheme that supplies it, from a hand-entered table with the page of each price; the NSW coastal Hunter and Bega-Brogo prices are computed from the NSW 71Q entitlement trade table (FY2020-21 volume-weighted price per share; two trades and one trade). Where no public price exists, northern Basin river regions take 2,869 $/ML (the mean of Gwydir and Macquarie general security) and all other regions, including southern Basin regions without a public price, 1,442 $/ML (the median of the twelve public prices outside the Basin in FY2021 dollars); both are inferred. Public prices cover 72 per cent of irrigated land on the 2021-base map (the Basin 93 per cent, a further 6 per cent at the northern proxy). Queensland Border Rivers rests on a single trade (Macintyre Brook, 92 ML). The 2010-build layer, source and price year not recorded, was $0 on 29 per cent of irrigated land, including all of Tasmania, WA and NT. Because the licence term is symmetric, retiring irrigation is credited at the same price, including where no entitlement market exists.
+ **Climate multipliers.** The 2010 multipliers are filled on cells the 2021-base map allocates or makes eligible, from the area-weighted mean of the same column in the SA2, else state, else nationally; they are not rebuilt.

**Measured alternatives.** Exact calibration without a band gives yields of 108 t per hectare for Victorian summer cereals and 434 t for Tasmanian vegetables where NLUM maps little area against large ABS production; the band leaves those as named rows instead. Pricing 'other non-cereal crops' at the 2010 files' $12,310 per t matches ABS value with 9.3 kt; the grain-group price matches the same value with about 0.42 Mt. Charging full Farm Data Portal costs per head gave dairy a margin of -$96.6 per head; scaling by the receipts share gives $186.2.

**Production, value and water against ABS.** Base-year crop production on the 2021-base map is 0.991 of the ABS window mean and value 0.985; the crops below 0.99 or above 1.01 are those with named rows (section 10). Source: `t4a_crops_by_crop_state.csv`.

| Crop, Australia | Cropping land (kha) | Production / ABS | Value / ABS |
|---|---:|---:|---:|
| Winter cereals | 29,725.8 | 1.000 | 1.000 |
| Winter oilseeds | 3,838.5 | 1.000 | 1.000 |
| Winter legumes | 3,078.8 | 1.000 | 1.000 |
| Hay | 2,482.3 | 1.000 | 1.000 |
| Summer cereals | 891.0 | 0.964 | 0.957 |
| Summer legumes | 530.7 | 0.694 | 0.885 |
| Sugar | 493.6 | 1.000 | 1.000 |
| Cotton | 436.8 | 0.999 | 1.000 |
| Summer oilseeds | 261.1 | 1.089 | 1.092 |
| Other non-cereal crops | 243.3 | 0.823 | 0.826 |
| Grapes | 219.1 | 1.000 | 1.000 |
| Vegetables | 160.0 | 0.889 | 0.918 |
| Nuts | 131.2 | 1.000 | 1.000 |
| Stone fruit | 88.8 | 1.000 | 1.000 |
| Rice | 64.3 | 1.000 | 1.000 |
| Citrus | 59.6 | 1.000 | 1.000 |
| Tropical stone fruit | 46.3 | 0.871 | 0.866 |
| Plantation fruit | 30.1 | 1.000 | 1.000 |
| Apples | 23.7 | 0.909 | 0.914 |
| Pears | 2.4 | 1.048 | 1.050 |

Irrigation water on the 2021-base map is 7,691 GL against 7,764 GL in ABS 4618.0 2020-21 (0.991), with the Murray-Darling Basin at 1.000. Source: `t5_water_drainage_division.csv`.

| GL, 2020-21 | LUTO crops | LUTO pasture | LUTO total | ABS | LUTO / ABS |
|---|---:|---:|---:|---:|---:|
| Murray-Darling Basin | 4,148.6 | 697.5 | 4,846.1 | 4,846.8 | 1.000 |
| North East Coast | 1,400.5 | 15.4 | 1,415.9 | 1,409.6 | 1.004 |
| South East Coast (Victoria) | 215.4 | 333.1 | 548.6 | 574.3 | 0.955 |
| Tasmania | 97.6 | 165.4 | 263.0 | 337.9 | 0.778 |
| South West Coast | 142.4 | 63.9 | 206.3 | 182.4 | 1.131 |
| South East Coast (NSW) | 73.7 | 125.4 | 199.1 | 117.2 | 1.700 |
| South Australian Gulf | 84.7 | 5.7 | 90.4 | 77.6 | 1.164 |
| Tanami-Timor Sea Coast | 62.9 | 0.6 | 63.5 | 91.5 | 0.695 |
| Carpentaria Coast | 42.9 | 0.7 | 43.5 | 39.4 | 1.104 |
| Pilbara-Gascoyne | 11.6 | 0.0 | 11.6 | 58.9 | 0.197 |
| Other divisions | 2.3 | 0.3 | 2.6 | 28.0 | 0.093 |
| Australia | 6,282.7 | 1,407.9 | 7,690.6 | 7,763.5 | 0.991 |

Water per hectare is calibrated by state, so divisions within a state can sit above or below ABS; the largest gaps are in divisions with little irrigation.

## 9. Inherited inputs, interim items and what replaces them

Several layers rest partly on 2010-build inputs or carry values that are interim until a document arrives; none of these is described as final.

| Item | Where | Basis now | Replaced by |
|---|---|---|---|
| 2010 `lumap.h5`, `lmmap.h5` | map, HIR | study area footprint, held cells, file layout; HIR project cells | the map is the footprint by design; HIR at the base-year switch |
| `real_area.h5`, `state_id.npy`, `NLUM_2010-11_mask.tif` | all | cell grid and areas | kept (grid definition) |
| `safe_pur_natl.h5`, `safe_pur_modl.h5` | livestock | safe utilisation | no replacement planned |
| `agGHG_*` 2010 files | GHG factors | layout and unchanged non-scope fields | rebuilt fields only |
| `demand_projections.h5` | off-land intensities | 2010 demand basis | demand re-anchoring at the switch |
| `agec_crops.h5` 2010 file | crop economics | cost-to-revenue ratio (13 crops), FDP cost split, irrigated/dry yield ratio | cost budgets when supplied |
| `water_delivery_price.h5` 2010 file | crop and livestock water price | interim, state median fill | ACCC water monitoring reports, before the gate run |
| `water_licence_price.h5` fallbacks | transitions (licence cost and credit) | inferred prices where no public price: 2,869 $/ML northern Basin, 1,442 $/ML elsewhere (28 per cent of irrigated land) | NSW unregulated and groundwater 71Q trades (the regulated-river table is used); none planned before the gate |
| `bio_GBF3_NVIS_MVG.nc` | clearing | NVIS groups, origin not recorded | a pinned NVIS release |
| `climate_change_impacts_*.h5` | multipliers | filled, not rebuilt | re-anchoring at the switch |
| crop energy and lifecycle fields | `agGHG_crops.h5` | 2010 intensities on rebuilt activity | a public series by crop |
| live-export weights | livestock | 0.337 t (cattle), 0.0484 t (sheep), 2010 basis | no public series |
| inferred values | crop economics | field peas and faba unit value; summer legumes residual; other non-cereal crops price; dry pears from dry apples | ABS or ABARES series if published |
| `Water_Use_Domestic.csv` 2010 file | domestic and industrial water use | regional pattern (ABS 2010-11 by region, apportionment not recorded) | a public regional series |

**Domestic and industrial water use.** The water limits subtract each region's domestic and industrial use from its yield. The 2010-build file is the ABS Water Account 2010-11 (households and ANZSIC divisions B to S, so excluding agriculture, forestry and fishing), apportioned to drainage divisions and river regions by a method that is not recorded; it totals 5,889,627 ML against the account's 5,986,324 ML, and the 96.7 GL gap is unattributed. The 2021-base file keeps that regional pattern and scales each state's part to 2020-21 (New South Wales and the ACT as one group, as NIR reports them and as their river regions are shared): total consumption less agriculture, forestry and fishing, plus the water industry's effluent returns, which the 2020-21 account credits to that industry (5,688,073 ML nationally). Cells the ABS suppresses are recovered from published totals where they can be; agriculture in Western Australia, the Northern Territory and the ACT is estimated, splitting their joint figure by self-extracted agricultural use. River regions are split across states by cell area, and drainage divisions are the sums of their river regions, as in the 2010-build file.

## 10. Named rows

Where a statistic cannot be met on the map, the difference is held in a named row with its size rather than spread over other rows; the rows below are the ones reviewers should read alongside the layers.

### 10.1 Sown share

LUTO crop area is cropping land as NLUM maps it, fallow and rotation phases included, not area sown in the year. Nationally about 0.64 of cropping land is sown in a year (winter cereals, oilseeds and legumes alike, against ABS), which is the fallow and rotation share this basis expects. Per-hectare factors and economics are therefore per hectare of cropping land and lower than per sown hectare by about that share, base-year production still matches ABS by calibration, and crop area on the map is larger than ABS sown area. Fourteen (state, crop group) rows fell outside a sown share of 0.4 to 1.05 without a verified mechanism; on the final map 12 remain and stand as 'NLUM against ABS/NIR, cause unverified'. Source: `t4d_sown_share_group_state.csv`.

| State | Crop group | Statistic | Cropping land (ha) | Sown area (ha) | Sown share | Final map |
|---|---|---|---:|---:|---:|---|
| Tas | Legumes (winter and summer) | ABS 2020-21 | 8,405 | 347 | 0.04 | cause unverified |
| NT | Hay | ABS 2020-21 | 28,309 | 8,341 | 0.29 | cause unverified |
| Tas | Vegetables | ABS 2020-21 | 1,383 | 14,536 | 10.51 | cause unverified |
| Vic | Grapes | ABS 2020-21 | 76,163 | 29,735 | 0.39 | cause unverified |
| NT | Broadacre | ABS 2020-21 | 15,066 | 5,934 | 0.39 | cause unverified |
| Tas | All crops | ABS 2020-21 | 97,537 | 135,464 | 1.39 | cause unverified |
| NT | All crops | ABS 2020-21 | 53,192 | 20,570 | 0.39 | cause unverified |
| Vic | Irrigated crops | NIR FY2021 | 201,047 | 12,585 | 0.06 | cause unverified |
| Tas | Irrigated crops | NIR FY2021 | 41,660 | 7,531 | 0.18 | cause unverified |
| Tas | Horticulture | NIR FY2021 | 7,046 | 20,277 | 2.88 | cause unverified |
| Vic | Irrigated crops, excluding hay | NIR FY2021 | 93,736 | 12,585 | 0.13 | cause unverified |
| Tas | Irrigated crops, excluding hay | NIR FY2021 | 28,717 | 7,531 | 0.26 | cause unverified |
| Tas | Winter oilseeds | ABS 2020-21 | 4,991 | 1,324 | 0.27 | under 5 kha on both sides |
| WA | Horticulture | NIR FY2021 | 50,137 | 20,763 | 0.41 | in range |

### 10.2 Tonnes not placed

Where NLUM maps less area than ABS records and the yield factor reaches the top of its band, the remaining production is not placed on the map; the rows below carry it with its value at the window farm-gate price, under the label 'NLUM maps less area than ABS records (NLUM against ABS, cause unverified); tonnes not placed'. The crops named for this are Tasmanian vegetables, summer legumes, tropical stone fruit, apples, pears, other non-cereal crops and summer oilseeds. Source: `t4b_crops_named_rows.csv`.

| Crop | State | ABS window mean (t) | On the map (t) | Not placed (t) | Value not placed ($m) |
|---|---|---:|---:|---:|---:|
| Vegetables | Tas | 634,032 | 124,881 | 509,151 | 321.24 |
| Summer legumes | WA | 252,896 | 22,883 | 230,013 | 92.04 |
| Other non-cereal crops | Tas | 123,156 | 48,710 | 74,447 | 24.11 |
| Summer legumes | SA | 137,524 | 102,455 | 35,069 | 20.55 |
| Apples | NSW | 39,314 | 16,138 | 23,176 | 41.13 |
| Tropical stone fruit | WA | 28,300 | 7,317 | 20,982 | 96.12 |
| Summer legumes | Vic | 153,756 | 133,698 | 20,058 | 9.47 |
| Apples | Tas | 31,605 | 27,930 | 3,675 | 6.49 |
| Pears (no NLUM area) | NSW, QLD, Tas, ACT | 2,272 | 0 | 2,272 | 3.03 |
| Other non-cereal crops, apples (no NLUM area) | ACT | 270 | 0 | 270 | 0.08 |

Across the named crops, net of the over-placed rows of those crops, 816,384 t worth $518.4m are not placed. Four further rows sit at the band with the same label, in crops whose national loss is under 5 per cent: summer cereals in Victoria (62,250 t not placed), Western Australia (12,825 t) and Tasmania (1,986 t), and cotton in Western Australia (2,670 t).

### 10.3 Over-placed rows

Twelve rows place more than the ABS mean, because NLUM maps more area than ABS records and the yield factor is at the bottom of its band, or because ABS publishes no production for the state and the national yield factor applies; 106,214 t worth $102.5m at farm-gate prices sit above ABS. Source: `t4b_crops_named_rows.csv`.

| Crop | State | ABS window mean (t) | On the map (t) | Above ABS (t) | Label |
|---|---|---:|---:|---:|---|
| Summer legumes | NSW | 30,515 | 79,150 | 48,636 | NLUM maps more area than ABS records at the lower yield cap |
| Summer legumes | QLD | 63,310 | 97,657 | 34,347 | as above |
| Pears | WA | 4,386 | 12,090 | 7,704 | as above |
| Summer legumes | Tas | 1,006 | 6,600 | 5,594 | as above |
| Summer oilseeds | SA | 31 | 3,372 | 3,341 | as above |
| Cotton | NT | 2,032 | 4,072 | 2,040 | as above |
| Winter oilseeds | QLD | 3,609 | 4,250 | 641 | as above |
| Summer legumes | NT | 1 | 813 | 812 | as above |
| Tropical stone fruit | Tas | 43 | 88 | 45 | as above |
| Summer oilseeds | Tas | 0 | 1,860 | 1,860 | NLUM maps more area than ABS records; no ABS production published (zero); national yield factor |
| Winter oilseeds | NT | 0 | 805 | 805 | as for Tasmanian summer oilseeds (zero); the FY2022 value rests on a missing row under the ABS convention |
| Summer oilseeds | NT | 0 | 390 | 390 | NLUM maps more area than ABS records; no ABS production published (zero or suppressed, unverified); national yield factor |

### 10.4 Other named rows

+ **Tasmanian 'Other oilseeds'.** ABS 7121.0 publishes Tasmanian 'Other oilseeds' production of 349 to 3,211 t a year over 2016-21 (mean 1,763 t over FY2017-21) without itemising the crops; no LUTO crop group takes it, since summer oilseeds is ABARES sunflower and soybeans (zero in Tasmania) and winter oilseeds is canola.
+ **Livestock.** Head not on a cell is held in three rows (section 3): the scaled ABS SA2 rounding and suppression (beef 5,470); 'no cell to carry it: SA2 has no grazing cell in scope' (beef 23,274, dairy 6,052, sheep 38,665); and 'no cell to carry it: no cell of this type in SA2' (beef 4,058, sheep 62,458). Geraldton's 5,150 sheep sit in the 'no grazing cell in scope' row, its cereal cells having been retyped with the other held 2010 crop cells.

## 11. Caveats and limitations

+ **Cropping land, not sown area.** Crop area and per-hectare values follow NLUM's cropping land; base-year production matches ABS by calibration, but area-based comparisons with ABS sown area are not like for like, and demand must be re-anchored to base-year production at the switch.
+ **Inherited 2010 inputs.** Section 9 lists the 2010-build files the layers still read; values derived from them (notably the cost ratios for 13 crops and the water prices) carry the 2010 files' basis until replaced.
+ **Clearing decay tail.** Direct clearing emissions only are charged, which understates clearing emissions by NIR's decay tail on previously cleared land: 0.49 of the direct emission nationally (42.2 against 86.1 t CO2e per cleared hectare; geometric-lag fit to NIR FY1990 to FY2022, R2 0.57, timing not identified).
+ **Livestock factors by state.** Per-head factors are state averages; within-state variation in the 2010 files had no public basis and is not kept.
+ **Memory.** The crop economics step peaks at 130.3 GiB of resident memory and the GHG factors at 73.2 GiB (both measured in the full input build of 19 October 2026: resident set size of the process and its children, sampled every 0.5 s with psutil on Windows). The scripts run one after another, so the largest step sets the requirement: a machine with at least 256 GB of RAM; the class allocation reaches about 30 GB across 48 worker processes (less with fewer workers), the comparison tool about 15 GB, the climate-multiplier fill 8.7 GiB (measured in the build of 22 October 2026, which writes the NetCDF copies) and every other tool about 6 GB or less (measured October 2026; each tool's header states its own figure).
+ **Live sources.** The CER register and the National Greenhouse Accounts API can change without notice; a rebuild on a machine without the cache stops when they do, by design, until the change is reviewed. DAFF's live export statistics and the AussieGRASS grids are pinned on the values used rather than the file bytes, so DAFF's monthly additions and LongPaddock's rebuilds of recent issues pass, while any change to a value used stops the rebuild.
