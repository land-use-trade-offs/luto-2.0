# DataFlow.md

## Table of Contents

- [DataFlow.md](#dataflowmd)
  - [Table of Contents](#table-of-contents)
  - [luto.economics.non\_agricultural.ghg.get\_ghg\_env\_plantings](#lutoeconomicsnon_agriculturalghgget_ghg_env_plantings)
    - [Data Flow Summary](#data-flow-summary)
    - [Key Transformations](#key-transformations)
    - [Data Processing Pipeline](#data-processing-pipeline)
  - [luto.economics.non\_agricultural.ghg.get\_ghg\_rip\_plantings](#lutoeconomicsnon_agriculturalghgget_ghg_rip_plantings)
    - [Data Flow Summary](#data-flow-summary-1)
    - [Key Transformations](#key-transformations-1)
    - [Data Processing Pipeline](#data-processing-pipeline-1)
  - [luto.economics.non\_agricultural.ghg.get\_ghg\_agroforestry\_base](#lutoeconomicsnon_agriculturalghgget_ghg_agroforestry_base)
    - [Data Flow Summary](#data-flow-summary-2)
    - [Key Transformations](#key-transformations-2)
    - [Data Processing Pipeline](#data-processing-pipeline-2)
  - [luto.economics.non\_agricultural.ghg.get\_ghg\_sheep\_agroforestry](#lutoeconomicsnon_agriculturalghgget_ghg_sheep_agroforestry)
    - [Data Flow Summary](#data-flow-summary-3)
    - [Key Transformations](#key-transformations-3)
    - [Data Processing Pipeline](#data-processing-pipeline-3)
  - [luto.economics.non\_agricultural.ghg.get\_ghg\_beef\_agroforestry](#lutoeconomicsnon_agriculturalghgget_ghg_beef_agroforestry)
    - [Data Flow Summary](#data-flow-summary-4)
    - [Key Transformations](#key-transformations-4)
    - [Data Processing Pipeline](#data-processing-pipeline-4)
  - [luto.economics.non\_agricultural.ghg.get\_ghg\_carbon\_plantings\_block](#lutoeconomicsnon_agriculturalghgget_ghg_carbon_plantings_block)
    - [Data Flow Summary](#data-flow-summary-5)
    - [Key Transformations](#key-transformations-5)
    - [Data Processing Pipeline](#data-processing-pipeline-5)
  - [luto.economics.non\_agricultural.ghg.get\_ghg\_carbon\_plantings\_belt\_base](#lutoeconomicsnon_agriculturalghgget_ghg_carbon_plantings_belt_base)
    - [Data Flow Summary](#data-flow-summary-6)
    - [Key Transformations](#key-transformations-6)
    - [Data Processing Pipeline](#data-processing-pipeline-6)
  - [luto.economics.non\_agricultural.ghg.get\_ghg\_sheep\_carbon\_plantings\_belt](#lutoeconomicsnon_agriculturalghgget_ghg_sheep_carbon_plantings_belt)
    - [Data Flow Summary](#data-flow-summary-7)
    - [Key Transformations](#key-transformations-7)
    - [Data Processing Pipeline](#data-processing-pipeline-7)
  - [luto.economics.non\_agricultural.ghg.get\_ghg\_beef\_carbon\_plantings\_belt](#lutoeconomicsnon_agriculturalghgget_ghg_beef_carbon_plantings_belt)
    - [Data Flow Summary](#data-flow-summary-8)
    - [Key Transformations](#key-transformations-8)
    - [Data Processing Pipeline](#data-processing-pipeline-8)
  - [luto.economics.non\_agricultural.ghg.get\_ghg\_beccs](#lutoeconomicsnon_agriculturalghgget_ghg_beccs)
    - [Data Flow Summary](#data-flow-summary-9)
    - [Key Transformations](#key-transformations-9)
    - [Data Processing Pipeline](#data-processing-pipeline-9)
  - [luto.economics.non\_agricultural.ghg.get\_ghg\_destocked\_land](#lutoeconomicsnon_agriculturalghgget_ghg_destocked_land)
    - [Data Flow Summary](#data-flow-summary-10)
    - [Key Transformations](#key-transformations-10)
    - [Data Processing Pipeline](#data-processing-pipeline-10)
  - [luto.economics.non\_agricultural.ghg.get\_ghg\_matrix](#lutoeconomicsnon_agriculturalghgget_ghg_matrix)
    - [Data Flow Summary](#data-flow-summary-11)
    - [Key Transformations](#key-transformations-11)
    - [Data Processing Pipeline](#data-processing-pipeline-11)
    - [Function Summary Matrix](#function-summary-matrix)

---

## luto.economics.non_agricultural.ghg.get_ghg_env_plantings

### Data Flow Summary

```
Raw FullCAM Output → 4_assemble_biophysical_data.py → cell_biophysical_df.h5 →
dataprep.py → ep_block_avg_t_co2_ha_yr.h5 →
data.py (with fire/reversal risk adjustments) → EP_BLOCK_AVG_T_CO2_HA →
get_ghg_env_plantings() → GHG emissions per cell
```

### Key Transformations
1. **Annual Averaging**: Raw cumulative carbon (90 years) ÷ settings.CARBON_EFFECTS_WINDOW = annual rate (done in data.py)
2. **Component Aggregation**: Trees + Debris = Above-ground carbon
3. **Risk Discounting**: Above-ground × fire_risk × (1 - reversal_risk)
4. **Spatial Scaling**: Per-hectare rate × cell area = total per cell
5. **Sign Convention**: Negative values for carbon sequestration

### Data Processing Pipeline

| Step | Code Location | Process |
|------|---------------|---------|
| 1. Raw FullCAM Data Processing | **Code**: N:\Data-Master\LUTO_2.0_input_data\Scripts\4_assemble_biophysical_data.py<br>**Section**: ############## Average annual carbon sequestration by reforestation land uses | • `EP_BLOCK_TREES_AVG_T_CO2_HA_YR` = h5f['Trees_tCO2_ha'][90] (Above Ground Biomass cumulative from 2010-2100, no averaging)<br>• `EP_BLOCK_DEBRIS_AVG_T_CO2_HA_YR` = h5f['Debris_tCO2_ha'][90] (Debris carbon cumulative, no averaging)<br>• `EP_BLOCK_SOIL_AVG_T_CO2_HA_YR` = (h5f['Soil_tCO2_ha'][90] - h5f['Soil_tCO2_ha'][0]) (Marginal soil carbon change, no averaging) |
| 2. LUTO Data Preprocessing | **Code**: luto/dataprep.py<br>**Lines**: 807-811 - Average annual carbon sequestration by Environmental Plantings (block plantings) | • Load biophysical data: `bioph = pd.read_hdf(raw_data + 'cell_biophysical_df.h5')` (line 250)<br>• Create AG/BG DataFrame: `s = pd.DataFrame(columns=['EP_BLOCK_AG_AVG_T_CO2_HA_YR', 'EP_BLOCK_BG_AVG_T_CO2_HA_YR'])` (line 808)<br>• Combine above-ground: `s['EP_BLOCK_AG_AVG_T_CO2_HA_YR'] = bioph.eval('EP_BLOCK_TREES_AVG_T_CO2_HA_YR + EP_BLOCK_DEBRIS_AVG_T_CO2_HA_YR')` (line 809)<br>• Extract below-ground: `s['EP_BLOCK_BG_AVG_T_CO2_HA_YR'] = bioph['EP_BLOCK_SOIL_AVG_T_CO2_HA_YR']` (line 810)<br>• Save to HDF5: `s.to_hdf(outpath + 'ep_block_avg_t_co2_ha_yr.h5', ...)` (line 811) |
| 3. LUTO Runtime Data Loading | **Code**: luto/data.py<br>**Lines**: 750-754 - Load environmental plantings (block) GHG sequestration | • Load fire risk data: `fr_df = pd.read_hdf(..., "fire_risk.h5", where=self.MASK)` (line 745)<br>• Select fire risk level: `fire_risk = fr_df[fr_dict[settings.FIRE_RISK]]` (line 747)<br>• Load EP data with spatial mask: `ep_df = pd.read_hdf(..., "ep_block_avg_t_co2_ha_yr.h5", where=self.MASK)` (line 750)<br>• Apply risk adjustments and annual averaging: `EP_BLOCK_AVG_T_CO2_HA = (ep_df.EP_BLOCK_AG_AVG_T_CO2_HA_YR * (fire_risk / 100) * (1 - settings.RISK_OF_REVERSAL) + ep_df.EP_BLOCK_BG_AVG_T_CO2_HA_YR).to_numpy(dtype=np.float32) / settings.CARBON_EFFECTS_WINDOW` (lines 751-754)<br>• **Risk Logic**: Above-ground carbon discounted by fire risk & reversal risk; below-ground carbon stable<br>• **Annual Averaging**: Cumulative carbon divided by CARBON_EFFECTS_WINDOW (default 91 years) |
| 4. GHG Calculation Function Usage | **Code**: luto/economics/non_agricultural/ghg.py<br>**Lines**: 29-48 - get_ghg_env_plantings function | • **Function Purpose**: Calculate GHG emissions (negative = sequestration) for each spatial cell<br>• **Input Parameters**: `data` (Data object), `aggregate` (Boolean output format flag)<br>• **Calculation**: `return -data.EP_BLOCK_AVG_T_CO2_HA * data.REAL_AREA` (line 46 if aggregate=True, line 48 if aggregate=False)<br>• **Units**: Tonnes CO2e per cell<br>• **Scaling**: Multiplied by `data.REAL_AREA` to convert from per-hectare to per-cell basis<br>• **Sign Convention**: Negative values indicate CO2 removal from atmosphere (climate beneficial) |


## luto.economics.non_agricultural.ghg.get_ghg_rip_plantings

### Data Flow Summary

```
Raw FullCAM Output → 4_assemble_biophysical_data.py → cell_biophysical_df.h5 →
dataprep.py → ep_rip_avg_t_co2_ha_yr.h5 →
data.py (with fire/reversal risk adjustments) → EP_RIP_AVG_T_CO2_HA →
get_ghg_rip_plantings() → GHG emissions per cell
```

### Key Transformations
1. **Annual Averaging**: Raw cumulative carbon (90 years) ÷ settings.CARBON_EFFECTS_WINDOW = annual rate (done in data.py)
2. **Component Aggregation**: Trees + Debris = Above-ground carbon
3. **Risk Discounting**: Above-ground × fire_risk × (1 - reversal_risk)
4. **Spatial Scaling**: Per-hectare rate × cell area = total per cell
5. **Sign Convention**: Negative values for carbon sequestration

### Data Processing Pipeline

| Step | Code Location | Process |
|------|---------------|---------|
| 1. Raw FullCAM Data Processing | **Code**: N:\Data-Master\LUTO_2.0_input_data\Scripts\4_assemble_biophysical_data.py<br>**Lines**: 131-134 - Riparian plantings carbon sequestration | • `EP_RIP_TREES_AVG_T_CO2_HA_YR` = h5f['Trees_tCO2_ha'][90] (Above Ground Biomass cumulative from 2010-2100, no averaging)<br>• `EP_RIP_DEBRIS_AVG_T_CO2_HA_YR` = h5f['Debris_tCO2_ha'][90] (Debris carbon cumulative, no averaging)<br>• `EP_RIP_SOIL_AVG_T_CO2_HA_YR` = (h5f['Soil_tCO2_ha'][90] - h5f['Soil_tCO2_ha'][0]) (Marginal soil carbon change, no averaging) |
| 2. LUTO Data Preprocessing | **Code**: luto/dataprep.py<br>**Lines**: 813-817 - Average annual carbon sequestration by Riparian Plantings | • Load biophysical data: `bioph = pd.read_hdf(raw_data + 'cell_biophysical_df.h5')` (line 250)<br>• Create AG/BG DataFrame: `s = pd.DataFrame(columns=['EP_RIP_AG_AVG_T_CO2_HA_YR', 'EP_RIP_BG_AVG_T_CO2_HA_YR'])` (line 814)<br>• Combine above-ground: `s['EP_RIP_AG_AVG_T_CO2_HA_YR'] = bioph.eval('EP_RIP_TREES_AVG_T_CO2_HA_YR + EP_RIP_DEBRIS_AVG_T_CO2_HA_YR')` (line 815)<br>• Extract below-ground: `s['EP_RIP_BG_AVG_T_CO2_HA_YR'] = bioph['EP_RIP_SOIL_AVG_T_CO2_HA_YR']` (line 816)<br>• Save to HDF5: `s.to_hdf(outpath + 'ep_rip_avg_t_co2_ha_yr.h5', ...)` (line 817) |
| 3. LUTO Runtime Data Loading | **Code**: luto/data.py<br>**Lines**: 764-769 - Load riparian plantings GHG sequestration | • Load fire risk data: `fr_df = pd.read_hdf(..., "fire_risk.h5", where=self.MASK)` (line 745)<br>• Select fire risk level: `fire_risk = fr_df[fr_dict[settings.FIRE_RISK]]` (line 747)<br>• Load EP data with spatial mask: `ep_df = pd.read_hdf(..., "ep_rip_avg_t_co2_ha_yr.h5", where=self.MASK)` (line 765)<br>• Apply risk adjustments and annual averaging: `EP_RIP_AVG_T_CO2_HA = ((ep_df.EP_RIP_AG_AVG_T_CO2_HA_YR * (fire_risk / 100) * (1 - settings.RISK_OF_REVERSAL)) + ep_df.EP_RIP_BG_AVG_T_CO2_HA_YR).to_numpy(dtype=np.float32) / settings.CARBON_EFFECTS_WINDOW` (lines 766-769)<br>• **Risk Logic**: Above-ground carbon discounted by fire risk & reversal risk; below-ground carbon stable<br>• **Annual Averaging**: Cumulative carbon divided by CARBON_EFFECTS_WINDOW (default 91 years) |
| 4. GHG Calculation Function Usage | **Code**: luto/economics/non_agricultural/ghg.py<br>**Lines**: 54-75 - get_ghg_rip_plantings function | • **Function Purpose**: Calculate GHG emissions (negative = sequestration) for riparian plantings per cell<br>• **Input Parameters**: `data` (Data object), `aggregate` (Boolean output format flag)<br>• **Calculation**: `return -data.EP_RIP_AVG_T_CO2_HA * data.REAL_AREA` (line 71 if aggregate=True, line 73 if aggregate=False)<br>• **Units**: Tonnes CO2e per cell<br>• **Scaling**: Multiplied by `data.REAL_AREA` to convert from per-hectare to per-cell basis<br>• **Sign Convention**: Negative values indicate CO2 removal from atmosphere (climate beneficial) |


## luto.economics.non_agricultural.ghg.get_ghg_agroforestry_base

### Data Flow Summary

```
Raw FullCAM Output → 4_assemble_biophysical_data.py → cell_biophysical_df.h5 →
dataprep.py → ep_belt_avg_t_co2_ha_yr.h5 →
data.py (with fire/reversal risk adjustments) → EP_BELT_AVG_T_CO2_HA →
get_ghg_agroforestry_base() → Base agroforestry GHG per cell
```

### Key Transformations
1. **Belt Plantation Type**: Linear/boundary tree plantings integrated with agriculture
2. **Annual Averaging**: Raw cumulative carbon (90 years) ÷ 91 = annual rate
3. **Component Aggregation**: Trees + Debris = Above-ground carbon
4. **Risk Discounting**: Above-ground × fire_risk × (1 - reversal_risk)
5. **Spatial Scaling**: Per-hectare rate × cell area = total per cell
6. **Sign Convention**: Negative values for carbon sequestration

### Data Processing Pipeline

| Step | Code Location | Process |
|------|---------------|---------|
| 1. Raw FullCAM Data Processing | **Code**: N:\Data-Master\LUTO_2.0_input_data\Scripts\4_assemble_biophysical_data.py<br>**Lines**: 136-139 - Environmental plantings (belt) carbon sequestration | • `EP_BELT_TREES_AVG_T_CO2_HA_YR` = h5f['Trees_tCO2_ha'][90] (Above Ground Biomass cumulative from 2010-2100, no averaging)<br>• `EP_BELT_DEBRIS_AVG_T_CO2_HA_YR` = h5f['Debris_tCO2_ha'][90] (Debris carbon cumulative, no averaging)<br>• `EP_BELT_SOIL_AVG_T_CO2_HA_YR` = (h5f['Soil_tCO2_ha'][90] - h5f['Soil_tCO2_ha'][0]) (Marginal soil carbon change, no averaging) |
| 2. LUTO Data Preprocessing | **Code**: luto/dataprep.py<br>**Lines**: 819-823 - Average annual carbon sequestration by Environmental Plantings (belt) | • Load biophysical data: `bioph = pd.read_hdf(raw_data + 'cell_biophysical_df.h5')` (line 250)<br>• Create AG/BG DataFrame: `s = pd.DataFrame(columns=['EP_BELT_AG_AVG_T_CO2_HA_YR', 'EP_BELT_BG_AVG_T_CO2_HA_YR'])` (line 820)<br>• Combine above-ground: `s['EP_BELT_AG_AVG_T_CO2_HA_YR'] = bioph.eval('EP_BELT_TREES_AVG_T_CO2_HA_YR + EP_BELT_DEBRIS_AVG_T_CO2_HA_YR')` (line 821)<br>• Extract below-ground: `s['EP_BELT_BG_AVG_T_CO2_HA_YR'] = bioph['EP_BELT_SOIL_AVG_T_CO2_HA_YR']` (line 822)<br>• Save to HDF5: `s.to_hdf(outpath + 'ep_belt_avg_t_co2_ha_yr.h5', ...)` (line 823) |
| 3. LUTO Runtime Data Loading | **Code**: luto/data.py<br>**Lines**: 757-762 - Load environmental plantings (belt) GHG sequestration | • Load fire risk data: `fr_df = pd.read_hdf(..., "fire_risk.h5", where=self.MASK)` (line 745)<br>• Select fire risk level: `fire_risk = fr_df[fr_dict[settings.FIRE_RISK]]` (line 747)<br>• Load EP data with spatial mask: `ep_df = pd.read_hdf(..., "ep_belt_avg_t_co2_ha_yr.h5", where=self.MASK)` (line 758)<br>• Apply risk adjustments and annual averaging: `EP_BELT_AVG_T_CO2_HA = ((ep_df.EP_BELT_AG_AVG_T_CO2_HA_YR * (fire_risk / 100) * (1 - settings.RISK_OF_REVERSAL)) + ep_df.EP_BELT_BG_AVG_T_CO2_HA_YR).to_numpy(dtype=np.float32) / settings.CARBON_EFFECTS_WINDOW` (lines 759-762)<br>• **Risk Logic**: Above-ground carbon discounted by fire risk & reversal risk; below-ground carbon stable<br>• **Annual Averaging**: Cumulative carbon divided by CARBON_EFFECTS_WINDOW (default 91 years) |
| 4. GHG Calculation Function Usage | **Code**: luto/economics/non_agricultural/ghg.py<br>**Lines**: 78-93 - get_ghg_agroforestry_base function | • **Function Purpose**: Calculate base agroforestry GHG sequestration for belt plantings per cell<br>• **Input Parameters**: `data` (Data object only - no aggregate flag)<br>• **Calculation**: `return -data.EP_BELT_AVG_T_CO2_HA * data.REAL_AREA` (line 93)<br>• **Units**: Tonnes CO2e per cell<br>• **Scaling**: Multiplied by `data.REAL_AREA` to convert from per-hectare to per-cell basis<br>• **Sign Convention**: Negative values indicate CO2 removal from atmosphere (climate beneficial)<br>• **Usage**: Base function for hybrid agroforestry systems (sheep/beef + forestry) |


## luto.economics.non_agricultural.ghg.get_ghg_sheep_agroforestry

### Data Flow Summary

```
Agricultural GHG Matrix (ag_g_mrj) + Base Agroforestry (EP_BELT) + Exclusion Matrix →
Proportional Contribution Calculation → Mixed sheep-agroforestry GHG per cell
```

### Key Transformations
1. **Hybrid Land Use**: Combines sheep grazing with agroforestry plantings
2. **Proportional Allocation**: Uses exclusion matrix to determine area splits
3. **Agricultural Component**: Sheep GHG from dryland production
4. **Forestry Component**: Belt plantation carbon sequestration
5. **Weighted Contribution**: `(1 - exclusion) × sheep_ghg + exclusion × agroforestry_ghg`

### Data Processing Pipeline

| Step | Code Location | Process |
|------|---------------|---------|
| 1. Agricultural GHG Matrix Input | **External Input**: `ag_g_mrj` from agricultural economics module | • **Structure**: `ag_g_mrj[m, r, j]` where m=water regime, r=cell, j=land use<br>• **Sheep Selection**: `ag_g_mrj[0, :, sheep_j]` - dryland sheep GHG per cell<br>• **Source**: Agricultural module calculations for sheep production GHG |
| 2. Exclusion Matrix Calculation | **Code**: luto/tools/__init__.py<br>**Lines**: 367-384 - get_exclusions_agroforestry_base function | • **Purpose**: Determine proportion of cell area for agroforestry vs agriculture<br>• **Base Proportion**: `exclude = np.ones(data.NCELLS) * settings.AF_PROPORTION` (line 379)<br>• **Existing Agroforestry**: `exclude[get_agroforestry_cells(lumap)] = settings.AF_PROPORTION` (line 382)<br>• **Configuration**: `settings.AF_PROPORTION` defines maximum agroforestry proportion per cell |
| 3. Base Agroforestry GHG | **Code**: luto/economics/non_agricultural/ghg.py<br>**Lines**: 78-93 - get_ghg_agroforestry_base function | • **Calculation**: `-data.EP_BELT_AVG_T_CO2_HA * data.REAL_AREA` (line 93)<br>• **Data Source**: EP_BELT carbon sequestration from belt plantings<br>• **Risk Adjustments**: Fire risk and reversal risk already applied in data.py |
| 4. Sheep-Agroforestry Integration | **Code**: luto/economics/non_agricultural/ghg.py<br>**Lines**: 96-130 - get_ghg_sheep_agroforestry function | • **Sheep Code**: `sheep_j = tools.get_sheep_code(data)` - get land use index for 'Sheep - modified land' (line 114)<br>• **Sheep GHG**: `sheep_cost = ag_g_mrj[0, :, sheep_j]` - dryland sheep emissions per cell (line 117)<br>• **Base Agroforestry**: `base_agroforestry_cost = get_ghg_agroforestry_base(data)` (line 118)<br>• **Agroforestry Contribution**: `agroforestry_contr = base_agroforestry_cost * agroforestry_x_r` (line 121)<br>• **Sheep Contribution**: `sheep_contr = sheep_cost * (1 - agroforestry_x_r)` (line 122)<br>• **Total GHG**: `ghg_total = agroforestry_contr + sheep_contr` (line 123)<br>• **Output Format**: Returns numpy array (aggregate=True) or DataFrame (aggregate=False) |


## luto.economics.non_agricultural.ghg.get_ghg_beef_agroforestry

### Data Flow Summary

```
Agricultural GHG Matrix (ag_g_mrj) + Base Agroforestry (EP_BELT) + Exclusion Matrix →
Proportional Contribution Calculation → Mixed beef-agroforestry GHG per cell
```

### Key Transformations
1. **Hybrid Land Use**: Combines beef grazing with agroforestry plantings
2. **Proportional Allocation**: Uses exclusion matrix to determine area splits
3. **Agricultural Component**: Beef GHG from dryland production
4. **Forestry Component**: Belt plantation carbon sequestration
5. **Weighted Contribution**: `(1 - exclusion) × beef_ghg + exclusion × agroforestry_ghg`

### Data Processing Pipeline

| Step | Code Location | Process |
|------|---------------|---------|
| 1. Agricultural GHG Matrix Input | **External Input**: `ag_g_mrj` from agricultural economics module | • **Structure**: `ag_g_mrj[m, r, j]` where m=water regime, r=cell, j=land use<br>• **Beef Selection**: `ag_g_mrj[0, :, beef_j]` - dryland beef GHG per cell<br>• **Source**: Agricultural module calculations for beef production GHG |
| 2. Exclusion Matrix Calculation | **Code**: luto/tools/__init__.py<br>**Lines**: 367-384 - get_exclusions_agroforestry_base function | • **Purpose**: Determine proportion of cell area for agroforestry vs agriculture<br>• **Base Proportion**: `exclude = np.ones(data.NCELLS) * settings.AF_PROPORTION` (line 379)<br>• **Existing Agroforestry**: `exclude[get_agroforestry_cells(lumap)] = settings.AF_PROPORTION` (line 382)<br>• **Configuration**: `settings.AF_PROPORTION` defines maximum agroforestry proportion per cell |
| 3. Base Agroforestry GHG | **Code**: luto/economics/non_agricultural/ghg.py<br>**Lines**: 78-93 - get_ghg_agroforestry_base function | • **Calculation**: `-data.EP_BELT_AVG_T_CO2_HA * data.REAL_AREA` (line 93)<br>• **Data Source**: EP_BELT carbon sequestration from belt plantings<br>• **Risk Adjustments**: Fire risk and reversal risk already applied in data.py |
| 4. Beef-Agroforestry Integration | **Code**: luto/economics/non_agricultural/ghg.py<br>**Lines**: 133-167 - get_ghg_beef_agroforestry function | • **Beef Code**: `beef_j = tools.get_beef_code(data)` - get land use index for 'Beef - modified land' (line 151)<br>• **Beef GHG**: `beef_cost = ag_g_mrj[0, :, beef_j]` - dryland beef emissions per cell (line 154)<br>• **Base Agroforestry**: `base_agroforestry_cost = get_ghg_agroforestry_base(data)` (line 155)<br>• **Agroforestry Contribution**: `agroforestry_contr = base_agroforestry_cost * agroforestry_x_r` (line 158)<br>• **Beef Contribution**: `beef_contr = beef_cost * (1 - agroforestry_x_r)` (line 159)<br>• **Total GHG**: `ghg_total = agroforestry_contr + beef_contr` (line 160)<br>• **Output Format**: Returns numpy array (aggregate=True) or DataFrame (aggregate=False) |


## luto.economics.non_agricultural.ghg.get_ghg_carbon_plantings_block

### Data Flow Summary

```
Raw FullCAM Output → 4_assemble_biophysical_data.py → cell_biophysical_df.h5 →
dataprep.py → cp_block_avg_t_co2_ha_yr.h5 →
data.py (with fire/reversal risk adjustments) → CP_BLOCK_AVG_T_CO2_HA →
get_ghg_carbon_plantings_block() → GHG emissions per cell
```

### Key Transformations
1. **Block Plantation Type**: Full area coverage for carbon sequestration purposes
2. **Annual Averaging**: Raw cumulative carbon (90 years) ÷ 91 = annual rate
3. **Component Aggregation**: Trees + Debris = Above-ground carbon
4. **Risk Discounting**: Above-ground × fire_risk × (1 - reversal_risk)
5. **Spatial Scaling**: Per-hectare rate × cell area = total per cell
6. **Sign Convention**: Negative values for carbon sequestration

### Data Processing Pipeline

| Step | Code Location | Process |
|------|---------------|---------|
| 1. Raw FullCAM Data Processing | **Code**: N:\Data-Master\LUTO_2.0_input_data\Scripts\4_assemble_biophysical_data.py<br>**Lines**: 141-144 - Carbon plantings (block) carbon sequestration | • `CP_BLOCK_TREES_AVG_T_CO2_HA_YR` = h5f['Trees_tCO2_ha'][90] (Above Ground Biomass cumulative from 2010-2100, no averaging)<br>• `CP_BLOCK_DEBRIS_AVG_T_CO2_HA_YR` = h5f['Debris_tCO2_ha'][90] (Debris carbon cumulative, no averaging)<br>• `CP_BLOCK_SOIL_AVG_T_CO2_HA_YR` = (h5f['Soil_tCO2_ha'][90] - h5f['Soil_tCO2_ha'][0]) (Marginal soil carbon change, no averaging) |
| 2. LUTO Data Preprocessing | **Code**: luto/dataprep.py<br>**Lines**: 825-829 - Average annual carbon sequestration by Carbon Plantings (block) | • Load biophysical data: `bioph = pd.read_hdf(raw_data + 'cell_biophysical_df.h5')` (line 250)<br>• Create AG/BG DataFrame: `s = pd.DataFrame(columns=['CP_BLOCK_AG_AVG_T_CO2_HA_YR', 'CP_BLOCK_BG_AVG_T_CO2_HA_YR'])` (line 826)<br>• Combine above-ground: `s['CP_BLOCK_AG_AVG_T_CO2_HA_YR'] = bioph.eval('CP_BLOCK_TREES_AVG_T_CO2_HA_YR + CP_BLOCK_DEBRIS_AVG_T_CO2_HA_YR')` (line 827)<br>• Extract below-ground: `s['CP_BLOCK_BG_AVG_T_CO2_HA_YR'] = bioph['CP_BLOCK_SOIL_AVG_T_CO2_HA_YR']` (line 828)<br>• Save to HDF5: `s.to_hdf(outpath + 'cp_block_avg_t_co2_ha_yr.h5', ...)` (line 829) |
| 3. LUTO Runtime Data Loading | **Code**: luto/data.py<br>**Lines**: 771-777 - Load carbon plantings (block) GHG sequestration | • Load fire risk data: `fr_df = pd.read_hdf(..., "fire_risk.h5", where=self.MASK)` (line 745)<br>• Select fire risk level: `fire_risk = fr_df[fr_dict[settings.FIRE_RISK]]` (line 747)<br>• Load CP data with spatial mask: `cp_df = pd.read_hdf(..., "cp_block_avg_t_co2_ha_yr.h5", where=self.MASK)` (line 772)<br>• Apply risk adjustments and annual averaging: `CP_BLOCK_AVG_T_CO2_HA = ((cp_df.CP_BLOCK_AG_AVG_T_CO2_HA_YR * (fire_risk / 100) * (1 - settings.RISK_OF_REVERSAL)) + cp_df.CP_BLOCK_BG_AVG_T_CO2_HA_YR).to_numpy(dtype=np.float32) / settings.CARBON_EFFECTS_WINDOW` (lines 773-777)<br>• **Risk Logic**: Above-ground carbon discounted by fire risk & reversal risk; below-ground carbon stable<br>• **Annual Averaging**: Cumulative carbon divided by CARBON_EFFECTS_WINDOW (default 91 years) |
| 4. GHG Calculation Function Usage | **Code**: luto/economics/non_agricultural/ghg.py<br>**Lines**: 170-192 - get_ghg_carbon_plantings_block function | • **Function Purpose**: Calculate GHG emissions (negative = sequestration) for carbon plantings (block) per cell<br>• **Input Parameters**: `data` (Data object), `aggregate` (Boolean output format flag)<br>• **Calculation**: `return -data.CP_BLOCK_AVG_T_CO2_HA * data.REAL_AREA` (line 188 if aggregate=True, line 190 if aggregate=False)<br>• **Units**: Tonnes CO2e per cell<br>• **Scaling**: Multiplied by `data.REAL_AREA` to convert from per-hectare to per-cell basis<br>• **Sign Convention**: Negative values indicate CO2 removal from atmosphere (climate beneficial) |


## luto.economics.non_agricultural.ghg.get_ghg_carbon_plantings_belt_base

### Data Flow Summary

```
Raw FullCAM Output → 4_assemble_biophysical_data.py → cell_biophysical_df.h5 →
dataprep.py → cp_belt_avg_t_co2_ha_yr.h5 →
data.py (with fire/reversal risk adjustments) → CP_BELT_AVG_T_CO2_HA →
get_ghg_carbon_plantings_belt_base() → Base carbon plantings belt GHG per cell
```

### Key Transformations
1. **Belt Plantation Type**: Linear/boundary tree plantings for carbon sequestration
2. **Annual Averaging**: Raw cumulative carbon (90 years) ÷ 91 = annual rate
3. **Component Aggregation**: Trees + Debris = Above-ground carbon
4. **Risk Discounting**: Above-ground × fire_risk × (1 - reversal_risk)
5. **Spatial Scaling**: Per-hectare rate × cell area = total per cell
6. **Sign Convention**: Negative values for carbon sequestration

### Data Processing Pipeline

| Step | Code Location | Process |
|------|---------------|---------|
| 1. Raw FullCAM Data Processing | **Code**: N:\Data-Master\LUTO_2.0_input_data\Scripts\4_assemble_biophysical_data.py<br>**Lines**: 146-149 - Carbon plantings (belt) carbon sequestration | • `CP_BELT_TREES_AVG_T_CO2_HA_YR` = h5f['Trees_tCO2_ha'][90] (Above Ground Biomass cumulative from 2010-2100, no averaging)<br>• `CP_BELT_DEBRIS_AVG_T_CO2_HA_YR` = h5f['Debris_tCO2_ha'][90] (Debris carbon cumulative, no averaging)<br>• `CP_BELT_SOIL_AVG_T_CO2_HA_YR` = (h5f['Soil_tCO2_ha'][90] - h5f['Soil_tCO2_ha'][0]) (Marginal soil carbon change, no averaging) |
| 2. LUTO Data Preprocessing | **Code**: luto/dataprep.py<br>**Lines**: 831-835 - Average annual carbon sequestration by Carbon Plantings (belt) | • Load biophysical data: `bioph = pd.read_hdf(raw_data + 'cell_biophysical_df.h5')` (line 250)<br>• Create AG/BG DataFrame: `s = pd.DataFrame(columns=['CP_BELT_AG_AVG_T_CO2_HA_YR', 'CP_BELT_BG_AVG_T_CO2_HA_YR'])` (line 832)<br>• Combine above-ground: `s['CP_BELT_AG_AVG_T_CO2_HA_YR'] = bioph.eval('CP_BELT_TREES_AVG_T_CO2_HA_YR + CP_BELT_DEBRIS_AVG_T_CO2_HA_YR')` (line 833)<br>• Extract below-ground: `s['CP_BELT_BG_AVG_T_CO2_HA_YR'] = bioph['CP_BELT_SOIL_AVG_T_CO2_HA_YR']` (line 834)<br>• Save to HDF5: `s.to_hdf(outpath + 'cp_belt_avg_t_co2_ha_yr.h5', ...)` (line 835) |
| 3. LUTO Runtime Data Loading | **Code**: luto/data.py<br>**Lines**: 779-785 - Load carbon plantings (belt) GHG sequestration | • Load fire risk data: `fr_df = pd.read_hdf(..., "fire_risk.h5", where=self.MASK)` (line 745)<br>• Select fire risk level: `fire_risk = fr_df[fr_dict[settings.FIRE_RISK]]` (line 747)<br>• Load CP data with spatial mask: `cp_df = pd.read_hdf(..., "cp_belt_avg_t_co2_ha_yr.h5", where=self.MASK)` (line 780)<br>• Apply risk adjustments and annual averaging: `CP_BELT_AVG_T_CO2_HA = ((cp_df.CP_BELT_AG_AVG_T_CO2_HA_YR * (fire_risk / 100) * (1 - settings.RISK_OF_REVERSAL)) + cp_df.CP_BELT_BG_AVG_T_CO2_HA_YR).to_numpy(dtype=np.float32) / settings.CARBON_EFFECTS_WINDOW` (lines 781-785)<br>• **Risk Logic**: Above-ground carbon discounted by fire risk & reversal risk; below-ground carbon stable<br>• **Annual Averaging**: Cumulative carbon divided by CARBON_EFFECTS_WINDOW (default 91 years) |
| 4. GHG Calculation Function Usage | **Code**: luto/economics/non_agricultural/ghg.py<br>**Lines**: 195-211 - get_ghg_carbon_plantings_belt_base function | • **Function Purpose**: Calculate base carbon plantings (belt) GHG sequestration per cell<br>• **Input Parameters**: `data` (Data object only - no aggregate flag)<br>• **Calculation**: `return -data.CP_BELT_AVG_T_CO2_HA * data.REAL_AREA` (line 210)<br>• **Units**: Tonnes CO2e per cell<br>• **Scaling**: Multiplied by `data.REAL_AREA` to convert from per-hectare to per-cell basis<br>• **Sign Convention**: Negative values indicate CO2 removal from atmosphere (climate beneficial)<br>• **Usage**: Base function for hybrid carbon plantings belt systems (sheep/beef + forestry) |


## luto.economics.non_agricultural.ghg.get_ghg_sheep_carbon_plantings_belt

### Data Flow Summary

```
Agricultural GHG Matrix (ag_g_mrj) + Base Carbon Plantings Belt (CP_BELT) + Exclusion Matrix →
Proportional Contribution Calculation → Mixed sheep-carbon plantings belt GHG per cell
```

### Key Transformations
1. **Hybrid Land Use**: Combines sheep grazing with carbon plantings belt
2. **Proportional Allocation**: Uses exclusion matrix to determine area splits
3. **Agricultural Component**: Sheep GHG from dryland production
4. **Forestry Component**: Belt carbon plantation sequestration
5. **Weighted Contribution**: `(1 - exclusion) × sheep_ghg + exclusion × carbon_plantings_ghg`

### Data Processing Pipeline

| Step | Code Location | Process |
|------|---------------|---------|
| 1. Agricultural GHG Matrix Input | **External Input**: `ag_g_mrj` from agricultural economics module | • **Structure**: `ag_g_mrj[m, r, j]` where m=water regime, r=cell, j=land use<br>• **Sheep Selection**: `ag_g_mrj[0, :, sheep_j]` - dryland sheep GHG per cell<br>• **Source**: Agricultural module calculations for sheep production GHG |
| 2. Exclusion Matrix Calculation | **Code**: luto/tools/__init__.py<br>**Lines**: 387-404 - get_exclusions_carbon_plantings_belt_base function | • **Purpose**: Determine proportion of cell area for carbon plantings belt vs agriculture<br>• **Base Proportion**: `exclude = np.ones(data.NCELLS) * settings.CP_BELT_PROPORTION` (line 399)<br>• **Existing Carbon Plantings**: `exclude[get_carbon_plantings_belt_cells(lumap)] = settings.CP_BELT_PROPORTION` (line 402)<br>• **Configuration**: `settings.CP_BELT_PROPORTION` defines maximum carbon plantings belt proportion per cell |
| 3. Base Carbon Plantings Belt GHG | **Code**: luto/economics/non_agricultural/ghg.py<br>**Lines**: 195-211 - get_ghg_carbon_plantings_belt_base function | • **Calculation**: `-data.CP_BELT_AVG_T_CO2_HA * data.REAL_AREA` (line 210)<br>• **Data Source**: CP_BELT carbon sequestration from belt plantings<br>• **Risk Adjustments**: Fire risk and reversal risk already applied in data.py |
| 4. Sheep-Carbon Plantings Belt Integration | **Code**: luto/economics/non_agricultural/ghg.py<br>**Lines**: 213-247 - get_ghg_sheep_carbon_plantings_belt function | • **Sheep Code**: `sheep_j = tools.get_sheep_code(data)` - get land use index for 'Sheep - modified land' (line 231)<br>• **Sheep GHG**: `sheep_cost = ag_g_mrj[0, :, sheep_j]` - dryland sheep emissions per cell (line 234)<br>• **Base Carbon Plantings**: `base_cp_cost = get_ghg_carbon_plantings_belt_base(data)` (line 235)<br>• **Carbon Plantings Contribution**: `cp_contr = base_cp_cost * cp_belt_x_r` (line 238)<br>• **Sheep Contribution**: `sheep_contr = sheep_cost * (1 - cp_belt_x_r)` (line 239)<br>• **Total GHG**: `ghg_total = cp_contr + sheep_contr` (line 240)<br>• **Output Format**: Returns numpy array (aggregate=True) or DataFrame (aggregate=False) |


## luto.economics.non_agricultural.ghg.get_ghg_beef_carbon_plantings_belt

### Data Flow Summary

```
Agricultural GHG Matrix (ag_g_mrj) + Base Carbon Plantings Belt (CP_BELT) + Exclusion Matrix →
Proportional Contribution Calculation → Mixed beef-carbon plantings belt GHG per cell
```

### Key Transformations
1. **Hybrid Land Use**: Combines beef grazing with carbon plantings belt
2. **Proportional Allocation**: Uses exclusion matrix to determine area splits
3. **Agricultural Component**: Beef GHG from dryland production
4. **Forestry Component**: Belt carbon plantation sequestration
5. **Weighted Contribution**: `(1 - exclusion) × beef_ghg + exclusion × carbon_plantings_ghg`

### Data Processing Pipeline

| Step | Code Location | Process |
|------|---------------|---------|
| 1. Agricultural GHG Matrix Input | **External Input**: `ag_g_mrj` from agricultural economics module | • **Structure**: `ag_g_mrj[m, r, j]` where m=water regime, r=cell, j=land use<br>• **Beef Selection**: `ag_g_mrj[0, :, beef_j]` - dryland beef GHG per cell<br>• **Source**: Agricultural module calculations for beef production GHG |
| 2. Exclusion Matrix Calculation | **Code**: luto/tools/__init__.py<br>**Lines**: 387-404 - get_exclusions_carbon_plantings_belt_base function | • **Purpose**: Determine proportion of cell area for carbon plantings belt vs agriculture<br>• **Base Proportion**: `exclude = np.ones(data.NCELLS) * settings.CP_BELT_PROPORTION` (line 399)<br>• **Existing Carbon Plantings**: `exclude[get_carbon_plantings_belt_cells(lumap)] = settings.CP_BELT_PROPORTION` (line 402)<br>• **Configuration**: `settings.CP_BELT_PROPORTION` defines maximum carbon plantings belt proportion per cell |
| 3. Base Carbon Plantings Belt GHG | **Code**: luto/economics/non_agricultural/ghg.py<br>**Lines**: 195-211 - get_ghg_carbon_plantings_belt_base function | • **Calculation**: `-data.CP_BELT_AVG_T_CO2_HA * data.REAL_AREA` (line 210)<br>• **Data Source**: CP_BELT carbon sequestration from belt plantings<br>• **Risk Adjustments**: Fire risk and reversal risk already applied in data.py |
| 4. Beef-Carbon Plantings Belt Integration | **Code**: luto/economics/non_agricultural/ghg.py<br>**Lines**: 250-284 - get_ghg_beef_carbon_plantings_belt function | • **Beef Code**: `beef_j = tools.get_beef_code(data)` - get land use index for 'Beef - modified land' (line 268)<br>• **Beef GHG**: `beef_cost = ag_g_mrj[0, :, beef_j]` - dryland beef emissions per cell (line 271)<br>• **Base Carbon Plantings**: `base_cp_cost = get_ghg_carbon_plantings_belt_base(data)` (line 272)<br>• **Carbon Plantings Contribution**: `cp_contr = base_cp_cost * cp_belt_x_r` (line 275)<br>• **Beef Contribution**: `beef_contr = beef_cost * (1 - cp_belt_x_r)` (line 276)<br>• **Total GHG**: `ghg_total = cp_contr + beef_contr` (line 277)<br>• **Output Format**: Returns numpy array (aggregate=True) or DataFrame (aggregate=False)<br>• **Note**: Line 282 has incorrect column name 'SHEEP_CARBON_PLANTINGS_BELT' - should be 'BEEF_CARBON_PLANTINGS_BELT' |


## luto.economics.non_agricultural.ghg.get_ghg_beccs

### Data Flow Summary

```
Raw BECCS Input Data → data.py → BECCS_TCO2E_HA_YR →
get_ghg_beccs() → BECCS GHG emissions per cell
```

### Key Transformations
1. **BECCS Technology**: Bio-Energy with Carbon Capture and Storage
2. **Direct CO2 Removal**: Active atmospheric carbon capture through biomass processing
3. **NaN Handling**: Uses `np.nan_to_num()` to handle missing data points
4. **Spatial Scaling**: Per-hectare rate × cell area = total per cell
5. **Sign Convention**: Negative values for carbon sequestration

### Data Processing Pipeline

| Step | Code Location | Process |
|------|---------------|---------|
| 1. Raw BECCS Data Loading | **Code**: luto/data.py<br>**Lines**: 1370-1377 - Load BECCS data | • **Data Source**: External BECCS technology data (costs, revenues, GHG, energy output)<br>• **Economic Data**: `BECCS_COSTS_AUD_HA_YR` and `BECCS_REV_AUD_HA_YR` (lines 1374-1375)<br>• **GHG Data**: `BECCS_TCO2E_HA_YR` - tonnes CO2e captured per hectare per year (line 1376)<br>• **Energy Data**: `BECCS_MWH_HA_YR` - energy output in MWh per hectare per year (line 1377)<br>• **Data Format**: Numpy arrays indexed by spatial cells |
| 2. GHG Calculation Function Usage | **Code**: luto/economics/non_agricultural/ghg.py<br>**Lines**: 287-309 - get_ghg_beccs function | • **Function Purpose**: Calculate BECCS GHG emissions (negative = capture) per cell<br>• **Input Parameters**: `data` (Data object), `aggregate` (Boolean output format flag)<br>• **NaN Handling**: `np.nan_to_num(data.BECCS_TCO2E_HA_YR)` - converts NaN to 0 for cells without BECCS capability (line 304, 306)<br>• **Calculation**: `return -np.nan_to_num(data.BECCS_TCO2E_HA_YR) * data.REAL_AREA` (line 304 if aggregate=True, line 306 if aggregate=False)<br>• **Units**: Tonnes CO2e per cell<br>• **Scaling**: Multiplied by `data.REAL_AREA` to convert from per-hectare to per-cell basis<br>• **Sign Convention**: Negative values indicate CO2 removal from atmosphere (climate beneficial)<br>• **Technology Note**: BECCS actively captures CO2 from atmosphere via biomass energy production with carbon storage |


## luto.economics.non_agricultural.ghg.get_ghg_destocked_land

### Data Flow Summary

```
Base Year Land Use Map + Target Year Land Use Map + Carbon Stock Data + Habitat Contribution →
Land Use Transition Analysis → Destocking Detection → Annualized Carbon Recovery →
get_ghg_destocked_land() → Destocked land GHG emissions per cell
```

### Key Transformations
1. **Land Use Transition**: Livestock natural land → Unallocated natural land conversion
2. **Destocking Definition**: Removal of livestock from natural ecosystems
3. **Carbon Stock Recovery**: Natural regeneration of carbon following livestock removal
4. **Habitat Contribution Factor**: Biodiversity-based multiplier for carbon capacity
5. **Annualization**: Divide by CARBON_EFFECTS_WINDOW (91 years) for annual rate
6. **Sign Convention**: Positive values for carbon sequestration benefit

### Data Processing Pipeline

| Step | Code Location | Process |
|------|---------------|---------|
| 1. Land Use Maps and Carbon Stock Data | **Code**: luto/data.py<br>**Multiple locations** - Land use maps and carbon stock loading | • **Base Year Map**: `data.lumaps[data.YR_CAL_BASE]` - land use in 2010 (line 332)<br>• **Target Year Map**: `lumap` parameter - land use in simulation year<br>• **Carbon Stock**: `data.CO2E_STOCK_UNALL_NATURAL_TCO2_HA` - carbon stock potential of unallocated natural land (line 337)<br>• **Livestock Categories**: `data.LU_LVSTK_NATURAL` - livestock on natural land categories (line 335)<br>• **Habitat Contribution**: `data.BIO_HABITAT_CONTRIBUTION_LOOK_UP` - biodiversity-based carbon capacity factors (line 338) |
| 2. Land Use Transition Detection | **Code**: luto/economics/non_agricultural/ghg.py<br>**Lines**: 331-342 - Destocked land identification | • **Initialization**: `penalty_ghg_r = np.zeros(data.NCELLS)` - create empty GHG array (line 333)<br>• **Transition Loop**: `for from_lu in data.LU_LVSTK_NATURAL:` - iterate through livestock natural land uses (line 335)<br>• **Cell Selection**: `lumap_BASE_YR == from_lu` - identify cells that were livestock natural land in 2010 (line 336)<br>• **Destocking Condition**: Only cells that transition from livestock natural land to unallocated natural land qualify |
| 3. Carbon Recovery Calculation | **Code**: luto/economics/non_agricultural/ghg.py<br>**Lines**: 336-341 - Carbon sequestration from destocking | • **Base Carbon Stock**: `data.CO2E_STOCK_UNALL_NATURAL_TCO2_HA[lumap_BASE_YR == from_lu]` - natural carbon capacity per hectare (line 337)<br>• **Habitat Factor**: `data.BIO_HABITAT_CONTRIBUTION_LOOK_UP[from_lu] - 1` - additional carbon capacity from biodiversity recovery (line 338)<br>• **Spatial Scaling**: `* data.REAL_AREA[lumap_BASE_YR == from_lu]` - convert to per-cell basis (line 339)<br>• **Annualization**: `/ settings.CARBON_EFFECTS_WINDOW` - divide by 91 years for annual carbon sequestration rate (line 340)<br>• **Accumulation**: `penalty_ghg_r[lumap_BASE_YR == from_lu] = (...)` - assign calculated values to relevant cells (line 336) |
| 4. GHG Output Function | **Code**: luto/economics/non_agricultural/ghg.py<br>**Lines**: 343-349 - get_ghg_destocked_land function | • **Function Purpose**: Calculate annual GHG benefit from destocking livestock from natural land<br>• **Input Parameters**: `data` (Data object), `lumap` (current land use map), `aggregate` (Boolean output format flag)<br>• **Calculation**: Returns `penalty_ghg_r` array with positive values for carbon sequestration (line 344, 346)<br>• **Units**: Tonnes CO2e per cell per year<br>• **Output Format**: Returns numpy array (aggregate=True) or DataFrame (aggregate=False)<br>• **Ecological Note**: Represents carbon recovery from removing livestock pressure on natural ecosystems |


## luto.economics.non_agricultural.ghg.get_ghg_matrix

### Data Flow Summary

```
All Individual Non-Agricultural GHG Functions + Agricultural GHG Matrix →
Matrix Assembly and Concatenation → Comprehensive Non-Agricultural GHG Matrix
```

### Key Transformations
1. **Matrix Assembly**: Combines all non-agricultural GHG sources into unified structure
2. **Agricultural Integration**: Incorporates agricultural GHG data for hybrid systems
3. **Exclusion Matrix Application**: Applies spatial constraints for agroforestry and carbon plantings
4. **Matrix Reshaping**: Converts 1D arrays to (r, k) matrix format for optimization solver
5. **Output Format Control**: Returns either aggregated matrix or detailed DataFrame

### Data Processing Pipeline

| Step | Code Location | Process |
|------|---------------|---------|
| 1. Exclusion Matrix Preparation | **Code**: luto/economics/non_agricultural/ghg.py<br>**Lines**: 372-373 - Calculate exclusion matrices | • **Agroforestry Exclusions**: `agroforestry_x_r = tools.get_exclusions_agroforestry_base(data, lumap)` (line 372)<br>• **Carbon Plantings Exclusions**: `cp_belt_x_r = tools.get_exclusions_carbon_plantings_belt_base(data, lumap)` (line 373)<br>• **Purpose**: Determine spatial allocation constraints for hybrid land use systems |
| 2. Non-Agricultural GHG Matrix Assembly | **Code**: luto/economics/non_agricultural/ghg.py<br>**Lines**: 375-386 - Call all individual GHG functions | • **Environmental Plantings**: `get_ghg_env_plantings(data, aggregate)` (line 378)<br>• **Riparian Plantings**: `get_ghg_rip_plantings(data, aggregate)` (line 379)<br>• **Sheep Agroforestry**: `get_ghg_sheep_agroforestry(data, ag_g_mrj, agroforestry_x_r, aggregate)` (line 380)<br>• **Beef Agroforestry**: `get_ghg_beef_agroforestry(data, ag_g_mrj, agroforestry_x_r, aggregate)` (line 381)<br>• **Carbon Plantings Block**: `get_ghg_carbon_plantings_block(data, aggregate)` (line 382)<br>• **Sheep Carbon Plantings Belt**: `get_ghg_sheep_carbon_plantings_belt(data, ag_g_mrj, cp_belt_x_r, aggregate)` (line 383)<br>• **Beef Carbon Plantings Belt**: `get_ghg_beef_carbon_plantings_belt(data, ag_g_mrj, cp_belt_x_r, aggregate)` (line 384)<br>• **BECCS**: `get_ghg_beccs(data, aggregate)` (line 385)<br>• **Destocked Land**: `get_ghg_destocked_land(data, lumap, aggregate)` (line 386) |
| 3. Matrix Aggregation (aggregate=True) | **Code**: luto/economics/non_agricultural/ghg.py<br>**Lines**: 388-393 - Matrix reshaping and concatenation | • **Reshape Operation**: `[non_agr_ghg_matrix.reshape((data.NCELLS, 1)) for non_agr_ghg_matrix in non_agr_ghg_matrices.values()]` (lines 390-392)<br>• **Purpose**: Convert 1D arrays (indexed by r) to 2D matrix format (r, k) where k represents different non-agricultural land use options<br>• **Concatenation**: `np.concatenate(non_agr_ghg_matrices, axis=1)` (line 393)<br>• **Final Structure**: Matrix with rows=spatial cells, columns=non-agricultural land use options |
| 4. DataFrame Output (aggregate=False) | **Code**: luto/economics/non_agricultural/ghg.py<br>**Lines**: 395-396 - DataFrame concatenation | • **DataFrame Assembly**: `pd.concat(list(non_agr_ghg_matrices.values()), axis=1)` (line 396)<br>• **Purpose**: Provides detailed view with named columns for each non-agricultural GHG source<br>• **Column Names**: 'ENV_PLANTINGS', 'RIP_PLANTINGS', 'SHEEP_AGROFORESTRY', 'BEEF_AGROFORESTRY', 'CARBON_PLANTINGS_BLOCK', 'SHEEP_CARBON_PLANTINGS_BELT', 'BEEF_CARBON_PLANTINGS_BELT', 'BECCS', 'DESTOCKED_LAND' |
| 5. Integration with Optimization Solver | **Code**: Used by solver modules for optimization | • **Matrix Usage**: Output matrix feeds into GUROBI optimization solver as constraint/objective coefficients<br>• **Dimensions**: Rows (r) = spatial cells, Columns (k) = non-agricultural land use decision variables<br>• **Units**: All values in tonnes CO2e per cell per year<br>• **Sign Convention**: Negative values represent climate benefits (carbon sequestration/emission reduction) |


### Function Summary Matrix

| Function | Primary Data Source | Plantation Type | Hybrid System | Key Features |
|----------|-------------------|-----------------|---------------|--------------|
| `get_ghg_env_plantings` | EP_BLOCK_AVG_T_CO2_HA | Block (full area) | No | Base environmental restoration |
| `get_ghg_rip_plantings` | EP_RIP_AVG_T_CO2_HA | Riparian (waterway) | No | Waterway restoration |
| `get_ghg_agroforestry_base` | EP_BELT_AVG_T_CO2_HA | Belt (boundary) | No | Base for hybrid systems |
| `get_ghg_sheep_agroforestry` | EP_BELT + ag_g_mrj | Belt + Agriculture | Yes | Sheep + Agroforestry |
| `get_ghg_beef_agroforestry` | EP_BELT + ag_g_mrj | Belt + Agriculture | Yes | Beef + Agroforestry |
| `get_ghg_carbon_plantings_block` | CP_BLOCK_AVG_T_CO2_HA | Block (full area) | No | Carbon sequestration focus |
| `get_ghg_carbon_plantings_belt_base` | CP_BELT_AVG_T_CO2_HA | Belt (boundary) | No | Base for hybrid systems |
| `get_ghg_sheep_carbon_plantings_belt` | CP_BELT + ag_g_mrj | Belt + Agriculture | Yes | Sheep + Carbon Plantings |
| `get_ghg_beef_carbon_plantings_belt` | CP_BELT + ag_g_mrj | Belt + Agriculture | Yes | Beef + Carbon Plantings |
| `get_ghg_beccs` | BECCS_TCO2E_HA_YR | Technology-based | No | Active CO2 capture |
| `get_ghg_destocked_land` | CO2E_STOCK_UNALL_NATURAL_TCO2_HA | Natural recovery | No | Livestock removal benefit |
| `get_ghg_matrix` | All above functions | All types | Mixed | Complete matrix assembly |