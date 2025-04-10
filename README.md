# LUTO 2: The Land-Use Trade-Offs Model Version 2

## Introduction
The Land Use Trade-Offs model v2 (LUTO2) is an integrated land systems model designed to simulate the optimal spatial arrangement of land use and land management decisions over time in Australia. It aims to achieve climate and biodiversity targets without compromising economic growth, food production or water security. The model is implemented as a Python package, offering users the flexibility to run interactively or to execute batch processes through scripted automation.

LUTO2 was developed through a collaboration between Deakin University and Climateworks Centre, with research contributions from CSIRO. The model is a cornerstone of Climateworks’ Land Use Futures program, which supports Australia’s transition to sustainable food and land systems. The technical development of LUTO2 is led by Professor Brett Bryan at Deakin University. LUTO2 continues the approach to land-use change modelling of its predecessor, the original LUTO, which was developed by CSIRO from 2010 - 2015 (see also Pedigree, below) and published under the GNU GPLv3 in 2021.

## Pedigree
LUTO2 builds on the approach and pedigree of nearly two decades of land-use modelling expertise starting with the original LUTO model. The original LUTO model was developed by CSIRO for the Australian National Outlook in 2015 and was groundbreaking for quantifying and projecting land use changes and their sustainability impacts in Australia, illustrated by its published works in *Nature* in 2015 and 2017.

LUTO2 represents a generational leap in sophistication and functionality for national-scale land-use change modelling in Australia. Both LUTO versions are optimisation models but different commercial solvers are used (CPLEX in original LUTO, GUROBI in LUTO2). The spatial domains are different in extent, with LUTO2's being nearly 5 times as large. The data requirements to run LUTO2 are consequently different and heavier. There is no backwards compatibility whatsoever.

The original LUTO model is available online and should be cited as:
> Bryan, Brett; Nolan, Martin; Stock, Florian; Graham, Paul; Dunstall, Simon; Ernst, Andreas; Connor, Jeff (2021): Land Use Trade-Offs (LUTO) Model. v1. CSIRO. Software Collection. https://doi.org/10.25919/y8ee-sk45.

This new version represents an entirely new model featuring a complete rewrite of the codebase and comprehensive upgrades to data and functionality. Enhancements to the original model include extended spatial coverage and timespan (2010 to 2100), a complete refresh of input data, additional land-use options and sustainability indicators and management solutions, the ability to model demand-side solutions, and additional environmental indicators and reporting. Due to LUTO2’s model complexity, the computational requirements to run the model are far more intensive.

LUTO2’s modelling approach, indicators and solutions have been guided by extensive stakeholder consultation (documented here: https://doi.org/10.1007/s11625-024-01574-7) following principles of participatory model co-design.

## Authors
Coordinating lead author: **Bryan, B.A.**  
Lead authors (in order of contribution): **Williams, N., Archibald, C.L., de Haan, F., Wang, J., van Schoten, N., Hadjikakou, M., Sanson, J., Zyngier, R., Marcos-Martinez, R., Navarro, J., Gao, L., Aghighi, H.**  
Other significant contributors (in alphabetical order): **Armstrong, T., Bohl, H., Jaffe, P., Khan, M.S., Moallemi, E.A., Nazari, A., Pan, X., Steyl, D., Thiruvady, D.R.**

## Documentation
Documentation, including instructions on how to set up and run LUTO2, can be found at `docs/luto2-overview.pdf`.

LUTO2 comes with a full diagram to illustrate its data preparation, workflow, and code logics. The diagram link can be found in this link.  
*Replace with updated documentation currently in preparation when ready.*

## Steps to Run LUTO2
1. Clone the repository:
   ```sh
   git clone https://github.com/land-use-trade-offs/luto-2.0.git
   ```
2. Create a Conda environment and install dependencies:
   ```sh
   cd luto-2.0
   conda env create luto/tools/create_task_runs/bash_scripts/conda_env.yml
   conda activate luto_env
   ```

3. Run a basic simulation:
   ```python
   import luto.simulation as sim
   data = sim.load_data()
   sim.run(data=data, base_year=2010, target_year=2030, step_size=5)
   ```

4. After execution, an HTML report will be generated for easier visualization:
   ```
   /output/<run_dir>/DATA_REPORT/REPORT_HTML/index.html
   ```
   This report provides a structured overview of the results in an interactive format.
   
5. *Note:* LUTO2 requires a large spatio-temporal database located in the `/input` folder before simulation. This dataset can be obtained from **b.bryan@deakin.edu.au**.

## Copyright
Copyright 2024 **Bryan, B.A., Williams, N., Archibald, C.L., de Haan, F., Wang, J., van Schoten, N., Hadjikakou, M., Sanson, J., Zyngier, R., Marcos-Martinez, R., Navarro, J., Gao, L., Aghighi, H., Armstrong, T., Bohl, H., Jaffe, P., Khan, M.S., Moallemi, E.A., Nazari, A., Pan, X., Steyl, D., and Thiruvady, D.R.**  
Copyright 2021-2023 **Fjalar J. de Haan and Brett A. Bryan, Deakin University.** (see `CITATION.cff`).

## License
LUTO2 is free software: you can redistribute it and/or modify it under the terms of the **GNU General Public License** as published by the **Free Software Foundation**, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but **without any warranty**; without even the implied warranty of **merchantability** or **fitness for a particular purpose**. See the **GNU General Public License** for more details.

You should have received a copy of the **GNU General Public License** along with this program. If not, see <https://www.gnu.org/licenses/>.

## Citation
> Bryan, B.A., Williams, N., Archibald, C.L., de Haan, F., Wang, J., van Schoten, N., Hadjikakou, M., Sanson, J., Zyngier, R., Marcos-Martinez, R., Navarro, J., Gao, L., Aghighi, H., Armstrong, T., Bohl, H., Jaffe, P., Khan, M.S., Moallemi, E.A., Nazari, A., Pan, X., Steyl, D., and Thiruvady, D.R. (2024). The Land-Use Trade-Offs Model Version 2 (LUTO2): an integrated land system model for Australia. Software Collection. https://github.com/land-use-trade-offs/luto-2.0

