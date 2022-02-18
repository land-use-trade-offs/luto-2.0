LUTO II - Land-Use Trade-Offs modelling framework
=================================================

LUTO II is version two of the Land-Use Trade-Offs model. The model predicts future spatial land-use distributions based on a economic cost-minimisation logic under various constraints, including environmental limits. The model is contained in a Python package (the framework) and can be run interactively or in a scripted manner (for batch runs).

The LUTO II modelling framework was developed at Deakin University by Fjalar de Haan, Brett Bryan and colleagues (see [CITATION.cff](CITATION.cff)) from early 2021 to early 2022. It it the product of the Land-Use Futures project which is a collaboration between ClimateWorks Australia, the CSIRO and Deakin University (see [luf@cwa](https://www.climateworksaustralia.org/project/land-use-futures/) and [luf@deakin](https://www.planet-a.earth/other-projects-1/e6xzzv5emwd7p9fsd8pxyluv4840iz)). LUTO II continues the approach to land-use change modelling of its predecessor, the original LUTO, which was developed at the CSIRO in the latter half of the 2010s (see also [Pedigree](#pedigree), below) and published under the GNU GPLv3 in 2021.

# Pedigree #

LUTO II is based on the original LUTO as regards its overall approach to land-use change modelling. In particular aspects like discretising the land-use map at a 1x1 square kilometres scale and representing it as a 1D array with land-uses as integers. The code and more information about the original LUTO model can be found at [luto@csiro](https://data.csiro.au/collection/csiro:52376v1) and should be cited as: Bryan, Brett; Nolan, Martin; Stock, Florian; Graham, Paul; Dunstall, Simon; Ernst, Andreas; Connor, Jeff (2021): Land Use Trade-Offs (LUTO) Model. v1. CSIRO. Software Collection. [https://doi.org/10.25919/y8ee-sk45](https://doi.org/10.25919/y8ee-sk45).

The LUTO II code base encompasses, however, a new model, written completely from scratch - there is no original LUTO code in the LUTO II code base. The LUTO II economic logic governing land-use change is very different from that of the original LUTO. While the original LUTO is based on the assumption that farmers try to optimise their profits, the leading LUTO II assumption is that the overall agricultural system tries to minimise its costs of production (including costs of switching between agricultural commodities). Whilst the original LUTO allowed switching from a current land-use to an alternative in a one-way fashion, LUTO II features full commodity switching. Both LUTOs are optimisation models but different commercial solvers are used (CPLEX in original LUTO, GUROBI in LUTO II). The spatial domains are different in extent, with LUTO II's being nearly 10 times as large. The data requirements to run LUTO II are consequentially different and heavier. There is no backwards compatibility whatsoever.

# Documentation #
Documentation, including instructions on how to set up and run LUTO II, can be found at [docs/luto2-overview.pdf](docs/luto2-overview.pdf).

# Copyright #
Copyright 2021 Fjalar J. de Haan and Brett A. Bryan, Deakin University.

# Licence #
LUTO II is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see [\<https://www.gnu.org/licenses/\>](https://www.gnu.org/licenses/).

# Authors and Citation #
See the [CITATION.cff](CITATION.cff) file for the LUTO II authors and how to cite the code base.




