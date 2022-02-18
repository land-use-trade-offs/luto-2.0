LUTO II - Land-Use Trade-Offs modelling framework
=================================================

LUTO II is version two of the Land-Use Trade-Offs model. The model predicts future spatial land-use distributions based on a economic cost-minimisation logic under various constraints, including environmental limits. The model is contained in a Python package (the framework) and can be run interactively or in a scripted manner (for batch runs).

The LUTO II modelling framework was developed at Deakin University by Fjalar de Haan, Brett Bryan and colleagues (see [CITATION.cff](CITATION.cff)) from early 2021 to early 2022. It it the product of the Land-Use Futures project which is a collaboration between ClimateWorks Australia, the CSIRO and Deakin University (see [luf@cwa](https://www.climateworksaustralia.org/project/land-use-futures/) and [luf@deakin](https://www.planet-a.earth/other-projects-1/e6xzzv5emwd7p9fsd8pxyluv4840iz)). LUTO II continues the approach to land-use change modelling of its predecessor, the original LUTO, which was developed at the CSIRO in the latter half of the 2010s (see also [Pedigree](#pedigree), below) and published under the GNU GPLv3 in 2021.

# Pedigree #

LUTO II is based on the original LUTO as regards its overall approach to land-use change modelling. In particular aspects like discretising the land-use map at a 1x1 square kilometres scale representing it as a 1D array with the land-uses as integers. The original LUTO model was developed at the Commonwealth Scientific and Industrial Research Organisation (CSIRO) between 201x and 201x and published under the GNU GPLv3 in 2021. The code and more information about the original LUTO model can be at [luto@csiro](https://data.csiro.au/collection/csiro:52376v1) and should be cited as Bryan, Brett; Nolan, Martin; Stock, Florian; Graham, Paul; Dunstall, Simon; Ernst, Andreas; Connor, Jeff (2021): Land Use Trade-Offs (LUTO) Model. v1. CSIRO. Software Collection. [https://doi.org/10.25919/y8ee-sk45](https://doi.org/10.25919/y8ee-sk45).

The LUTO II code base is, however, a new model, written completely from scratch. The LUTO II economic logic governing land-use change is very different from that of the original LUTO. While the original LUTO is based on the assumption that farmers try to optimise their profits, the leading LUTO II assumption is that the overall agricultural system tries to minimise its costs of production (including costs of switching between agricultural commodities). Both LUTOs are optimisation models but different commercial solvers are used (CPLEX in original LUTO, GUROBI in LUTO II).





LUTO II is therefore an optimisation model with a linear programme at heart. To do the actual solving, the model interfaces with commercial solver GUROBI.

The original LUTO (i.e. LUTO I) was developed at the Commonwealth Scientific and Industrial Research Organisation (CSIRO) between 201x and 201x and published under the GNU GPLv3 in 2021.


