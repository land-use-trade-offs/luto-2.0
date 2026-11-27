#!/usr/bin/env python
"""Build the NLUM v7.1 land-use class allocation on the LUTO grid, 2010-11 (control), 2015-16 and 2020-21, from public sources.

Sources (public, downloaded at run time, sha256-checked against pinned values):
  - ABARES NLUM v7.1 250 m ALUMV8 rasters and class tables, 2010-11, 2015-16 and 2020-21 (CC BY 4.0).
  - ABARES NLUM v7.1 agricultural commodity probability surfaces, 2010-11, 2015-16 and 2020-21 (CC BY 4.0).
  - ABS ASGS Edition 3 SA2 2021 boundaries, GDA94 (CC BY 4.0).
Inherited 2010-build inputs read from --input (inherited_2010/, copies of 2010-build files), each sha256 recorded in provenance.csv: NLUM_2010-11_mask.tif,
real_area.h5, lumap.h5, lmmap.h5, state_id.npy. The outputs are therefore not built from public sources alone.

Method:
  1. Definition crosswalk from ALUM v8 tertiary class to LUTO (lu, lm): the XWALK reference table below, written out as
     xwalk_definition.csv.
  2. Every non-zero v7.1 pixel centre (EPSG:3577) goes to the 0.01 degree LUTO cell containing it (EPSG:4283, same datum).
     One-to-many classes are split over their candidates by the probability surfaces at the pixel centre, normalised over
     the candidates; pixels without probability information take the class's split in their SA2, else state, else
     nationally.
  3. SA2_2021 per cell: the polygon at the cell centre, else the nearest labelled pixel; a cell whose SA2 state digit
     disagrees with state_id takes the SA2 of the nearest cell that agrees.
  4. Area-preserving allocation of whole cells to classes within each SA2: floor quota in descending share order, then
     largest remainder.
  5. Map rules: footprint fixed at the 2010 map; held cells keep the 2010 lumap/lmmap, except 2010 crop
     cells, retyped 23 (natural v7.1 class) or 22 (modified or intensive); grazing persistence against
     2010-11 (for 2015-16 and 2020-21 alike); unassigned modified pasture keeps the class's irrigation flag. Livestock
     typing of grazing cells is not done here.
  The classes come from the 2010-11 and 2020-21 tables; the 2015-16 table adds no value and changes no attribute read here.
  2015-16 is allocated the same way for its irrigation flag (class_alloc_2015_16_lmmap.npy, read by the irrigation rule
  of eligibility.py); the 2010-11 and 2020-21 outputs do not depend on it.
Outputs in --out: xwalk_definition.csv; sa2_2021_cells.npy (int32 SA2_2021 code per LUTO cell); for each year Y in 2010_11,
2015_16 and 2020_21, class_alloc_Y_lumap.npy (int16), class_alloc_Y_lmmap.npy (int8), class_alloc_Y_category.npy (int8,
codes in CATEGORY); provenance.csv (last).

Run as a plain script from the repository root (no luto imports):

    python input/Nick_build/scripts/nlum_map.py --input input/Nick_build/inherited_2010 --out input/Nick_build/nlum_map --cache input/Nick_build/_cache
        [--offline] [--source NAME=PATH]

Dependencies: numpy, pandas, PyTables, rasterio, pyproj, geopandas, scipy. No luto imports.
Memory: peak about 9.9 GB in the main process and 30.3 GB across the process tree with 48 worker processes (measured
October 2026); run on a machine with at least 64 GB of RAM, or lower --workers.
Output is deterministic: every output file reproduces byte for byte from the same sources, inputs and package versions;
no run timestamps.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
import shutil
import sys
import time
import urllib.request
import zipfile
from importlib import metadata
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyproj
import rasterio
from rasterio.windows import Window

# ----------------------------------------------------------------------------- sources and pins
ABARES = 'https://www.agriculture.gov.au/sites/default/files/documents/'
SOURCES = {   # name: (url, sha256 of the raw file, bytes); URLs as linked from the ABARES and ABS release pages
    'alum_2010_11': (ABARES + 'NLUM_v7_1_250m_ALUMV8_2010_11_alb_20260814.zip',
                     '0f079cd08da38756477fc0a8dc4faf06a767284f96bbd9b24d2b3f0b29c7f694', 64994041),
    'alum_2020_21': (ABARES + 'NLUM_v7_1_250m_ALUMV8_2020_21_alb_20260814.zip',
                     '6e937a1cf1e99bffc251b684cdd3f55429830fa0c7790b7191abfef995851fb7', 64220773),
    'alum_2015_16': (ABARES + 'NLUM_v7_1_250m_ALUMV8_2015_16_alb_20260814.zip',
                     'c63c3e2120e9c263863c1c2c1bd7107c38f7efb7e569af8a5164ac1b4106af29', 65005403),
    'prob_2010_11': (ABARES + 'NLUM_v7_1_250m_AgProbabilitySurfaces_2010_11_geo_20260814.zip',
                     'eaec043e7781b53b0ae9125766758d31118e333f068c16ce82061bb31f8fbb8c', 181693179),
    'prob_2020_21': (ABARES + 'NLUM_v7_1_250m_AgProbabilitySurfaces_2020_21_geo_20260814.zip',
                     'ce8549585962cfc824534edbd0a5de10bb407f90f9d18338f5f5a6817023cdd0', 177434322),
    'prob_2015_16': (ABARES + 'NLUM_v7_1_250m_AgProbabilitySurfaces_2015_16_geo_20260814.zip',
                     'b9ea2a0fc8563c211dfca99af1ed59b8dcc96a48b66bea839cbb2b933473e27c', 181437263),
    'sa2_2021': ('https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/'
                 'jul2021-jun2026/access-and-downloads/digital-boundary-files/SA2_2021_AUST_SHP_GDA94.zip',
                 '87345a241c20d64068a012616adb74c0eada6cc47ce2a897d62d78026deaf0cc', 49770620),
}
SOURCE_NOTES = {
    'alum_2010_11': 'ABARES NLUM v7.1 (release 2026-08-14), CC BY 4.0, DOI 10.25814/n0z4-5y98',
    'alum_2020_21': 'ABARES NLUM v7.1 (release 2026-08-14), CC BY 4.0, DOI 10.25814/n0z4-5y98',
    'prob_2010_11': 'ABARES NLUM v7.1 (release 2026-08-14), CC BY 4.0, DOI 10.25814/n0z4-5y98',
    'prob_2020_21': 'ABARES NLUM v7.1 (release 2026-08-14), CC BY 4.0, DOI 10.25814/n0z4-5y98',
    'alum_2015_16': 'ABARES NLUM v7.1 (release 2026-08-14), CC BY 4.0, DOI 10.25814/n0z4-5y98',
    'prob_2015_16': 'ABARES NLUM v7.1 (release 2026-08-14), CC BY 4.0, DOI 10.25814/n0z4-5y98',
    'sa2_2021': 'ABS ASGS Edition 3 (July 2021 to June 2026), SA2 2021 GDA94, CC BY 4.0',
}
INHERITED = ['NLUM_2010-11_mask.tif', 'real_area.h5', 'lumap.h5', 'lmmap.h5', 'state_id.npy']
YEARS = ('2010_11', '2020_21')                 # the class tables
ALLOC_YEARS = ('2010_11', '2015_16', '2020_21')  # the allocations; 2015-16 for its irrigation flag
NCELL = 6956407
STEP = 128                # raster rows per block; the float sums depend on it, keep it fixed
PIX_HA = 6.25             # 250 m x 250 m pixel, EPSG:3577 (equal area)

# One-to-many ALUM values: raster value -> [(LUTO lu, probability-surface unit)]
SPLIT = {331: [(25, 5), (18, 6)], 431: [(25, 5), (18, 6)], 334: [(27, 10), (20, 11)], 434: [(27, 10), (20, 11)],
         338: [(26, 8), (19, 9)], 438: [(26, 8), (19, 9)],
         341: [(0, 18), (11, 19), (16, 20), (21, 21)], 441: [(0, 18), (11, 19), (16, 20), (21, 21)]}
SPLIT_IDS = sorted(SPLIT)
NONAG, GNAT, GMOD0, GMOD1 = 0, 1, 2, 3     # allocation classes before the (lu, lm) pairs
NATC = {1: 2, 5: 6, 14: 15}                # modified-land livestock lu -> natural-land lu
MODC = {2: 1, 6: 5, 15: 14}
CATEGORY = {0: 'allocated (lu, lm) pair', 3: 'outside footprint, no v7.1 data', 10: 'natural grazing, persistent, lu kept',
            11: 'natural grazing, persistent, lu converted to natural land', 12: 'natural grazing, persistent, unassigned',
            13: 'natural grazing, new', 20: 'modified grazing, persistent, lu kept',
            21: 'modified grazing, persistent, lu converted to modified land', 22: 'modified grazing, persistent, unassigned',
            23: 'modified grazing, new', 30: 'one-to-many crop candidate', 50: 'outside footprint, ag majority',
            51: 'held at the 2010 map', 53: 'held 2010 crop cell retyped by its v7.1 class'}

LU_NAMES = ["Apples", "Beef - modified land", "Beef - natural land", "Citrus", "Cotton", "Dairy - modified land",
            "Dairy - natural land", "Grapes", "Hay", "Nuts", "Other non-cereal crops", "Pears", "Plantation fruit", "Rice",
            "Sheep - modified land", "Sheep - natural land", "Stone fruit", "Sugar", "Summer cereals", "Summer legumes",
            "Summer oilseeds", "Tropical stone fruit", "Unallocated - modified land", "Unallocated - natural land",
            "Vegetables", "Winter cereals", "Winter legumes", "Winter oilseeds"]   # LUTO ag_landuses.csv order

# ============================================================================= reference tables (hand-entered)
# XWALK: the definition crosswalk, ALUM v8 tertiary class -> LUTO (lu, lm), one row per class with its basis.
# Source of every row: ALUMC v8 handbook (ABARES, "Guidelines for land use mapping in Australia", 4th edn Part 2, update
# October 2016; ALUMCv8_Handbook4ednPart2_UpdateOctober2016.pdf, sha256
# 94c72079ecbff50a28e54cb09afca60914e696cb22a3741a5ae9e4c7478f87d7) cited as "HB p." (printed page), and the NLUM v7.1
# descriptive metadata (NLUM_v7_1_DescriptiveMetadata_20260814_0.pdf, sha256
# 1aa61c1fe34feb1686123bdc670fe789849f039141f29ff7bb70301dd08dd6a9) cited as "MD p.". Ruled as the map crosswalk.
# Columns per row: codes, LUTO lu codes (-1 = non-agricultural), lm, provenance label, rationale, confidence,
# alternative candidates, how to resolve, policy choice (Y/N).
ALUM_NAMES = {  # ALUMC v8 handbook Appendix 1 class names, in its order
    '1.1.1': 'Strict nature reserves', '1.1.2': 'Wilderness area', '1.1.3': 'National park',
    '1.1.4': 'Natural feature protection', '1.1.5': 'Habitat/species management area',
    '1.1.6': 'Protected landscape', '1.1.7': 'Other conserved area', '1.2.0': 'Managed resource protection',
    '1.2.5': 'Traditional indigenous uses', '1.3.0': 'Other minimal use', '1.3.1': 'Defence land - natural areas',
    '1.3.2': 'Stock route', '1.3.3': 'Residual native cover', '1.3.4': 'Rehabilitation',
    '2.1.0': 'Grazing native vegetation', '2.2.0': 'Production native forests', '2.2.1': 'Wood production forestry',
    '3.1.0': 'Plantation forests', '3.1.1': 'Hardwood plantation forestry', '3.1.2': 'Softwood plantation forestry',
    '3.1.3': 'Other forest plantation', '3.1.4': 'Environmental forest plantation',
    '3.2.0': 'Grazing modified pastures', '3.3.0': 'Cropping', '3.3.1': 'Cereals', '3.3.3': 'Hay and silage',
    '3.3.4': 'Oilseeds', '3.3.5': 'Sugar', '3.3.6': 'Cotton', '3.3.8': 'Pulses', '3.4.0': 'Perennial horticulture',
    '3.4.1': 'Tree fruits', '3.4.3': 'Tree nuts', '3.4.5': 'Shrub berries and fruits', '3.4.8': 'Citrus',
    '3.4.9': 'Grapes', '3.5.4': 'Seasonal vegetables and herbs', '4.1.0': 'Irrigated plantation forests',
    '4.1.1': 'Irrigated hardwood plantation forestry', '4.1.2': 'Irrigated softwood plantation forestry',
    '4.1.3': 'Irrigated other forest production', '4.1.4': 'Irrigated environmental forest plantation',
    '4.2.0': 'Grazing irrigated modified pastures', '4.3.0': 'Irrigated cropping', '4.3.1': 'Irrigated cereals',
    '4.3.3': 'Irrigated hay and silage', '4.3.4': 'Irrigated oilseeds', '4.3.5': 'Irrigated sugar',
    '4.3.6': 'Irrigated cotton', '4.3.8': 'Irrigated pulses', '4.3.9': 'Irrigated rice',
    '4.4.0': 'Irrigated perennial horticulture', '4.4.1': 'Irrigated tree fruits', '4.4.3': 'Irrigated tree nuts',
    '4.4.5': 'Irrigated shrub berries and fruits', '4.4.8': 'Irrigated citrus', '4.4.9': 'Irrigated grapes',
    '4.5.3': 'Irrigated seasonal vegetables and herbs', '5.0.0': 'Intensive uses', '5.1.0': 'Intensive horticulture',
    '5.1.1': 'Production nurseries', '5.1.2': 'Shadehouses', '5.1.3': 'Glasshouses',
    '5.1.4': 'Glasshouses (hydroponic)', '5.1.5': 'Abandoned intensive horticulture',
    '5.2.0': 'Intensive animal production', '5.2.1': 'Dairy sheds and yards', '5.2.2': 'Feedlots',
    '5.2.3': 'Poultry farms', '5.2.4': 'Piggeries', '5.2.5': 'Aquaculture', '5.2.6': 'Horse studs',
    '5.2.7': 'Saleyards/stockyards', '5.2.8': 'Abandoned intensive animal production',
    '5.3.0': 'Manufacturing and industrial', '5.3.1': 'General purpose factory', '5.3.2': 'Food processing factory',
    '5.3.3': 'Major industrial complex', '5.3.4': 'Bulk grain storage', '5.3.5': 'Abattoirs',
    '5.3.6': 'Oil refinery', '5.3.7': 'Sawmill', '5.3.8': 'Abandoned manufacturing and industrial',
    '5.4.0': 'Residential and farm infrastructure', '5.4.1': 'Urban residential',
    '5.4.2': 'Rural residential with agriculture', '5.4.3': 'Rural residential without agriculture',
    '5.4.4': 'Remote communities', '5.4.5': 'Farm buildings/infrastructure', '5.5.0': 'Services',
    '5.5.1': 'Commercial services', '5.5.2': 'Public services', '5.5.3': 'Recreation and culture',
    '5.5.4': 'Defence facilities - urban', '5.5.5': 'Research facilities', '5.6.0': 'Utilities',
    '5.6.1': 'Fuel powered electricity generation', '5.6.2': 'Hydro electricity generation',
    '5.6.3': 'Wind electricity generation', '5.6.4': 'Solar electricity generation',
    '5.6.5': 'Electricity substations and transmission', '5.6.6': 'Gas treatment, storage and transmission',
    '5.6.7': 'Water extraction and transmission', '5.7.0': 'Transport and communication',
    '5.7.1': 'Airports/aerodromes', '5.7.2': 'Roads', '5.7.3': 'Railways', '5.7.4': 'Ports and water transport',
    '5.7.5': 'Navigation and communication', '5.8.0': 'Mining', '5.8.1': 'Mines', '5.8.2': 'Quarries',
    '5.8.3': 'Tailings', '5.8.4': 'Extractive industry not in use', '5.9.0': 'Waste treatment and disposal',
    '5.9.1': 'Effluent pond', '5.9.2': 'Landfill', '5.9.3': 'Solid garbage', '5.9.5': 'Sewage/sewerage',
    '6.0.0': 'Water', '6.1.0': 'Lake', '6.1.1': 'Lake - conservation', '6.1.2': 'Lake - production',
    '6.1.3': 'Lake - intensive use', '6.1.4': 'Lake - saline', '6.2.0': 'Reservoir/dam', '6.2.1': 'Reservoir',
    '6.2.2': 'Water storage - intensive use/farm dams', '6.2.3': 'Evaporation basin', '6.3.0': 'River',
    '6.3.1': 'River - conservation', '6.3.2': 'River - production', '6.3.3': 'River - intensive use',
    '6.4.0': 'Channel/aqueduct', '6.4.1': 'Supply channel/aqueduct', '6.4.2': 'Drainage channel/aqueduct',
    '6.4.3': 'Stormwater', '6.5.0': 'Marsh/wetland', '6.5.1': 'Marsh/wetland - conservation',
    '6.5.2': 'Marsh/wetland - production', '6.5.3': 'Marsh/wetland - intensive use',
    '6.5.4': 'Marsh/wetland - saline', '6.6.0': 'Estuary/coastal waters',
    '6.6.1': 'Estuary/coastal waters - conservation', '6.6.2': 'Estuary/coastal waters - production',
    '6.6.3': 'Estuary/coastal waters - intensive use',
}
TO_NONAG = [-1]

XWALK = {}


def m(codes, lus, lm, prov, why, conf, alt="", resolve="", policy="N"):
    for c in codes.split():
        assert c not in XWALK, c
        XWALK[c] = (lus, lm, prov, why, conf, alt, resolve, policy)


# ---- Primary 1
m("1.1.1 1.1.2 1.1.3 1.1.4 1.1.5", TO_NONAG, "-", "inferred",
  "CAPAD/IUCN Ia-IV protected area; nature conservation is prime use (HB p.5); not an ag land use", "high")
m("1.1.6", TO_NONAG, "-", "inferred",
  "IUCN V protected landscape; conservation prime use (HB p.6); grazing may occur as ancillary use", "medium",
  alt="2;15 (natural-land grazing) or 23", resolve="policy: grazing inside protected areas in or out of LUTO scope",
  policy="Y")
m("1.1.7", TO_NONAG, "-", "inferred",
  "Other conserved area incl. heritage/voluntary agreements on private land (HB p.6); conservation prime use",
  "medium", alt="23 or 2;15", resolve="policy: private conservation covenants in or out of LUTO study area",
  policy="Y")
m("1.2.0", TO_NONAG, "-", "inferred",
  "Managed resource protection, CAPAD IUCN VI, sustainable use of natural resources (HB p.6); reserved tenure",
  "medium", alt="23", resolve="policy: treat IUCN VI like other reserves", policy="Y")
m("1.2.5", [23], 0, "inferred",
  "Traditional Indigenous uses as current prime use (HB p.6); native cover, no ag commodity; not a CAPAD class",
  "low", alt="-1", resolve="policy: Indigenous-managed land in or out of study area", policy="Y")
m("1.3.0", [23], 0, "inferred",
  "Other minimal use: largely unused land (HB p.6); v7.1 also assigns 1.3.0 to low-NDVI-range cells reclassified "
  "from cropping inside CLUM class 1 areas (MD p.12)", "medium")
m("1.3.1", TO_NONAG, "-", "inferred",
  "Defence land natural areas for field training/weapons testing (HB p.6); defence tenure not available to ag",
  "medium", alt="23", resolve="policy: defence land in or out of study area", policy="Y")
m("1.3.2", [23], 0, "inferred",
  "Stock route: stock reserves under intermittent use or unused (HB p.6); native cover, intermittent grazing only",
  "low", alt="2;15 (natural-land grazing)",
  resolve="policy: count intermittent stock-route grazing as a livestock land use", policy="Y")
m("1.3.3", [23], 0, "inferred",
  "Residual native cover: native cover, mainly unused or non-production; if grazing is prime use it is 2.1.0 not "
  "1.3.3 (HB p.6-7)", "high", alt="-1",
  resolve="policy: whether residual native cover is available for clearing/ag in LUTO", policy="Y")
m("1.3.4", [23], 0, "inferred", "Rehabilitation: land restored to a near natural state (HB p.7)", "medium")
# ---- Primary 2
m("2.1.0", [2, 15, 6], 0, "stated and inferred",
  "Grazing by domestic stock on native vegetation, >50% native species (HB p.7); cattle meat, cattle dairy and "
  "sheep commodities apply to 2.1.0 (HB p.32-34). natural land = native vegetation (inferred). Dryland: class 2 has "
  "no irrigated variant and LUTO has no irrigated natural-land grazing (dataprep.py:554)", "medium",
  alt="23 where no stock present",
  resolve="ABS census livestock by SA2 (beef/sheep/dairy) or LUTO livestock yield layers; ALUM commodity field "
          "(HB p.7) if available")
m("2.2.0 2.2.1", [23], 0, "inferred",
  "Production native forests: commercial production from native forest (HB p.7); v7.1 allocates 2.2.0 across all "
  "multiple-use public forest and harvested area is <1% (MD p.12); native cover, no LUTO forestry land use", "low",
  alt="-1", resolve="policy: native production forest in or out of study area; grazing in state forests is common "
                    "(HB p.27)", policy="Y")
# ---- Primary 3 plantations and pasture
m("3.1.0 3.1.1 3.1.2 3.1.3", [22], 0, "inferred",
  "Plantation forests for production (HB p.8); class 3 = native vegetation largely replaced (HB p.8), so modified "
  "land; LUTO has no plantation-forestry ag land use", "low", alt="-1",
  resolve="policy: plantation forestry as frozen non-ag (-1) or convertible Unallocated - modified land", policy="Y")
m("3.1.4", [22], 0, "inferred",
  "Environmental forest plantation incl. trees planted for carbon credits (HB p.8); modified land, no LUTO ag "
  "land use", "low", alt="-1 or non-ag Environmental Plantings / Carbon Plantings (Block)",
  resolve="policy: seed base-year non-ag plantings from 3.1.4/4.1.4", policy="Y")
m("3.2.0", [1, 14, 5], 0, "stated and inferred",
  "Grazing modified pastures, >50% exotic species (HB p.8); cattle meat, cattle dairy and sheep apply to 3.2.0 "
  "(HB p.32-34). modified land = modified pasture (inferred). Dryland: primary 3 (HB p.8)", "medium",
  alt="22 where no stock; a cropping lu if in crop-pasture rotation (HB p.8-9)",
  resolve="ABS census livestock by SA2 or LUTO livestock layers; dairy region layers")
# ---- Primary 3 cropping/horticulture (dry)
m("3.3.0", [10], 0, "stated and inferred",
  "Cropping, generic (HB p.9). NLUM v7.1 mapping unit 15 'Other non-cereal crops' carries ALUM code 330 (MD p.25), "
  "same name as LUTO lu 10", "medium", alt="any LUTO crop lu",
  resolve="if any 3.3.0 cells come from a non-ONCC source, split with commodity probability surfaces")
m("3.3.1", [25, 18], 0, "stated",
  "Cereals grown for grain (HB p.10). NLUM mapping units 5 Winter cereals and 6 Summer cereals excl. rice both "
  "carry ALUM 331 (MD p.25)", "high",
  resolve="probability surfaces 331_5_W_CER vs 331_6_S_CER_EX_RICE (MD p.25) or ABS census winter/summer cereals")
m("3.3.3", [8], 0, "stated",
  "Hay and silage (HB p.10); mapping unit 13 'Pastures and crops for hay and silage' = ALUM 333 (MD p.25)", "high")
m("3.3.4", [27, 20], 0, "stated",
  "Oilseeds (HB p.10); mapping units 10 Winter oilseeds and 11 Summer oilseeds = ALUM 334 (MD p.25)", "high",
  resolve="probability surfaces 334_10 vs 334_11 or ABS census")
m("3.3.5", [17], 0, "stated", "Sugar cane (HB p.10); mapping unit 12 Sugar cane = ALUM 335 (MD p.25)", "high")
m("3.3.6", [4], 0, "stated", "Dryland cotton (HB p.10); mapping unit 14 Cotton = ALUM 336 (MD p.25)", "high")
m("3.3.8", [26, 19], 0, "stated",
  "Pulses, grain legumes (HB p.10); mapping units 8 Winter legumes and 9 Summer legumes = ALUM 338 (MD p.25)",
  "high", resolve="probability surfaces 338_8 vs 338_9 or ABS census")
m("3.4.0", [12], 0, "stated",
  "Perennial horticulture, generic (HB p.11); mapping unit 24 'Plantation fruit' = ALUM 340 (MD p.25), same name "
  "as LUTO lu 12", "medium")
m("3.4.1", [0, 11, 16, 21], 0, "stated",
  "Tree fruits incl. pome, stone and tropical fruit (HB p.11); mapping units 18 Apples, 19 Pears and other pome, "
  "20 Stone fruit excl. tropical, 21 Tropical stone fruit all = ALUM 341 (MD p.25)", "high",
  resolve="probability surfaces 341_18 to 341_21 (MD p.25) or ABS census orchard areas")
m("3.4.3", [9], 0, "stated", "Tree nuts (HB p.11); mapping unit 22 Nuts = ALUM 343 (MD p.25)", "high")
m("3.4.5", [12], 0, "inferred",
  "Shrub berries and fruits, includes plantation fruit such as pineapples (HB p.11); mapping unit 23 'Berry fruit' "
  "= ALUM 345 (MD p.25) has no LUTO lu; nearest by definition is Plantation fruit", "low",
  alt="10 Other non-cereal crops",
  resolve="policy: which LUTO lu absorbs berry fruit (check LUTO commodity/yield tables for berries)", policy="Y")
m("3.4.8", [3], 0, "stated", "Citrus (HB p.11); mapping unit 17 Citrus = ALUM 348 (MD p.25)", "high")
m("3.4.9", [7], 0, "stated", "Grapes (HB p.11); mapping unit 25 Grapes = ALUM 349 (MD p.25)", "high")
m("3.5.4", [24], 0, "stated",
  "Seasonal vegetables and herbs (HB p.12; coded 3.5.3 in v8, 3.5.4 is the v7 code per HB p.58); mapping unit 16 "
  "Vegetables = ALUM 353 (MD p.25)", "high")
# ---- Primary 4 (irrigated)
m("4.1.0 4.1.1 4.1.2 4.1.3", [22], 0, "inferred",
  "Irrigated plantation forests (HB p.13); modified land, no LUTO forestry lu. lm set to 0 because Unallocated is "
  "never irrigated in LUTO (dataprep.py:561-567); irrigation status is dropped", "low", alt="-1",
  resolve="policy: as for 3.1.x, plus whether to keep the irrigation status", policy="Y")
m("4.1.4", [22], 0, "inferred",
  "Irrigated environmental forest plantation incl. carbon credits (HB p.13); as 3.1.4; lm 0 because Unallocated "
  "is never irrigated (dataprep.py:561-567)", "low", alt="-1 or non-ag Environmental Plantings",
  resolve="policy: as for 3.1.4", policy="Y")
m("4.2.0", [1, 14, 5], 1, "stated and inferred",
  "Irrigated modified pasture, >50% exotic, irrigation infrastructure present (HB p.14); cattle meat, cattle dairy "
  "and sheep apply to 4.2.0 (HB p.32-34)", "medium",
  resolve="ABS census livestock by SA2; irrigated dairy regions; LUTO livestock layers")
m("4.3.0", [10], 1, "stated and inferred",
  "Irrigated cropping, generic (HB p.14); ONCC mapping unit = ALUM 330 with irrigation status set outside SPREAD "
  "(MD p.10, p.25)", "medium", alt="any LUTO crop lu", resolve="as 3.3.0")
m("4.3.1", [25, 18], 1, "stated",
  "Irrigated cereals excl. rice (HB p.14-15); units 5 and 6 = ALUM 331, irrigation set by irrigation potential "
  "index (MD p.4, p.10, p.25)", "high", resolve="as 3.3.1")
m("4.3.3", [8], 1, "stated", "Irrigated hay and silage (HB p.15); unit 13 = ALUM 333 (MD p.25)", "high")
m("4.3.4", [27, 20], 1, "stated", "Irrigated oilseeds (HB p.15); units 10/11 = ALUM 334 (MD p.25)", "high",
  resolve="as 3.3.4")
m("4.3.5", [17], 1, "stated", "Irrigated sugar cane (HB p.15); unit 12 = ALUM 335 (MD p.25)", "high")
m("4.3.6", [4], 1, "stated", "Irrigated cotton (HB p.15); unit 14 = ALUM 336 (MD p.25)", "high")
m("4.3.8", [26, 19], 1, "stated", "Irrigated pulses (HB p.15); units 8/9 = ALUM 338 (MD p.25)", "high",
  resolve="as 3.3.8")
m("4.3.9", [13], 1, "stated",
  "Irrigated rice; rice may always be assumed irrigated (HB p.13, p.34); unit 7 Rice = ALUM 439 (MD p.25)", "high")
m("4.4.0", [12], 1, "stated",
  "Irrigated perennial horticulture, generic (HB p.15); Plantation fruit unit = ALUM 340 (MD p.25), irrigated",
  "medium")
m("4.4.1", [0, 11, 16, 21], 1, "stated", "Irrigated tree fruits (HB p.15); units 18-21 = ALUM 341 (MD p.25)",
  "high", resolve="as 3.4.1")
m("4.4.3", [9], 1, "stated", "Irrigated tree nuts (HB p.15); unit 22 = ALUM 343 (MD p.25)", "high")
m("4.4.5", [12], 1, "inferred",
  "Irrigated shrub berries and fruits incl. pineapples (HB p.16); Berry fruit unit has no LUTO lu, nearest by "
  "definition is Plantation fruit", "low", alt="10 Other non-cereal crops", resolve="as 3.4.5", policy="Y")
m("4.4.8", [3], 1, "stated", "Irrigated citrus (HB p.16); unit 17 = ALUM 348 (MD p.25)", "high")
m("4.4.9", [7], 1, "stated", "Irrigated grapes (HB p.16); unit 25 = ALUM 349 (MD p.25)", "high")
m("4.5.3", [24], 1, "stated", "Irrigated seasonal vegetables and herbs (HB p.16); unit 16 = ALUM 353 (MD p.25)",
  "high")
# ---- Primary 5
m("5.0.0 5.3.0 5.3.1 5.3.2 5.3.3 5.3.4 5.3.5 5.3.6 5.3.7 5.3.8 5.4.0 5.4.1 5.4.3 5.4.4 5.5.0 5.5.1 5.5.2 5.5.3 "
  "5.5.4 5.5.5 5.6.0 5.6.1 5.6.2 5.6.3 5.6.4 5.6.5 5.6.6 5.6.7 5.7.0 5.7.1 5.7.2 5.7.3 5.7.4 5.7.5 5.8.0 5.8.1 "
  "5.8.2 5.8.3 5.9.0 5.9.1 5.9.2 5.9.3 5.9.5",
  TO_NONAG, "-", "inferred", "Intensive use: high interference with natural processes (HB p.17); not an ag land use",
  "high")
m("5.1.0 5.1.1 5.1.2 5.1.3 5.1.4", TO_NONAG, "-", "inferred",
  "Intensive horticulture with special-purpose structures (HB p.18); protected cropping is not a LUTO lu",
  "medium", alt="24 Vegetables (irr) for glasshouse vegetables",
  resolve="policy: protected cropping in or out of LUTO", policy="Y")
m("5.1.5 5.2.8", TO_NONAG, "-", "inferred",
  "Abandoned intensive horticulture/animal production, infrastructure still in place (HB p.18-19)", "medium",
  alt="22", resolve="policy: abandoned intensive sites as convertible modified land", policy="Y")
m("5.2.0 5.2.3 5.2.4 5.2.5 5.2.6 5.2.7", TO_NONAG, "-", "inferred",
  "Intensive animal production facilities, excludes associated grazing (HB p.18-19); LUTO pork, poultry and "
  "aquaculture are off-land", "high")
m("5.2.1", TO_NONAG, "-", "stated and inferred",
  "Dairy sheds and yards only; dairy pastures are mapped to 3.2/4.2 (HB p.18, p.30)", "high",
  alt="5 Dairy - modified land (absorb into surrounding pasture)")
m("5.2.2", TO_NONAG, "-", "inferred",
  "Feedlots: infrastructure for intensive feeding (HB p.18); LUTO beef and sheep are grazing land uses", "high")
m("5.4.2", TO_NONAG, "-", "inferred",
  "Rural residential with sub-commercial/hobby agriculture; ag >2 ha is mapped separately to crop/pasture classes "
  "(HB p.20)", "medium", alt="22 or 1;14 (modified-land grazing)",
  resolve="policy: hobby farms in or out of LUTO", policy="Y")
m("5.4.5", TO_NONAG, "-", "inferred", "Farm buildings/infrastructure above minimum mapping unit (HB p.20)", "medium",
  alt="absorb into surrounding ag lu")
m("5.8.4", TO_NONAG, "-", "inferred",
  "Extractive industry not in use, incl. mine rehabilitation; sites rehabilitated for ag go to 3.6.3/4.6.3 "
  "instead (HB p.22)", "medium", alt="22",
  resolve="policy: disused mine land as convertible modified land", policy="Y")
# ---- Primary 6
m("6.0.0 6.1.0 6.1.1 6.1.2 6.1.3 6.1.4 6.2.0 6.2.1 6.2.2 6.2.3 6.3.0 6.3.1 6.3.2 6.3.3 6.4.0 6.4.1 6.4.2 6.4.3 "
  "6.5.0 6.5.1 6.5.3 6.5.4 6.6.0 6.6.1 6.6.2 6.6.3", TO_NONAG, "-", "inferred",
  "Water feature; water classes take precedence over land use (HB p.23)", "high")
m("6.5.2", TO_NONAG, "-", "inferred",
  "Marsh/wetland where the water feature relates to class 2 production uses, e.g. grazed wetland (HB p.24); water "
  "takes precedence (HB p.23)", "medium", alt="2;15 (natural-land grazing) or 23",
  resolve="policy: grazed wetlands in or out of LUTO", policy="Y")

XWALK_COLUMNS = ['TERTV8', 'ALUM_name', 'PRIMV8', 'SECV8', 'luto_lu_codes', 'luto_lu_names', 'lm', 'mapping_type',
                 'provenance', 'rationale', 'confidence', 'policy_choice', 'alt_candidates', 'resolve_by']
# ============================================================================= end of reference tables


def sha(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


# ----------------------------------------------------------------------------- cache (shared, content-addressed)
def cache_path(cache, digest, url):
    """<cache>/<first two hex>/<sha256><extension of the URL's file name> (GDAL needs the .zip to read inside an archive)."""
    return os.path.join(cache, digest[:2], digest + os.path.splitext(url.rsplit('/', 1)[-1])[1])


def cache_index_add(cache, name, url, digest, nbytes):
    p = os.path.join(cache, 'index.csv')
    rows = {}
    if os.path.exists(p):
        with open(p, encoding='utf-8', newline='') as f:
            rows = {(r['sha256'], r['url']): r for r in csv.DictReader(f)}
    rows[(digest, url)] = dict(sha256=digest, url=url, bytes=str(nbytes), name=name)
    with open(p, 'w', encoding='utf-8', newline='\n') as f:
        w = csv.DictWriter(f, fieldnames=['sha256', 'url', 'bytes', 'name'], lineterminator='\n')
        w.writeheader()
        w.writerows(rows[k] for k in sorted(rows))


def quarantine(cache, name, data, digest):
    q = os.path.join(cache, '_quarantine')
    os.makedirs(q, exist_ok=True)
    p = os.path.join(q, f'{name}_{digest}')
    with open(p, 'wb') as f:
        f.write(data)
    return p


def fetch(name, cache, offline, overrides):
    """Path of the pinned raw file `name` in the cache; download it, or take an override copy, on a cache miss.
    Any byte difference from the pin stops the run; a changed download is set aside in _quarantine, never re-pinned."""
    url, pin, nbytes = SOURCES[name]
    p = cache_path(cache, pin, url)
    if os.path.exists(p):
        if sha_file(p) != pin:
            raise SystemExit(f'{name}: cached file {p} does not match its pin {pin}; remove it and rerun')
        return p
    if name in overrides:
        src = overrides[name]
        got = sha_file(src)
        if got != pin:
            raise SystemExit(f'{name}: --source {src} has sha256 {got}, pinned {pin}')
        os.makedirs(os.path.dirname(p), exist_ok=True)
        shutil.copyfile(src, p + '.part')
    else:
        if offline:
            raise SystemExit(f'--offline: {name} ({url}) is not in the cache {cache}')
        last = None
        for attempt in range(5):
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (luto-nlum-map/1.0)'})
                with urllib.request.urlopen(req, timeout=300) as r:
                    data = r.read()
                break
            except Exception as e:      # network retry
                last = e
                time.sleep(5 * (attempt + 1))
        else:
            raise SystemExit(f'{name}: download failed from {url}: {last}; supply a copy with --source {name}=PATH')
        got = sha(data)
        if got != pin:
            q = quarantine(cache, name, data, got)
            raise SystemExit(f'{name}: {url} now serves sha256 {got} ({len(data)} bytes), pinned {pin} ({nbytes} bytes). '
                             f'Download set aside at {q}; the pin is not changed.')
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p + '.part', 'wb') as f:
            f.write(data)
    os.replace(p + '.part', p)
    cache_index_add(cache, name, url, pin, nbytes)
    return p


def vsizip(zpath, member):
    return '/vsizip/' + zpath.replace('\\', '/') + '/' + member


# ----------------------------------------------------------------------------- crosswalk and classes
def class_tables(zips):
    """ALUMV8 raster attribute tables (raw bytes), 2010-11 then 2020-21. The 2015-16 table adds no value and changes no
    attribute of the classes read here (checked against all three tables), so it is not read; its raster is allocated
    with these classes."""
    ts = []
    for y in YEARS:
        with zipfile.ZipFile(zips[f'alum_{y}']) as z:
            ts.append(z.read(f'NLUM_v7_1_250m_ALUMV8_{y}_alb.csv'))
    return ts


def xwalk_rows(tables):
    attrs = {}
    for t in tables:
        for r in csv.DictReader(io.StringIO(t.decode('utf-8'))):
            attrs.setdefault(r['TERTV8'].split(' ')[0], (r['SECV8'], r['PRIMV8']))
    assert len(ALUM_NAMES) == 146 and set(XWALK) == set(ALUM_NAMES), set(XWALK) ^ set(ALUM_NAMES)
    users = {}
    for c, (lus, lm, *_) in XWALK.items():
        if lus != TO_NONAG and len(lus) == 1:
            users[(lus[0], lm)] = users.get((lus[0], lm), 0) + 1
    rows = []
    for code, name in ALUM_NAMES.items():
        lus, lm, prov, why, conf, alt, resolve, policy = XWALK[code]
        if lus == TO_NONAG:
            mt = 'to-nonag'
        elif len(lus) > 1:
            mt = 'one-to-many'
        elif users[(lus[0], lm)] > 1:
            mt = 'many-to-one member'
        else:
            mt = 'one-to-one'
        sec, prim = attrs[code]
        rows.append({'TERTV8': code, 'ALUM_name': name, 'PRIMV8': prim, 'SECV8': sec,
                     'luto_lu_codes': ';'.join(map(str, lus)),
                     'luto_lu_names': 'Non-agricultural / outside LUTO' if lus == TO_NONAG else ';'.join(LU_NAMES[i] for i in lus),
                     'lm': lm, 'mapping_type': mt, 'provenance': prov, 'rationale': why, 'confidence': conf,
                     'policy_choice': policy, 'alt_candidates': alt, 'resolve_by': resolve})
    return rows


def write_xwalk(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=XWALK_COLUMNS)
        w.writeheader()
        w.writerows(rows)


def build_classes(tables, xw_path):
    """(CLS frame by class id with lu, lm, name), LUT[raster value] -> class id (-1 for split values),
    SPLIT_CLS[split id] -> class ids of the candidates, CT (class table by raster value)."""
    ct = pd.concat([pd.read_csv(io.BytesIO(t)) for t in tables]).drop_duplicates('Value').drop(columns='Count')
    ct = ct.set_index('Value').sort_index()
    D = pd.read_csv(xw_path, dtype=str)
    val = {str(t).split(' ')[0]: int(v) for v, t in ct.TERTV8.items() if v}
    pairs, vmap = set(), {}
    for _, r in D.iterrows():
        v = val.get(r.TERTV8)
        if v is None:
            continue                                               # class absent from the raster tables
        cs = [int(c) for c in str(r.luto_lu_codes).split(';')]
        lm = 0 if r.lm == '-' else int(r.lm)
        if v in (125, 133):
            vmap[v] = ('NONAG',)                                   # 1.2.5 / 1.3.3 -> -1
        elif v == 210:
            vmap[v] = ('GNAT',)
        elif v == 320:
            vmap[v] = ('GMOD0',)
        elif v == 420:
            vmap[v] = ('GMOD1',)
        elif v in SPLIT:
            assert sorted(cs) == sorted(c for c, _ in SPLIT[v]), (v, cs)
            vmap[v] = ('SPLIT',)
            pairs |= {(c, 1 if v >= 400 else 0) for c, _ in SPLIT[v]}
        else:
            assert len(cs) == 1, (v, cs)
            if cs[0] == -1:
                vmap[v] = ('NONAG',)
            else:
                vmap[v] = ('PAIR', cs[0], lm)
                pairs.add((cs[0], lm))
    rows = [dict(cls=0, key='NONAG', lu=-1, lm=0, name='Non-ag / outside (-1)'),
            dict(cls=1, key='GNAT', lu=100, lm=0, name='Grazing natural (2.1.0), dry'),
            dict(cls=2, key='GMOD0', lu=101, lm=0, name='Grazing modified (3.2.0), dry'),
            dict(cls=3, key='GMOD1', lu=101, lm=1, name='Grazing modified (4.2.0), irr')]
    for lu, lm in sorted(pairs, key=lambda p: (p[1], p[0])):
        rows.append(dict(cls=len(rows), key=f'{lu}|{lm}', lu=lu, lm=lm, name=f"{LU_NAMES[lu]} | {'irr' if lm else 'dry'}"))
    CLS = pd.DataFrame(rows).set_index('cls')
    cid = {(r.lu, r.lm): c for c, r in CLS.iterrows()}
    LUT = np.full(1000, -9, np.int16)
    for v, mm in vmap.items():
        LUT[v] = {'NONAG': NONAG, 'GNAT': GNAT, 'GMOD0': GMOD0, 'GMOD1': GMOD1}.get(mm[0], -1) if mm[0] != 'PAIR' else cid[(mm[1], mm[2])]
    SPLIT_CLS = [[cid[(c, 1 if v >= 400 else 0)] for c, _ in SPLIT[v]] for v in SPLIT_IDS]
    return CLS, LUT, SPLIT_CLS, ct


# ----------------------------------------------------------------------------- step 2: pixels to cells (worker pool)
_G = {}


def _init(alum, probs, mask_path, LUT, SPLIT_CLS, NC):
    with rasterio.open(mask_path) as r:
        mk = r.read(1) > 0
        LT = r.transform
    cof = np.full(mk.size, -1, np.int64)
    cof[np.flatnonzero(mk.ravel())] = np.arange(int(mk.sum()))
    _G.update(LUT=LUT, SPLIT_CLS=SPLIT_CLS, NCLS=NC, cof=cof, LT=LT, LH=mk.shape[0], LW=mk.shape[1],
              A=rasterio.open(alum), P={u: rasterio.open(p) for u, p in probs.items()},
              tr=pyproj.Transformer.from_crs(3577, 4283, always_xy=True))


def _block(r0):
    g = _G
    A = g['A']
    AT = A.transform
    H, W = A.shape
    n = min(STEP, H - r0)
    a = A.read(1, window=Window(0, r0, W, n))
    rr, cc = np.nonzero(a != 0)
    out = dict(r0=r0, npix=int(rr.size))
    if rr.size == 0:
        return out
    v = a[rr, cc].astype(np.int64)
    x = AT.c + (cc + 0.5) * AT.a
    y = AT.f + (r0 + rr + 0.5) * AT.e
    lon, lat = g['tr'].transform(x, y)
    LT = g['LT']
    lc = np.floor((lon - LT.c) / LT.a).astype(np.int64)
    lr = np.floor((lat - LT.f) / LT.e).astype(np.int64)
    inb = (lc >= 0) & (lc < g['LW']) & (lr >= 0) & (lr < g['LH'])
    cell = np.full(v.size, -1, np.int64)
    cell[inb] = g['cof'][lr[inb] * g['LW'] + lc[inb]]
    cls = g['LUT'][v].astype(np.int64)
    assert (cls != -9).all(), np.unique(v[cls == -9])
    NC = g['NCLS']
    keys = [(cell[cls >= 0] + 1) * NC + cls[cls >= 0]]
    wts = [np.ones(int((cls >= 0).sum()))]
    npk, npkind = [], []
    for sid, sv in enumerate(SPLIT_IDS):
        k = np.flatnonzero(v == sv)
        if k.size == 0:
            continue
        cands = SPLIT[sv]
        PT = g['P'][cands[0][1]].transform
        PH, PW = g['P'][cands[0][1]].shape
        pr = np.floor((lat[k] - PT.f) / PT.e).astype(np.int64)
        pc = np.floor((lon[k] - PT.c) / PT.a).astype(np.int64)
        ok = (pr >= 0) & (pr < PH) & (pc >= 0) & (pc < PW)
        V = np.full((len(cands), k.size), np.nan)
        if ok.any():
            r_lo, r_hi = pr[ok].min(), pr[ok].max() + 1
            c_lo, c_hi = pc[ok].min(), pc[ok].max() + 1
            for j, (_, u) in enumerate(cands):
                b = g['P'][u].read(1, window=Window(c_lo, r_lo, c_hi - c_lo, r_hi - r_lo)).astype(np.float64)
                q = np.full(k.size, np.nan)
                q[ok] = b[pr[ok] - r_lo, pc[ok] - c_lo]
                q[q == 32767] = np.nan
                V[j] = q
        anyv = ~np.isnan(V).all(0)
        S = np.nansum(V, 0)
        good = anyv & (S > 0)
        F = np.where(good, np.nan_to_num(V) / np.where(good, S, 1.), 0.)    # candidate fractions, sum to 1 where good
        for j, cid in enumerate(g['SPLIT_CLS'][sid]):
            keys.append((cell[k[good]] + 1) * NC + cid)
            wts.append(F[j, good])
        bad = k[~good]
        npk.append((cell[bad] + 1) * len(SPLIT_IDS) + sid)
        npkind.append(np.where(anyv[~good], 1, 0))                          # 1 = all candidates 0, 0 = no valid value
    kk = np.concatenate(keys)
    ww = np.concatenate(wts)
    u, inv = np.unique(kk, return_inverse=True)
    out['E_key'] = u
    out['E_w'] = np.bincount(inv, weights=ww)
    if npk:
        nk = np.concatenate(npk) * 2 + np.concatenate(npkind)
        u, c = np.unique(nk, return_counts=True)
        out['NP_key'] = u
        out['NP_n'] = c
    u, c = np.unique(cell, return_counts=True)
    out['cell_u'] = u
    out['cell_n'] = c
    u, c = np.unique((cell + 1) * 1000 + v, return_counts=True)             # pixel counts per (cell, raster value)
    out['cv_u'] = u
    out['cv_n'] = c
    return out


def pixels(Y, zips, mask_path, LUT, SPLIT_CLS, NC, workers):
    alum = vsizip(zips[f'alum_{Y}'], f'NLUM_v7_1_250m_ALUMV8_{Y}_alb.tif')
    with zipfile.ZipFile(zips[f'prob_{Y}']) as z:
        names = z.namelist()
    probs = {}
    for sv in SPLIT_IDS:
        for _, u in SPLIT[sv]:
            # NLUM_v7_1_probSurf_<yyyy>_<yy>_<alum>_<unit>_<name>.tif
            f = [x for x in names if x.startswith(f'NLUM_v7_1_probSurf_{Y}_') and x.endswith('.tif') and x.split('_')[7] == str(u)]
            assert len(f) == 1, (Y, u, f)
            probs[u] = vsizip(zips[f'prob_{Y}'], f[0])
    with rasterio.open(alum) as A:
        H = A.height
    with Pool(workers, initializer=_init, initargs=(alum, probs, mask_path, LUT, SPLIT_CLS, NC)) as pool:
        res = pool.map(_block, range(0, H, STEP), chunksize=1)

    def merge(kk, ww):
        kk = [x for x in kk if x is not None]
        ww = [x for x in ww if x is not None]
        u, inv = np.unique(np.concatenate(kk), return_inverse=True)
        return u, np.bincount(inv, weights=np.concatenate(ww).astype(np.float64))
    Ek, Ew = merge([r.get('E_key') for r in res], [r.get('E_w') for r in res])
    Nk, Nn = merge([r.get('NP_key') for r in res], [r.get('NP_n') for r in res])
    Cu, Cn = merge([r.get('cell_u') for r in res], [r.get('cell_n') for r in res])
    Vk, Vn = merge([r.get('cv_u') for r in res], [r.get('cv_n') for r in res])
    npix = sum(r['npix'] for r in res)
    z = dict(E_cell=(Ek // NC - 1).astype(np.int64), E_cls=(Ek % NC).astype(np.int16), E_w=Ew,
             NP_cell=((Nk // 2) // len(SPLIT_IDS) - 1).astype(np.int64), NP_sid=((Nk // 2) % len(SPLIT_IDS)).astype(np.int16),
             NP_kind=(Nk % 2).astype(np.int8), NP_n=Nn, cell_u=Cu.astype(np.int64), cell_n=Cn,
             cv_cell=(Vk // 1000 - 1).astype(np.int64), cv_val=(Vk % 1000).astype(np.int64), cv_n=Vn)
    closure = npix - Ew.sum() - Nn.sum()
    assert abs(closure) < 1e-3, closure
    return z, dict(pixels=int(npix), expected_pixels=float(Ew.sum()), noprob_pixels=float(Nn.sum()))


# ----------------------------------------------------------------------------- step 3: SA2 per cell
def sa2_cells(shp_zip, mask_path):
    """SA2_2021 code per LUTO cell: polygon at the cell centre (rasterize, all_touched=False), else the nearest labelled
    pixel (Euclidean distance transform over all labelled pixels, inside or outside the land mask)."""
    import geopandas as gpd
    from rasterio.features import rasterize
    from scipy.ndimage import distance_transform_edt
    with rasterio.open(mask_path) as r:
        mk = r.read(1) == 1
        T = r.transform
    g = gpd.read_file('zip://' + shp_zip.replace('\\', '/') + '!SA2_2021_AUST_GDA94.shp')
    assert str(g.crs) == 'EPSG:4283', g.crs
    bad = ~g['SA2_CODE21'].str.fullmatch(r'[0-9]{9}')
    g = g[~bad].copy()
    g['code'] = g['SA2_CODE21'].astype(np.int64)
    gg = g[g.geometry.notna() & ~g.geometry.is_empty]
    ras = rasterize(((geom, int(c)) for geom, c in zip(gg.geometry, gg.code)), out_shape=mk.shape,
                    transform=T, fill=-1, all_touched=False, dtype='int32')
    cells = ras[mk].astype(np.int32)
    idx = distance_transform_edt(ras == -1, return_distances=False, return_indices=True)
    near = ras[idx[0], idx[1]]
    fill = cells.copy()
    miss = cells == -1
    fill[miss] = near[mk][miss]
    return fill


def fix_state_ties(sa2, ST, mask_path):
    """A cell whose SA2 state digit disagrees with state_id takes the SA2 of the nearest cell (grid distance)
    whose SA2 digit equals that state_id and agrees with its own state_id."""
    from scipy.spatial import cKDTree
    with rasterio.open(mask_path) as r:
        mk = r.read(1) == 1
    rows, cols = np.nonzero(mk)
    dig = sa2 // 10**8
    bad = np.where(dig != ST)[0]
    out = sa2.copy()
    for s in np.unique(ST[bad]):
        good = np.where((dig == s) & (ST == s))[0]
        b = bad[ST[bad] == s]
        if len(good) == 0:
            continue
        tree = cKDTree(np.c_[rows[good], cols[good]])
        d, j = tree.query(np.c_[rows[b], cols[b]])
        out[b] = sa2[good[j]]
    return out, bad


# ----------------------------------------------------------------------------- step 4: shares and allocation
def shares(z, SPLIT_CLS, NC, sa2, STATE, FP):
    """Expected pixels per (cell, class), with the no-probability pixels distributed by the class's split in the cell's
    SA2, else state, else nationally. Returns (idx of footprint cells with data, shares S, has-data flag, E)."""
    N = NCELL
    E = np.zeros((N + 1, NC))                                       # row N = pixels outside every land cell
    ci = np.where(z['E_cell'] < 0, N, z['E_cell'])
    np.add.at(E, (ci, z['E_cls'].astype(np.int64)), z['E_w'])
    NPc = np.where(z['NP_cell'] < 0, N, z['NP_cell'])
    NPs = z['NP_sid'].astype(int)
    NPn = z['NP_n']
    s_of = np.r_[sa2, -1]
    st_of = np.r_[STATE, -1]
    for sid, cands in enumerate(SPLIT_CLS):
        sel = NPs == sid
        if not sel.any():
            continue
        cells = NPc[sel]
        n = NPn[sel]
        df = pd.DataFrame(E[:, cands], columns=range(len(cands)))
        df['sa2'] = s_of
        df['st'] = st_of
        g_sa2 = df.groupby('sa2')[list(range(len(cands)))].sum()
        g_st = df.groupby('st')[list(range(len(cands)))].sum()
        g_nat = df[list(range(len(cands)))].sum()
        F = np.zeros((cells.size, len(cands)))
        for i, c in enumerate(cells):
            for tab, key in ((g_sa2, s_of[c]), (g_st, st_of[c])):
                if key in tab.index and tab.loc[key].sum() > 0:
                    F[i] = tab.loc[key].to_numpy() / tab.loc[key].sum()
                    break
            else:
                F[i] = g_nat.to_numpy() / g_nat.sum()
        for j, cid in enumerate(cands):
            np.add.at(E, (cells, cid), n * F[:, j])
    npx = E[:N].sum(1)
    has = npx > 0
    S = np.zeros((N, NC))
    S[has] = E[:N][has] / npx[has, None]
    idx = np.flatnonzero(FP & has)
    return idx, S[idx].astype(np.float64), has


def allocate(grp, area, S, NC):
    """Whole-cell allocation within each group: floor quota in descending share order (ties: lower cell index, then lower
    class id), then largest remainder (class with the largest deficit, ties lower class id, takes the left cell where its
    share is highest, ties lower cell index)."""
    n = len(grp)
    gu, g = np.unique(grp, return_inverse=True)
    G = len(gu)
    T = np.zeros((G, NC))
    np.add.at(T, g, area[:, None] * S)
    ii, cc = np.nonzero(S > 0)
    ss = S[ii, cc]
    o = np.lexsort((cc, ii, -ss, g[ii]))
    ii, cc, gg = ii[o].tolist(), cc[o].tolist(), g[ii[o]].tolist()
    al = np.zeros((G, NC))
    out = np.full(n, -1, np.int64)
    Tl = (T * (1 + 1e-12) + 1e-9).tolist()
    all_ = al.tolist()
    a = area.tolist()
    outl = out.tolist()
    for i, c, q in zip(ii, cc, gg):
        if outl[i] >= 0:
            continue
        if all_[q][c] + a[i] <= Tl[q][c]:
            outl[i] = c
            all_[q][c] += a[i]
    out = np.array(outl, np.int64)
    al = np.array(all_)
    left = np.flatnonzero(out < 0)
    lg = g[left]
    o = np.argsort(lg, kind='stable')
    left, lg = left[o], lg[o]
    starts = np.r_[0, np.flatnonzero(np.diff(lg)) + 1, len(lg)]
    for s0, s1 in zip(starts[:-1], starts[1:]):
        q = lg[s0]
        cells = list(left[s0:s1])
        d = T[q] - al[q]
        while cells:
            c = int(np.argmax(d))
            sv = S[cells, c]
            j = int(np.argmax(sv))                                  # first max = lowest cell index (cells ascending)
            i = cells.pop(j)
            out[i] = c
            d[c] -= area[i]
            al[q, c] += area[i]
    assert (out >= 0).all()
    return out.astype(np.int16)


# ----------------------------------------------------------------------------- step 5: map rules
def natural_value(CT, v):
    t = str(CT.loc[v, 'TERTV8']).split(' ')[0]
    p = int(CT.loc[v, 'PRIMV8N'])
    if p in (1, 2):
        return True
    if p == 6:
        return t[:3] in ('6.0', '6.1', '6.3', '6.5', '6.6')       # natural water classes
    return False


def retype_held_crop(z, held, lu, lm, cat, lu0, LUT, CT):
    """Held cells whose 2010 lumap is a crop -> 23 (natural v7.1 class) or 22 (modified or intensive), lm 0, cat 53.
    The class is the cell's dominant v7.1 value among its pixels that map to -1, else its dominant value overall."""
    H = held & (lu0 >= 0) & ~np.isin(lu0, [1, 2, 5, 6, 14, 15, 22, 23])
    cell, v, n = z['cv_cell'], z['cv_val'], z['cv_n']
    sel = (cell >= 0) & H[np.maximum(cell, 0)]
    d = pd.DataFrame({'c': cell[sel], 'v': v[sel], 'n': n[sel], 'non': LUT[v[sel]] == NONAG})
    d = d.sort_values(['c', 'non', 'n', 'v'], ascending=[True, False, False, True]).drop_duplicates('c')
    dom = pd.Series(d.v.to_numpy(), index=d.c.to_numpy())
    idx = np.flatnonzero(H)
    dv = dom.reindex(idx).to_numpy()
    nat = np.array([natural_value(CT, int(x)) if np.isfinite(x) else False for x in dv])
    lu[idx] = np.where(nat, 23, 22)
    lm[idx] = 0
    cat[idx] = 53


def build_map(z, has, idx, c, lu0, lm0, FP, CLS, SPLIT_CLS, LUT, CT, cls10_full=None):
    N = NCELL
    cfull = np.full(N, -1, np.int16)
    cfull[idx] = c
    ok = z['E_cell'] >= 0
    tot = np.bincount(z['E_cell'][ok], weights=z['E_w'][ok], minlength=N)
    nn = ok & (z['E_cls'] == NONAG)
    non = np.bincount(z['E_cell'][nn], weights=z['E_w'][nn], minlength=N)
    lu = np.full(N, -1, np.int16)
    lm = np.zeros(N, np.int8)
    cat = np.zeros(N, np.int8)
    out = ~FP
    cat[out & has & (tot - non > non)] = 50
    cat[out & ~has] = 3
    held = FP & ((cfull == NONAG) | (cfull < 0))
    lu[held] = lu0[held]
    lm[held] = lm0[held]
    cat[held] = 51
    pair = FP & (cfull >= 4)
    splitc = sorted({x for s in SPLIT_CLS for x in s})
    lu[pair] = CLS.lu.to_numpy()[cfull[pair]]
    lm[pair] = CLS.lm.to_numpy()[cfull[pair]]
    cat[pair] = np.where(np.isin(cfull[pair], splitc), 30, 0)
    for S_, keep, conv, ck, ct, cu, cn, ulu in (([GNAT], {2, 6, 15, 23}, NATC, 10, 11, 12, 13, 23),
                                                 ([GMOD0, GMOD1], {1, 5, 14}, MODC, 20, 21, 22, 23, 22)):
        here = FP & np.isin(cfull, S_)
        if cls10_full is None:
            base = here
            new = np.zeros(N, bool)
        else:
            base = here & np.isin(cls10_full, S_)
            new = here & ~base
        k = base & np.isin(lu0, list(keep))
        t = base & np.isin(lu0, list(conv))
        u = base & ~k & ~t
        lu[k] = lu0[k]
        cat[k] = ck
        if t.any():
            lu[t] = pd.Series(lu0[t]).map(conv).to_numpy()
        cat[t] = ct
        lu[u] = ulu
        cat[u] = cu
        lu[new] = ulu
        cat[new] = cn
        if S_ == [GNAT]:
            lm[k] = lm0[k]
            lm[t | u | new] = 0
        else:
            lm[here] = (cfull[here] == GMOD1).astype(np.int8)
    retype_held_crop(z, held, lu, lm, cat, lu0, LUT, CT)
    assert (lu[FP] >= 0).all() and (lu[~FP] == -1).all() and (lm[~FP] == 0).all()
    return cfull, lu.astype(np.int16), lm.astype(np.int8), cat.astype(np.int8)


# ----------------------------------------------------------------------------- provenance
def pkg_versions():
    out = {}
    for p in ('numpy', 'pandas', 'tables', 'rasterio', 'pyproj', 'geopandas', 'scipy', 'shapely', 'pyogrio', 'fiona'):
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    out['gdal'] = rasterio.__gdal_version__
    out['proj'] = pyproj.proj_version_str
    return out


def write_csv(path, header, rows):
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(header)
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--input', required=True, help='folder of the inherited 2010-build files listed in INHERITED (input/Nick_build/inherited_2010)')
    ap.add_argument('--out', required=True, help='output folder (e.g. input/Nick_build/nlum_map)')
    ap.add_argument('--cache', required=True, help='shared raw-download cache (content-addressed by sha256)')
    ap.add_argument('--offline', action='store_true', help='fail instead of downloading on a cache miss')
    ap.add_argument('--source', action='append', default=[], metavar='NAME=PATH',
                    help='a local copy of a pinned source (must match its pin); NAME in ' + ', '.join(SOURCES))
    ap.add_argument('--workers', type=int, default=min(48, os.cpu_count() or 1), help='processes for the pixel pass')
    a = ap.parse_args()
    t0 = time.time()
    overrides = dict(s.split('=', 1) for s in a.source)
    assert set(overrides) <= set(SOURCES), set(overrides) - set(SOURCES)
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.cache, exist_ok=True)
    zips = {k: fetch(k, a.cache, a.offline, overrides) for k in SOURCES}
    inp = {f: os.path.join(a.input, f) for f in INHERITED}
    inp_sha = {f: sha_file(p) for f, p in inp.items()}

    tables = class_tables(zips)
    xw_path = os.path.join(a.out, 'xwalk_definition.csv')
    write_xwalk(xw_path, xwalk_rows(tables))
    CLS, LUT, SPLIT_CLS, CT = build_classes(tables, xw_path)
    NC = len(CLS)

    A = pd.read_hdf(inp['real_area.h5']).to_numpy().astype(np.float64)
    lu0 = pd.read_hdf(inp['lumap.h5']).to_numpy().astype(np.int64)
    lm0 = pd.read_hdf(inp['lmmap.h5']).to_numpy().astype(np.int64)
    ST = np.load(inp['state_id.npy']).astype(np.int64)
    assert len(A) == len(lu0) == len(lm0) == len(ST) == NCELL
    FP = lu0 >= 0                                                  # footprint fixed at the 2010 map

    sa2_raw = sa2_cells(zips['sa2_2021'], inp['NLUM_2010-11_mask.tif'])
    np.save(os.path.join(a.out, 'sa2_2021_cells.npy'), sa2_raw)
    sa2, tie_cells = fix_state_ties(sa2_raw.astype(np.int64), ST, inp['NLUM_2010-11_mask.tif'])
    STATE = sa2 // 10**8

    info = {}
    c10 = None
    for Y in ALLOC_YEARS:
        z, pinfo = pixels(Y, zips, inp['NLUM_2010-11_mask.tif'], LUT, SPLIT_CLS, NC, a.workers)
        idx, S, has = shares(z, SPLIT_CLS, NC, sa2, STATE, FP)
        c = allocate(sa2[idx], A[idx], S, NC)
        cfull, lu, lm, cat = build_map(z, has, idx, c, lu0, lm0, FP, CLS, SPLIT_CLS, LUT, CT, c10)
        if Y == '2010_11':
            c10 = cfull                                            # grazing persistence of the later years against 2010-11
        for k, arr in (('lumap', lu), ('lmmap', lm), ('category', cat)):
            np.save(os.path.join(a.out, f'class_alloc_{Y}_{k}.npy'), arr)
        info[Y] = dict(pinfo, footprint_cells_with_data=int(len(idx)))
        print(Y, json.dumps(info[Y]), f'{time.time() - t0:.0f}s', flush=True)

    # ---- provenance (last; hashes the other outputs)
    me = open(os.path.abspath(__file__), 'rb').read()
    P = [['source', k, url, pin, f'sha256 of raw file, {nb} bytes', SOURCE_NOTES[k]] for k, (url, pin, nb) in SOURCES.items()]
    P += [['inherited', f, '', inp_sha[f], 'inherited 2010-build input (inherited_2010)', ''] for f in INHERITED]
    P += [['script', 'input/Nick_build/scripts/nlum_map.py', '', sha(me), 'sha256 of script', '']]
    P += [['package', k, '', v, 'version', ''] for k, v in sorted(pkg_versions().items())]
    P += [['python', sys.version.split()[0], '', '', 'interpreter', '']]
    outs = sorted(f for f in os.listdir(a.out) if f != 'provenance.csv' and not f.startswith('.'))
    P += [['output', f, '', sha_file(os.path.join(a.out, f)), 'sha256', ''] for f in outs]
    write_csv(os.path.join(a.out, 'provenance.csv'), ['kind', 'item', 'url', 'sha256', 'note', 'source'], P)
    print('done', f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
