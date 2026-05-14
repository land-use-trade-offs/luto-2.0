# Skill: Using `fakedata` to Inspect and Prototype with `data.py` Objects

`fakedata` is a lightweight stand-in for the full `luto.data.Data` object. It loads only the spatial masking and geo-metadata needed to use LUTO helpers (e.g. `arr_to_xr()`), so you can inspect arrays and prototype code without loading gigabytes of simulation inputs.

> **Note:** `fakedata` lives in `jinzhu_inspect_code/fakeData.py` which is not tracked by git. Always copy the full class code below when setting it up in a new notebook or script.

---

## When to Use

- You want to call `arr_to_xr()` or other helpers in `luto/tools/Manual_jupyter_books/helpers/` without a full simulation run.
- You are inspecting or debugging a 1D numpy array in spatial context (plotting, reprojecting, etc.).
- You are prototyping new code that takes a `data` argument and need a quick `data` object to test with.

---

## Full Class Code

Copy this into your notebook or script:

```python
import os
import xarray as xr
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import geopandas as gpd
import netCDF4  # necessary for running luto in Denethor

from luto import tools
import luto.settings as settings
import luto.economics.agricultural.quantity as ag_quantity
import luto.economics.non_agricultural.quantity as non_ag_quantity
import luto.economics.agricultural.water as ag_water
from luto.tools.Manual_jupyter_books.helpers import arr_to_xr

from collections import defaultdict
from typing import Any, Literal, Optional
from affine import Affine
from scipy.interpolate import interp1d
from math import ceil
from dataclasses import dataclass
from scipy.ndimage import distance_transform_edt, maximum_filter


class fakedata():
    """
    Sets up output containers (lumaps, lmmaps, etc) and loads all LUTO data, adjusted
    for resfactor.
    """

    def __init__(self):

        # Path for write module - overwrite when provided with a base and target year
        self.path = None

        # The latest simulation year;
        #   For simulation between 2010-2050, if the run stops at 2030, then it will be 2030
        #   The last_year is updated in the solve_timeseries.simulation module
        self.last_year = None

        # Setup output containers
        self.lumaps = {}
        self.lmmaps = {}
        self.ammaps = {}
        self.ag_dvars = {}
        self.non_ag_dvars = {}
        self.ag_man_dvars = {}
        self.prod_data = {}
        self.obj_vals = {}

        print('')
        print(f'Data Initialization at RES{settings.RESFACTOR}')

        self.YR_CAL_BASE = 2010  # The base year, i.e. where year index yr_idx == 0.

        ###############################################################
        # Masking and spatial coarse graining.
        ###############################################################
        print("├── Setting up masking and spatial course graining data", flush=True)

        # Set resfactor multiplier
        self.RESMULT = settings.RESFACTOR ** 2

        # Set the nodata and non-ag code
        self.NODATA = -9999
        self.MASK_LU_CODE = -1

        # Load LUMAP without resfactor
        self.LUMAP_NO_RESFACTOR = pd.read_hdf(os.path.join(settings.INPUT_DIR, "lumap.h5")).to_numpy().astype(np.int8)

        # NLUM mask.
        with rasterio.open(os.path.join(settings.INPUT_DIR, "NLUM_2010-11_mask.tif")) as rst:
            self.NLUM_MASK = rst.read(1).astype(np.int8)
            self.LUMAP_2D_FULLRES = np.full_like(self.NLUM_MASK, self.NODATA, dtype=np.int16)
            np.place(self.LUMAP_2D_FULLRES, self.NLUM_MASK == 1, self.LUMAP_NO_RESFACTOR)
            self.GEO_META_FULLRES = rst.meta
            self.GEO_META_FULLRES['dtype'] = 'float32'
            self.GEO_META_FULLRES['nodata'] = self.NODATA

        # Mask out non-agricultural, non-environmental plantings land (i.e., -1) from lumap
        self.LUMASK = self.LUMAP_NO_RESFACTOR != self.MASK_LU_CODE
        self.LUMASK_2D_FULLRES = np.nan_to_num(arr_to_xr(self, self.LUMASK))

        # Return combined land-use and resfactor mask
        if settings.RESFACTOR > 1:

            rf_mask = self.NLUM_MASK.copy()
            have_lu_cells = maximum_filter(self.LUMASK_2D_FULLRES, size=settings.RESFACTOR)
            have_lu_cell_downsampled = have_lu_cells[settings.RESFACTOR//2::settings.RESFACTOR, settings.RESFACTOR//2::settings.RESFACTOR]

            lu_mask_fullres = np.zeros_like(rf_mask, dtype=bool)
            lu_mask_fullres[settings.RESFACTOR//2::settings.RESFACTOR, settings.RESFACTOR//2::settings.RESFACTOR] = have_lu_cell_downsampled

            self.COORD_ROW_COL_FULLRES = np.argwhere(rf_mask & lu_mask_fullres).T
            self.COORD_ROW_COL_RESFACTORED = (self.COORD_ROW_COL_FULLRES - (settings.RESFACTOR//2)) // settings.RESFACTOR

            rf_mask[self.COORD_ROW_COL_FULLRES[0], self.COORD_ROW_COL_FULLRES[1]] = 2
            self.MASK = np.where(rf_mask[np.nonzero(self.NLUM_MASK)] == 2, True, False)

            self.LUMAP_2D_RESFACTORED = self.LUMAP_2D_FULLRES[settings.RESFACTOR//2::settings.RESFACTOR, settings.RESFACTOR//2::settings.RESFACTOR]
            self.GEO_META = self.update_geo_meta()

        elif settings.RESFACTOR == 1:
            self.MASK = self.LUMASK
            self.GEO_META = self.GEO_META_FULLRES
            self.LUMAP_2D_RESFACTORED = self.LUMAP_2D_FULLRES
            self.COORD_ROW_COL_FULLRES = np.argwhere(self.NLUM_MASK == 1).T
            self.COORD_ROW_COL_RESFACTORED = self.COORD_ROW_COL_FULLRES
        else:
            raise KeyError("Resfactor setting invalid")

        print(f"│   - Total land cells in full resolution: {self.NLUM_MASK.sum()}")
        print(f"│   - Total land cells in resfactored resolution: {self.MASK.sum()}")

    def update_geo_meta(self):
        meta = self.GEO_META_FULLRES.copy()
        height, width = self.LUMAP_2D_RESFACTORED.shape
        trans = list(self.GEO_META_FULLRES['transform'])
        trans[0] = trans[0] * settings.RESFACTOR
        trans[4] = trans[4] * settings.RESFACTOR
        trans = Affine(*trans)
        meta.update(width=width, height=height, compress='lzw', driver='GTiff', transform=trans, nodata=self.NODATA, dtype='float32')
        return meta
```

---

## Basic Usage

```python
data = fakedata()

# Convert any 1D array to a georeferenced xarray DataArray
from luto.tools.Manual_jupyter_books.helpers import arr_to_xr
da = arr_to_xr(data, my_1d_array)
da.plot()
```

---

## What `fakedata` Provides

| Attribute | Description |
|---|---|
| `NLUM_MASK` | 2D raster mask (0=ocean, 1=land) |
| `LUMASK` | 1D boolean mask — `True` for valid land-use cells |
| `LUMAP_NO_RESFACTOR` | 1D land-use map at full resolution |
| `LUMAP_2D_FULLRES` | 2D land-use map at full resolution |
| `LUMAP_2D_RESFACTORED` | 2D land-use map at resfactored resolution |
| `MASK` | 1D boolean mask at resfactored resolution |
| `GEO_META_FULLRES` | Rasterio geo-metadata at full resolution |
| `GEO_META` | Rasterio geo-metadata adjusted for `RESFACTOR` |
| `COORD_ROW_COL_FULLRES` | (row, col) coords of valid cells at full resolution |
| `COORD_ROW_COL_RESFACTORED` | (row, col) coords of valid cells at resfactored resolution |
| `NODATA` | Nodata value (`-9999`) |
| `RESMULT` | `RESFACTOR ** 2` |
| `YR_CAL_BASE` | Base year (`2010`) |
| `lumaps`, `lmmaps`, `ammaps`, `ag_dvars`, `non_ag_dvars`, `ag_man_dvars`, `prod_data`, `obj_vals` | Empty output containers (same as `Data`) |

---

## Array Size Conventions

`arr_to_xr()` accepts three array sizes and routes them automatically:

| Array size | Meaning | Branch in `arr_to_xr` |
|---|---|---|
| `data.LUMASK.size` | Full-res all-land 1D | Uses `GEO_META_FULLRES`, fills via `NLUM_MASK` |
| `data.LUMASK.sum()` | Full-res valid-LU 1D | Same as above, zero-fills non-LU cells first |
| anything else | Resfactored 1D | Uses `data.GEO_META`, places into `LUMAP_2D_RESFACTORED` |

---

## Extending `fakedata`

When you need an attribute from `data.py` that `fakedata` doesn't have, add it in `__init__` following the same pattern used in `luto/data.py`. Keep additions minimal — only load what the specific inspection task requires. Update the code block above in this skill file when you do.