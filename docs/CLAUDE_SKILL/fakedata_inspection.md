# Skill: Using `fakedata` to Inspect and Prototype with `data.py` Objects

`fakedata` is a lightweight stand-in for the full `luto.data.Data` object. It loads only the spatial masking and geo-metadata needed to use LUTO helpers (e.g. `arr_to_xr()`), so you can inspect arrays and prototype code without loading gigabytes of simulation inputs.

> **Note:** `fakedata` is not part of the `luto` package. Copy the full class code below into your own notebook or script, outside `luto/`. Its masking section is copied from `Data.__init__` ("Masking and spatial coarse graining"), so if that section of `luto/data.py` changes, copy it again.

---

## When to Use

- You want to call `arr_to_xr()` or other helpers in `luto/tools/__init__.py` (`arr_to_xr`, `mrj_to_xr`) without a full simulation run.
- You are inspecting or debugging a 1D numpy array in spatial context (plotting, reprojecting, etc.).
- You are prototyping new code that takes a `data` argument and need a quick `data` object to test with.

---

## Full Class Code

Copy this into your notebook or script:

```python
import os
from math import ceil

import numpy as np
import rasterio
import xarray as xr
from affine import Affine

import luto.settings as settings


class fakedata():
    """
    A stand-in for `luto.data.Data` that loads only the masks, the cell mappings and the geo metadata. The masking
    section is copied from `Data.__init__` ("Masking and spatial coarse graining"); re-copy it if that section changes.
    """

    def __init__(self):

        # Path for write module - overwrite when provided with a base and target year
        self.path = None
        self.last_year = None

        # Output containers, as on Data
        self.lumaps = {}
        self.lmmaps = {}
        self.ammaps = {}
        self.ag_dvars = {}
        self.non_ag_dvars = {}
        self.ag_man_dvars = {}
        self.prod_data = {}
        self.obj_vals = {}

        print(f'Fake Data Initialization at RES{settings.RESFACTOR}')

        self.YR_CAL_BASE = 2010  # The base year, i.e. where year index yr_idx == 0.

        ###############################################################
        # Masking and spatial coarse graining (copied from Data.__init__).
        ###############################################################
        self.RESMULT = settings.RESFACTOR ** 2
        self.NODATA = -9999
        self.MASK_LU_CODE = -1

        self.LUMAP_NO_RESFACTOR = xr.load_dataarray(os.path.join(settings.INPUT_DIR, "lumap.nc")).astype(np.int8)

        with rasterio.open(os.path.join(settings.INPUT_DIR, "NLUM_2010-11_mask.tif")) as rst:
            self.NLUM_MASK = rst.read(1).astype(np.int8)
            self.LUMAP_2D_FULLRES = np.full_like(self.NLUM_MASK, self.NODATA, dtype=np.int16)
            np.place(self.LUMAP_2D_FULLRES, self.NLUM_MASK == 1, self.LUMAP_NO_RESFACTOR.values)
            self.GEO_META_FULLRES = rst.meta
            self.GEO_META_FULLRES['dtype'] = 'float32'
            self.GEO_META_FULLRES['nodata'] = self.NODATA

        self.LUMASK = self.LUMAP_NO_RESFACTOR.values != self.MASK_LU_CODE
        self.FULLRES_1D_TO_2D = np.vstack(np.nonzero(self.NLUM_MASK))

        if settings.RESFACTOR > 1:
            fullres_rows, fullres_cols = self.FULLRES_1D_TO_2D

            # The block of every full-res land cell, numbered row-major on a grid of whole RF x RF blocks
            n_block_rows = ceil(self.NLUM_MASK.shape[0] / settings.RESFACTOR)
            n_block_cols = ceil(self.NLUM_MASK.shape[1] / settings.RESFACTOR)
            block_id_of_cell = (fullres_rows // settings.RESFACTOR) * n_block_cols + fullres_cols // settings.RESFACTOR

            # A block is a model cell if it holds >= 1 land-use cell
            has_land_use = np.bincount(block_id_of_cell, weights=self.LUMASK, minlength=n_block_rows * n_block_cols) > 0
            model_blocks = np.flatnonzero(has_land_use)

            # Each model cell's representative cell: the block's centre cell where the centre is land; for a coastal
            # block whose centre is sea, the block's land-use cell nearest the centre
            fullres_idx_2d = np.full(self.NLUM_MASK.shape, -1, dtype=np.int64)
            fullres_idx_2d[fullres_rows, fullres_cols] = np.arange(fullres_rows.size)
            block_rows, block_cols = np.divmod(model_blocks, n_block_cols)
            centre_rows = block_rows * settings.RESFACTOR + settings.RESFACTOR // 2
            centre_cols = block_cols * settings.RESFACTOR + settings.RESFACTOR // 2
            inside_grid = (centre_rows < self.NLUM_MASK.shape[0]) & (centre_cols < self.NLUM_MASK.shape[1])
            representative_cell = np.full(model_blocks.size, -1, dtype=np.int64)
            representative_cell[inside_grid] = fullres_idx_2d[centre_rows[inside_grid], centre_cols[inside_grid]]

            needs_substitute = np.zeros(n_block_rows * n_block_cols, dtype=bool)
            needs_substitute[model_blocks[representative_cell < 0]] = True
            candidate_cells = np.flatnonzero(self.LUMASK & needs_substitute[block_id_of_cell])
            candidate_blocks = block_id_of_cell[candidate_cells]
            distance_to_centre = ((fullres_rows[candidate_cells] - (candidate_blocks // n_block_cols * settings.RESFACTOR + settings.RESFACTOR // 2)) ** 2
                                  + (fullres_cols[candidate_cells] - (candidate_blocks % n_block_cols * settings.RESFACTOR + settings.RESFACTOR // 2)) ** 2)
            nearest_first = np.lexsort((candidate_cells, distance_to_centre, candidate_blocks))
            substitute_blocks, first = np.unique(candidate_blocks[nearest_first], return_index=True)
            position_of_block = np.full(n_block_rows * n_block_cols, -1, dtype=np.int64)
            position_of_block[model_blocks] = np.arange(model_blocks.size)
            representative_cell[position_of_block[substitute_blocks]] = candidate_cells[nearest_first[first]]

            # The model cells in the full-res order of their representative cells
            in_read_order = np.argsort(representative_cell)
            model_blocks, representative_cell = model_blocks[in_read_order], representative_cell[in_read_order]
            self.RES_1D_TO_2D = np.vstack(np.divmod(model_blocks, n_block_cols))

            # Each full-res land cell reads its block's model-cell index (-1: block not modelled)
            model_cell_id_of_block = np.full(n_block_rows * n_block_cols, -1, dtype=np.int32)
            model_cell_id_of_block[model_blocks] = np.arange(model_blocks.size)
            self.FULL2RES_IDX_MAP = model_cell_id_of_block[block_id_of_cell]

            # MASK: the representative cell of each model cell, among the full-res land cells
            self.MASK = np.zeros(fullres_rows.size, dtype=bool)
            self.MASK[representative_cell] = True

            self.LUMAP_2D_RESFACTORED = self.LUMAP_2D_FULLRES[settings.RESFACTOR//2::settings.RESFACTOR, settings.RESFACTOR//2::settings.RESFACTOR]
            self.GEO_META = self.update_geo_meta()

        elif settings.RESFACTOR == 1:
            self.MASK = self.LUMASK
            self.RES_1D_TO_2D = self.FULLRES_1D_TO_2D[:, self.LUMASK]
            self.FULL2RES_IDX_MAP = np.where(self.LUMASK, np.cumsum(self.LUMASK) - 1, -1).astype(np.int32)
            self.GEO_META = self.GEO_META_FULLRES
            self.LUMAP_2D_RESFACTORED = self.LUMAP_2D_FULLRES
        else:
            raise KeyError("Resfactor setting invalid")

        # Data reads it off REAL_AREA; here the model cells are counted directly
        self.NCELLS = int(self.MASK.sum())

        print(f"   - Total land cells in full resolution: {self.NLUM_MASK.sum()}")
        print(f"   - Total model cells at RES{settings.RESFACTOR}: {self.NCELLS}")

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
from luto.tools import arr_to_xr
da = arr_to_xr(data, my_1d_array)
da.plot()
```

---

## What `fakedata` Provides

| Attribute | Description |
|---|---|
| `NLUM_MASK` | 2D raster mask (0=ocean, 1=land) |
| `LUMASK` | 1D boolean mask over the full-res land cells: `True` for land-use cells |
| `LUMAP_NO_RESFACTOR` | 1D land-use map at full resolution (DataArray, from `lumap.nc`) |
| `LUMAP_2D_FULLRES` | 2D land-use map at full resolution |
| `LUMAP_2D_RESFACTORED` | 2D land-use map at resfactored resolution |
| `FULLRES_1D_TO_2D` | (row, col) of every full-res land cell, shape (2, n) |
| `RES_1D_TO_2D` | (row, col) of every model cell on the resfactored grid, shape (2, NCELLS) |
| `FULL2RES_IDX_MAP` | for every full-res land cell, the index of its block's model cell (-1: block not modelled) |
| `MASK` | 1D boolean mask over the full-res land cells: the cell each model cell's inputs are read at (the block centre, or for a coastal block with a sea centre the land-use cell nearest it) |
| `NCELLS` | number of model cells |
| `GEO_META_FULLRES` | Rasterio geo-metadata at full resolution |
| `GEO_META` | Rasterio geo-metadata adjusted for `RESFACTOR` |
| `NODATA` | Nodata value (`-9999`) |
| `RESMULT` | `RESFACTOR ** 2` |
| `YR_CAL_BASE` | Base year (`2010`) |
| `lumaps`, `lmmaps`, `ammaps`, `ag_dvars`, `non_ag_dvars`, `ag_man_dvars`, `prod_data`, `obj_vals` | Empty output containers (same as `Data`) |

---

## Array Size Conventions

`arr_to_xr()` accepts three array sizes and routes them automatically:

| Array size | Meaning | Branch in `arr_to_xr` |
|---|---|---|
| `data.LUMASK.size` | Full-res all-land 1D | Uses `GEO_META_FULLRES`, placed at `FULLRES_1D_TO_2D` |
| `data.LUMASK.sum()` | Full-res valid-LU 1D | Same, at the land-use cells `FULLRES_1D_TO_2D[:, LUMASK]` |
| anything else (`data.NCELLS`) | Resfactored 1D | Uses `data.GEO_META`, placed at `RES_1D_TO_2D` on `LUMAP_2D_RESFACTORED` |

---

## Extending `fakedata`

When you need an attribute from `data.py` that `fakedata` doesn't have, add it in `__init__` following the same pattern used in `luto/data.py`. Keep additions minimal — only load what the specific inspection task requires. Update the code block above in this skill file when you do.