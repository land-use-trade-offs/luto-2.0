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

import numpy as np
import xarray as xr

from scipy import sparse


# ═══════════════════════════ the row table: how the rows are stored, and how they are asked ═══════════════════════════

# The ONE declaration of the row table: every field at its dtype, as an empty table. ``get_rows`` stacks the families'
# tables onto it (``xr.concat``), so a field no family carries this step is still there; a family passes only the
# labels it has (``make_part``), and the rest is filled — -1 on an int field, None on a label. The ints: ``cell`` (every
# row of a per-cell family), ``m, j, k`` (the (lm, lu) / non-ag lu the row is about), ``am_idx`` (the ag-mgt option),
# ``from_m, from_j, from_k, local_r`` (a source cap's source). The labels, as they are (one shared str per row):
# ``family`` (the family's name: nothing on the table is a code, and it carries no index — a family's rows are the mask
# ``rows['family'].values == name``), ``region`` (the region a GBF / regional-cap / water / renewable row is about),
# ``GBF_target`` (the vegetation group / species / community a GBF3 / GBF4 / GBF8 row targets), ``GBF4_presence``,
# ``demand_commodity`` and ``demand_bound`` (eq / lower / upper).
ROW_SCHEMA = xr.Dataset(dict(
    family=(('row',), np.empty(0, dtype=object)),
    **{field: (('row',), np.empty(0, dtype=np.int32)) for field in ('cell', 'm', 'j', 'k', 'am_idx', 'from_m', 'from_j', 'from_k', 'local_r')},
    **{field: (('row',), np.empty(0, dtype=object)) for field in ('region', 'GBF_target', 'GBF4_presence', 'demand_commodity', 'demand_bound')},
    rhs=(('row',), np.empty(0, dtype=np.float64)),
    sense=(('row',), np.empty(0, dtype=object)),
    name=(('row',), np.empty(0, dtype=object)),
    scale=(('row',), np.empty(0, dtype=np.float64)),
    active=(('row',), np.empty(0, dtype=bool)),
    redundant=(('row',), np.empty(0, dtype=bool))))

ROW_FILL = {field: None if var.dtype == object else -1 for field, var in ROW_SCHEMA.data_vars.items()}   # what a family does not carry (only the labels and ints are ever missing)


def make_part(family: str, A: sparse.csr_matrix, rhs, sense, names, scale=None, **labels) -> tuple[sparse.csr_matrix, xr.Dataset]:
    """One family's rows, as the pair (A, table)"""

    n_rows = A.shape[0]
    unknown = set(labels) - set(ROW_SCHEMA.data_vars)
    assert not unknown, f'{family}: label field(s) {unknown} are not in the row schema'

    fields = {field: (('row',), np.asarray(values, dtype=ROW_SCHEMA[field].dtype)) for field, values in labels.items()}

    sense = np.full(n_rows, sense, dtype=object) if isinstance(sense, str) else np.asarray(sense, dtype=object)

    table = xr.Dataset(
        dict(family=(('row',), np.full(n_rows, family, dtype=object)),
             **fields,
             rhs=(('row',), np.asarray(rhs, dtype=np.float64)),
             sense=(('row',), sense),
             name=(('row',), np.asarray(names, dtype=object)),
             scale=(('row',), np.ones(n_rows, dtype=np.float64) if scale is None else np.asarray(scale, dtype=np.float64)),
             active=(('row',), np.ones(n_rows, dtype=bool)),
             redundant=(('row',), np.zeros(n_rows, dtype=bool))))

    return A, table


# ── the queries: how the rest of the model asks the table what it holds ──────────────────────────

def rows_where(table: xr.Dataset, **fields) -> np.ndarray:
    """A boolean mask over the table: the rows whose fields carry the given values (``family='GBF8', region='AUSTRALIA'``).
    Nothing on the table is a code and it carries no index: a family's rows are ``table['family'].values == name``."""
    mask = np.ones(table.sizes['row'], dtype=bool)
    for field, value in fields.items():
        mask &= table[field].values == value
    return mask
