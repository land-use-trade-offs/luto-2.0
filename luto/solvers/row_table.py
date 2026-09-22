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

FAMILIES = ('demand', 'ghg', 'GBF2', 'GBF3_NVIS', 'GBF4_SNES', 'GBF4_ECNES', 'GBF8', 'ag_mgt_adoption',
            'regional_adoption_ag', 'regional_adoption_nonag', 'regional_adoption_nonag_sum', 'water', 'renewable',
            'ag_mgt_link', 'renewable_ceiling', 'source_cap_ag', 'source_cap_nonag',
            'node_balance_ag', 'node_balance_nonag')
ROW_FIELDS_INT = ('cell', 'm', 'j', 'k', 'am_idx', 'from_m', 'from_j', 'from_k', 'local_r', 'commodity')
ROW_FIELDS_CODED = ('region', 'item', 'presence', 'bound', 'state')


# what ``xr.concat`` needs to stack the families' tables as they are (``get_rows``): an empty table carrying every field
# at its dtype, so a field no family carries this step is still there, and the fill of a field a family does not carry
ROW_SCHEMA = xr.Dataset(dict(
    **{field: (('row',), np.empty(0, dtype=np.int32)) for field in ('family', *ROW_FIELDS_INT, *ROW_FIELDS_CODED)},
    rhs=(('row',), np.empty(0, dtype=np.float64)),
    sense=(('row',), np.empty(0, dtype=object)),
    name=(('row',), np.empty(0, dtype=object)),
    scale=(('row',), np.empty(0, dtype=np.float64)),
    active=(('row',), np.empty(0, dtype=bool)),
    redundant=(('row',), np.empty(0, dtype=bool))))

ROW_FILL = {field: -1 for field in (*ROW_FIELDS_INT, *ROW_FIELDS_CODED)}



def make_part(family: str, A: sparse.csr_matrix, rhs, sense, names, scale=None, **labels) -> tuple[sparse.csr_matrix, xr.Dataset]:
    """One family's rows, as the pair (A, table)"""

    n_rows = A.shape[0]
    unknown = set(labels) - set(ROW_FIELDS_INT) - set(ROW_FIELDS_CODED)
    assert not unknown, f'{family}: label field(s) {unknown} are not in the row schema'

    fields = {}
    vocab = {}
    for field, values in labels.items():
        if field in ROW_FIELDS_CODED:                                                     # labels -> codes, the map kept beside them
            code_of = {}
            values = [code_of.setdefault(label, len(code_of)) for label in values]
            vocab[field] = list(code_of)
        fields[field] = (('row',), np.asarray(values, dtype=np.int32))

    sense = np.full(n_rows, sense, dtype=object) if isinstance(sense, str) else np.asarray(sense, dtype=object)

    table = xr.Dataset(
        dict(family=(('row',), np.full(n_rows, FAMILIES.index(family), dtype=np.int32)),
             **fields,
             rhs=(('row',), np.asarray(rhs, dtype=np.float64)),
             sense=(('row',), sense),
             name=(('row',), np.asarray(names, dtype=object)),
             scale=(('row',), np.ones(n_rows, dtype=np.float64) if scale is None else np.asarray(scale, dtype=np.float64)),
             active=(('row',), np.ones(n_rows, dtype=bool)),
             redundant=(('row',), np.zeros(n_rows, dtype=bool))),
        attrs={f'vocab_{family}': vocab} if vocab else {})

    return A, table


# ── the queries: how the rest of the model asks the table what it holds ──────────────────────────

def decode(table: xr.Dataset, field: str, rows=None) -> np.ndarray:
    """A field's values at ``rows`` (a slice / mask / index array; every row by default) as labels: ``family`` through
    ``FAMILIES``, a coded field through the map of each row's own family (None where -1), an int field as it is."""
    rows = slice(None) if rows is None else rows
    values = table[field].values[rows]
    if field == 'family':
        return np.array(FAMILIES, dtype=object)[values]
    if field not in ROW_FIELDS_CODED:
        return values
    family = table['family'].values[rows]
    labels = np.full(values.shape, None, dtype=object)
    for code in np.unique(family):                                                        # the codes are local to a family: one lookup per family present
        vocab = table.attrs.get(f'vocab_{FAMILIES[code]}', {}).get(field)
        if vocab is not None:
            on = (family == code) & (values >= 0)
            labels[on] = np.array(vocab, dtype=object)[values[on]]
    return labels


def rows_where(table: xr.Dataset, **fields) -> np.ndarray:
    """A boolean mask over the table: the rows whose fields carry the given labels (``family='GBF8', region='AUSTRALIA'``);
    a coded field's label is looked up in the map of the ``family`` given with it."""
    mask = np.ones(table.sizes['row'], dtype=bool)
    for field, label in fields.items():
        if field == 'family':
            label = FAMILIES.index(label)
        elif field in ROW_FIELDS_CODED:
            vocab = table.attrs.get(f"vocab_{fields['family']}", {}).get(field, [])       # a coded field needs its family: the codes are local to it
            label = vocab.index(label) if label in vocab else -2                          # a label the family has never seen matches no row
        mask &= table[field].values == label
    return mask
