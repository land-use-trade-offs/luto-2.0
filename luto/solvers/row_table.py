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

ROW_FIELDS_INT = ('cell', 'm', 'j', 'k', 'am_idx', 'from_m', 'from_j', 'from_k', 'local_r', 'commodity')   # -1 where n/a
ROW_FIELDS_CODED = ('family', 'group', 'region', 'item', 'presence', 'bound', 'state')                     # codes into attrs['vocab'][field]


def make_part(family: str, group: str, keys: dict, A: sparse.csr_matrix, rhs, sense, names, scale=None) -> xr.Dataset:
    """One family's rows: ``keys`` = {field: labels per row} over the row schema (``ROW_FIELDS_INT`` as ints,
    ``ROW_FIELDS_CODED`` as labels); ``sense`` one character or one per row; ``A`` the family's block."""
    n_rows = A.shape[0]
    unknown = set(keys) - set(ROW_FIELDS_INT) - set(ROW_FIELDS_CODED)
    assert not unknown, f'{family}: key field(s) {unknown} are not in the row schema'
    fields = {field: (('row',), np.asarray(labels, dtype=np.int32 if field in ROW_FIELDS_INT else object)) for field, labels in keys.items()}
    sense = np.full(n_rows, sense, dtype=object) if isinstance(sense, str) else np.asarray(sense, dtype=object)
    return xr.Dataset(
        dict(**fields,
             rhs=(('row',), np.asarray(rhs, dtype=np.float64)),
             sense=(('row',), sense),
             name=(('row',), np.asarray(names, dtype=object)),
             scale=(('row',), np.ones(n_rows, dtype=np.float64) if scale is None else np.asarray(scale, dtype=np.float64))),
        attrs=dict(family=family, group=group, A=A, keys=list(keys)))


def stack_rows(parts: list) -> xr.Dataset:
    """The row table: the parts back to back in the order given (the model's row order), dim ``row`` =
    Constr.index. Every field of the schema over every row, the coded fields as int32 codes into
    ``attrs['vocab']``, the ONE A vstacked into ``attrs['A']`` (rows × n_all), ``attrs['family_range']`` =
    {family: (start, stop)} (the rows each family owns, as ``block_range`` does for the columns),
    ``attrs['keys']`` = {family: its key fields}, and ``active`` — a dropped row is flagged off, the table
    never shrinks. The solver adds ``constr`` (the Gurobi handle) after ``addMConstr``."""
    widths = [part.sizes['row'] for part in parts]
    bounds = np.cumsum([0, *widths])
    n_rows = int(bounds[-1])
    family_range = {part.attrs['family']: (int(start), int(stop)) for part, start, stop in zip(parts, bounds[:-1], bounds[1:])}

    def field(name, dtype, fill):
        """One field over the whole table: each part's array for it, or the fill where the part has no such field."""
        return np.concatenate([np.asarray(part[name].values if name in part else np.full(width, fill), dtype=dtype)
                               for part, width in zip(parts, widths)]) if parts else np.empty(0, dtype=dtype)

    vocab = {}
    fields = {name: field(name, np.int32, -1) for name in ROW_FIELDS_INT}
    for name in ROW_FIELDS_CODED:                                                         # labels -> codes, the vocabulary in order of first appearance
        code_of = {}
        codes = np.full(n_rows, -1, dtype=np.int32)
        for part, start, stop in zip(parts, bounds[:-1], bounds[1:]):
            if name in ('family', 'group'):                                               # one label per part
                codes[start:stop] = code_of.setdefault(part.attrs[name], len(code_of))
            elif name in part:                                                            # one label per row, on the parts that carry the field
                codes[start:stop] = [code_of.setdefault(label, len(code_of)) for label in part[name].values]
        vocab[name] = list(code_of)
        fields[name] = codes

    return xr.Dataset(
        dict(**{name: (('row',), values) for name, values in fields.items()},
             rhs=(('row',), field('rhs', np.float64, np.nan)),
             sense=(('row',), field('sense', object, None)),
             name=(('row',), field('name', object, None)),
             scale=(('row',), field('scale', np.float64, 1.0)),
             active=(('row',), np.ones(n_rows, dtype=bool))),
        attrs=dict(A=sparse.vstack([part.attrs['A'] for part in parts], format='csr') if parts else None,
                   family_range=family_range,
                   keys={part.attrs['family']: part.attrs['keys'] for part in parts},
                   vocab=vocab))


# ── the queries: how the rest of the model asks the table what it holds ──────────────────────────

def family_rows(table: xr.Dataset, family: str) -> slice | None:
    """The rows one family owns (``attrs['family_range']``), None where the family was not built."""
    span = table.attrs['family_range'].get(family)
    return slice(*span) if span is not None else None


def decode(table: xr.Dataset, field: str, rows=None) -> np.ndarray:
    """A field's values at ``rows`` (a slice / mask / index array; every row by default) as labels: a coded
    field through its vocabulary (None where -1), an int field as it is."""
    values = table[field].values if rows is None else table[field].values[rows]
    if field not in ROW_FIELDS_CODED:
        return values
    labels = np.array([*table.attrs['vocab'][field], None], dtype=object)                 # -1 indexes the trailing None
    return labels[values]


def rows_where(table: xr.Dataset, **fields) -> np.ndarray:
    """A boolean mask over the table: the rows whose fields carry the given labels (``family='GBF8', region='AUSTRALIA'``)."""
    mask = np.ones(table.sizes['row'], dtype=bool)
    for field, label in fields.items():
        if field in ROW_FIELDS_CODED:
            vocab = table.attrs['vocab'][field]
            label = vocab.index(label) if label in vocab else -2                         # a label the table has never seen matches no row
        mask &= table[field].values == label
    return mask


def keys_of(table: xr.Dataset, family: str, rows=None) -> list:
    """The family's row keys as tuples (its key fields, decoded), at ``rows`` (its own rows by default), in row order."""
    rows = family_rows(table, family) if rows is None else rows
    columns = [decode(table, field, rows) for field in table.attrs['keys'][family]]
    n = table[table.attrs['keys'][family][0]].values[rows].size if columns else table['name'].values[rows].size
    return list(zip(*columns)) if columns else [()] * n
