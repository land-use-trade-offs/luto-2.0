# Collaborating on LUTO: key design rules

These are the rules the code is built around, for everyone contributing to LUTO. A PR that follows them is quick to
review and merge; one that breaks them usually has to be reworked first.

## 1. No test scripts in the repository

Keep test and validation scripts outside `luto/` and outside the repository. Run them on your side and put what they show in the PR description: what you checked, at which
resolution, and the result.

## 2. Process data in `dataprep.py`, only read it in `Data`

All processing of LUTO's raw data happens once, in `luto/dataprep.py`. It writes model-ready files to `input/`, NetCDF
for anything spatial. After that, `luto/data.py` only reads them: it selects cells and years, but doesn't compute, join
or reshape raw data.

This keeps a run fast, and keeps every input visible as a file that can be opened and checked on its own.

New data your work adds goes in one folder of its own under `input/` (e.g. `input/<your_name>_build/`), so it can be
sent as it is and stays apart from the existing inputs. The scripts that build it go in its `scripts/` folder
(`input/Nick_build/scripts/`): git tracks that folder and ignores the built data beside it (`input/.gitignore`). They
stay outside `luto/` and don't import it. Say in each PR which files a run needs.

> **Important reminder: NaN in the inputs.** In an input, NaN means "not applicable". Where a land use is applicable
> is said by one mask, `x_mrj.nc` (built in `dataprep.py`), not by the NaN in the data. So handle every NaN when the
> file is built, and leave none in `input/`: LUTO never fills, masks or replaces NaN (rule 5).
>
> Example: Apples yield is NaN on every cell of an SA2 that grows no apples.
>
> - Where `x_mrj` says Apples is **not allowed**, the NaN only repeats the mask. Write 0: the model never uses it.
> - Where `x_mrj` says Apples **is allowed** (say, a cell that grew apples in 2010 but whose SA2 has no apples data),
>   the NaN is a real gap. A 0 would give the model apples at no cost, or with no yield. Supply a real value (from a
>   neighbouring or regional figure), or switch Apples off on that cell in the mask, and say which in the build script.
>
> One exception, until it is re-encoded: the transition matrices (`ag_tmatrix.npy`, `ag_to_ep_tmatrix.npy`,
> `ep_to_ag_tmatrix.npy`, `ag_to_destock_tmatrix.npy`) use NaN to mean "this land-use change is not allowed". Never
> fill that NaN: a 0 there would make a forbidden change allowed and free, and nothing would crash.

## 3. `Data` holds the inputs; the economics modules turn them into solver inputs

The pipeline goes one way:

```
input/*.nc  ->  Data attributes  ->  economics modules  ->  solvers/row_inputs.py  ->  row_builder / col_builder  ->  solver
               (luto/data.py)       (luto/economics/...)   (get_row_inputs,
                                                            get_economics)
```

- Every input is an attribute of the `Data` object, e.g. `data.AGEC_CROPS` or `data.REAL_AREA`.
- The economics modules (`cost.py`, `revenue.py`, `quantity.py`, `ghg.py`, `water.py`, ...) read those attributes and
  compute the per-cell matrices. They never open a file themselves.
- `row_inputs.py` collects the matrices for the solver.

A new input is therefore one new attribute on `Data`, read by the economics function that uses it.

## 4. Save a `Data` object once, then step through the code with it

Build `Data` once, save it, and load it back whenever you need it:

```python
import luto.simulation as sim

data = sim.load_data()                                   # build once (slow)
sim.save_data_to_disk(data, 'data_RES5.lz4')

data = sim.load_data_from_disk('data_RES5.lz4')          # later: seconds; RESFACTOR in settings.py must match
```

With a loaded `data` you can walk through any economics function line by line, since they all take `data` as their
first argument. To walk through a method of `Data` itself, name the object `self` and run the method's body one line
at a time:

```python
self = data
# paste the lines of the method here and inspect each result
```

## 5. Keep things together, so errors can't be silent

The most important rule. When one piece of logic is spread over several places, a change in one place can quietly
disagree with another. Nothing crashes, and the results are just wrong. So:

- **One quantity, one place.** Compute a value in one function, and have everything else read it there. Don't
  recompute it, and don't correct it somewhere else, as when a weight is applied in `data.py` to values the economics
  modules compute.
- **Labels travel with the data.** Use xarray with named dimensions (`cell`, `lm`, `lu`, `year`) and select by name
  (`.sel(lm='dry', lu='Apples')`), not by position. A wrong position gives a wrong number; a wrong name gives an error.
- **Fail loudly.** A missing value is not zero. If NaN reaches the model, let the run crash: don't add `fillna`,
  `nan_to_num`, `np.where(np.isnan(...), ...)` or a NaN check to make it go through. The crash says the input is wrong,
  and the fix goes in the input file and its build script (rule 2), never in `luto/`.
- **One change, one PR.** A PR does one thing and starts from the latest `jinzhu`. Once it's merged it's frozen: a
  fix is a new PR, and a merged branch is never rebased or force-pushed.

## 6. Write down what you find in `docs/FINDINGS.md`

When debugging turns up something interesting (why a target is infeasible, where an input disagrees with another,
a result that surprised you), add an entry to `docs/FINDINGS.md`. Feel free to have an AI assistant write it up from
your session. Entries go newest first, each under a heading `## YYYYMMDD — <what was found>`, starting with a short
TL;DR, then the evidence (numbers, files, settings, resolution). That way the reasoning stays with the repository
instead of in a chat history, and neither of us has to rediscover it.
