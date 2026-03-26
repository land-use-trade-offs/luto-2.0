## Diagnostic Scripts

### `find_infeasible_ecnes.py`

Diagnoses infeasible GBF4 ECNES (Ecological Community NES) biodiversity constraints in a saved Gurobi MPS model. It removes all ECNES constraints to create a base model, then tests each constraint individually by maximizing its LHS to check if the target is achievable.

**Usage (Jupyter/IPython):**
```python
from luto.tests.find_infeasible_ecnes import find_infeasible_ecnes
base_model, results = find_infeasible_ecnes()
base_model, results = find_infeasible_ecnes("path/to/model.mps", workers=8)
```

- Uses joblib for parallel constraint checking
- Skips constraints with RHS <= 0 (trivially feasible)
- Reports infeasible constraints sorted by gap (worst first)
