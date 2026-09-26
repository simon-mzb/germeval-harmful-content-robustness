"""Does the grid the code RUNS match the grid the matrix DECLARES?

The most dangerous defect this project had (found 2026-08-27): the encoder's
GRIDS was a hardcoded Python constant while `configs/e2_matrix.yaml` declared a
different grid, and only the Python one ran. Two consequences, the second far
worse than the first:

  1. Widening the lr window in the matrix -- exactly what the lr probe concluded
     -- changed nothing. The campaign would have run the old window.
  2. `config_id` is computed FROM THE MATRIX, so a stored run would have carried
     the identity of a configuration it never used. No existing check could
     catch that, because every check compares the matrix against itself.

The encoder now derives its grid from the matrix. The TML grids are still
written out in Python -- deliberately, because they produced committed B3
results and re-deriving them would change the cell ORDER, which decides exact
ties. So the duplication stays, and this test is what makes it safe: it fails
the moment the two drift apart, in either direction.

Usage: .venv/bin/python -m tests.test_grid_matches_matrix
"""
import itertools
import sys

from src.component_store import load_matrix

PASS = FAIL = 0


def check(name, got, want):
    global PASS, FAIL
    if got == want:
        PASS += 1
        print("  PASS  {}".format(name))
    else:
        FAIL += 1
        print("  FAIL  {}\n          declared: {!r}\n          running : {!r}".format(name, want, got))


def declared_cells(component, matrix):
    """Every cell the matrix declares, as a set of frozen key/value pairs."""
    block = matrix["components"][component]
    if "grid" not in block and block.get("inherits"):
        block = matrix["components"][block["inherits"]]
    grid = block["grid"]
    keys = sorted(grid)
    values = [grid[k] if isinstance(grid[k], list) else [grid[k]] for k in keys]
    return {frozenset(zip(keys, combo)) for combo in itertools.product(*values)}


def running_cells(cells):
    return {frozenset(c.items()) for c in cells}


matrix = load_matrix()

print("\n=== the encoder derives its grid, so this is an identity ===")
import src.encoder_components as enc

for comp in enc.GRIDS:
    check("{} runs exactly what the matrix declares".format(comp),
          running_cells(enc.GRIDS[comp]), declared_cells(comp, matrix))

print("\n=== TML restates its grid in Python: the two must not drift ===")
import src.tml_components as tml

for comp in tml.GRIDS:
    check("{} runs exactly what the matrix declares".format(comp),
          running_cells(tml.GRIDS[comp]), declared_cells(comp, matrix))

print("\n=== and the encoder grid is the one the 2026-08-27 probe chose ===")
lrs = sorted({c["lr"] for c in enc.GRIDS["encoder_1b"]})
check("lr window is [2e-4, 3e-4, 5e-4]", lrs, [2.0e-4, 3.0e-4, 5.0e-4])
check("  ...and 3e-4, the probe's optimum, is INTERIOR",
      lrs[0] < 3.0e-4 < lrs[-1], True)

print("\n" + "-" * 40)
print("  {} passed, {} failed".format(PASS, FAIL))
print("-" * 40)
sys.exit(1 if FAIL else 0)
