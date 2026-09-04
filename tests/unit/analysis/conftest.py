"""Make the analysis/perturbations scripts importable as top-level modules.

Those scripts import each other by module name (they double as standalone
entry points), so their directory must be on sys.path before the tests import
them.
"""

import sys
from pathlib import Path

PERTURBATIONS_DIR = Path(__file__).resolve().parents[3] / "analysis" / "perturbations"
if str(PERTURBATIONS_DIR) not in sys.path:
    sys.path.insert(0, str(PERTURBATIONS_DIR))
