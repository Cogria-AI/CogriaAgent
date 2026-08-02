import os
import sys

# Make the shared `sample_actions` module importable by absolute name, without
# relying on the `tests` package (three same-named tests/ dirs across the
# workspace make relative imports ambiguous under --import-mode=importlib).
sys.path.insert(0, os.path.dirname(__file__))
