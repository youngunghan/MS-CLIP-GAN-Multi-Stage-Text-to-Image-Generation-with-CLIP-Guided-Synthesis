import os
import sys

# Ensure the repo root is importable (options, utils, scripts, networks,
# dataset, criteria, experiments, preprocessing) regardless of how pytest
# was invoked or which import mode it uses, without requiring callers to
# set PYTHONPATH manually.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
