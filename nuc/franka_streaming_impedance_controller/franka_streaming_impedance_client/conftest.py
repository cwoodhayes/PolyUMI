"""
Make the package importable when pytest is run directly against this directory.

Under colcon the package is installed before its tests run, so this is not needed there. Run
bare — ``pytest test/``, or as part of a wider sweep like ``pytest nuc/`` — nothing has put the
package root on ``sys.path``, and the import fails at collection.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
