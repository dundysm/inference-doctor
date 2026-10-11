"""Make CLI output plain text in every environment, so assertions on it are stable.

Rich colours output when it detects CI or FORCE_COLOR, and typer renders help with Rich
unless TYPER_USE_RICH is off; colour codes inside a word break substring checks. This
runs before the tests import typer or rich.
"""

import os

os.environ.pop("FORCE_COLOR", None)
os.environ["NO_COLOR"] = "1"
os.environ.setdefault("TYPER_USE_RICH", "0")
