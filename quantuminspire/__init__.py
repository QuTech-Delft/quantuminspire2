"""quantuminspire.

SDK for the Quantum Inspire platform.
"""

import sys
from contextlib import suppress
from importlib.metadata import PackageNotFoundError, version

with suppress(PackageNotFoundError):
    __version__ = version(__name__)
