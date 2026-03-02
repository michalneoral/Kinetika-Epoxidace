"""Application version.

Single source of truth for the runtime version.

Why a dedicated module:
- The app can display its version (e.g. Settings → O aplikaci).
- Build scripts can import this module and automatically inject the version into
  the installer filename/metadata (no manual edits).

Recommended release flow:
- Bump ``__version__`` here.
- Create a GitHub Release with tag: ``v<__version__>``.
"""

from __future__ import annotations

__version__ = "0.5.6"
