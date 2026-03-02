"""Generate Inno Setup preprocessor defines.

This avoids manual edits in the .iss file.

Outputs:
- installer/_version.iss  (contains #define AppVersion "x.y.z")
"""

from __future__ import annotations

import os
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    out_file = repo_root / 'installer' / '_version.iss'

    # Import version from app
    # (Import inside main so running this script from any cwd works.)
    import sys

    sys.path.insert(0, str(repo_root))
    from experimental_web.core.version import __version__

    out_file.write_text(f'#define AppVersion "{__version__}"\n', encoding='utf-8')
    print(f'Wrote {out_file} (AppVersion={__version__})')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
