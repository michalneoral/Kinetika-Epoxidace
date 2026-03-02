"""Generate checksums.txt for GitHub Release assets.

Usage:
  py -3 build\generate_checksums.py installer\FAME_EPO_Manager_Setup-1.2.3.exe

Output:
  installer\checksums.txt
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest().lower()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('installer', help='Path to the built Setup.exe (Inno Setup output)')
    p.add_argument('--out', default='installer/checksums.txt', help='Output file')
    args = p.parse_args()

    installer = Path(args.installer).resolve()
    if not installer.exists():
        raise SystemExit(f'File not found: {installer}')

    digest = sha256_file(installer)
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    # Match the older format: "SHA256  <hash>  <filename>"
    out.write_text(f'SHA256  {digest}  {installer.name}\n', encoding='utf-8')
    print(f'Wrote {out} for {installer.name}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
