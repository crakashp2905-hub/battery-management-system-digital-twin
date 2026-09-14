"""Generate a self-contained HTML battery health & validation report.

    python scripts/generate_report.py             # nmc -> bms_report.html
    python scripts/generate_report.py lfp out.html
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bms.report import save_report  # noqa: E402


def main() -> int:
    chem = sys.argv[1] if len(sys.argv) > 1 else "nmc"
    out = sys.argv[2] if len(sys.argv) > 2 else "bms_report.html"
    path = save_report(out, chemistry=chem)
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
