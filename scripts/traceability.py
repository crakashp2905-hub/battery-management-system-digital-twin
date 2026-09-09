"""FMEA → test traceability report.

Collects the test suite (``pytest --collect-only``), cross-references it against
``bms.fmea.FMEA_TEST_LINKS``, and prints a coverage matrix.  Exits non-zero if
any failure mode with ``RPN ≥ threshold`` has no covering test — a safety-case
gate you can run in CI.

Usage::

    python scripts/traceability.py            # human-readable
    python scripts/traceability.py --markdown  # matrix for docs
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bms  # noqa: E402

RPN_THRESHOLD = 100


def collect_test_ids() -> list[str]:
    """Test node ids (``file::Class`` and ``file::test_fn``), by scanning the
    test files directly — robust to pytest's ``--collect-only`` output format."""
    root = Path(__file__).resolve().parents[1]
    ids: list[str] = []
    for f in sorted((root / "tests").glob("test_*.py")):
        for line in f.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s.startswith("class Test") or s.startswith("def test_"):
                name = s.split("(")[0].removeprefix("class ").removeprefix("def ").strip(": ")
                ids.append(f"{f.name}::{name}")
    return ids


def main(markdown: bool = False) -> int:
    ids = collect_test_ids()
    df = bms.fmea_traceability(collected_test_ids=ids, rpn_threshold=RPN_THRESHOLD)

    if markdown:
        print("| failure mode | RPN | covered | # tests |")
        print("|---|---|---|---|")
        for _, r in df.iterrows():
            mark = "✅" if r["covered"] else ("❌" if r["gap"] else "—")
            print(f"| {r['failure_mode']} | {r['RPN']} | {mark} | {len(r['linked_tests'])} |")
    else:
        print(f"Collected {len(ids)} tests.\n")
        print(f"{'failure mode':30s} {'RPN':>4s}  {'covered':8s} tests")
        for _, r in df.iterrows():
            state = "OK" if r["covered"] else ("GAP" if r["gap"] else "n/a")
            print(f"{r['failure_mode']:30s} {r['RPN']:>4d}  {state:8s} {len(r['linked_tests'])}")

    gaps = df[df["gap"]]
    if len(gaps):
        print(f"\nTRACEABILITY FAIL: {len(gaps)} high-RPN mode(s) without a covering test:")
        for m in gaps["failure_mode"]:
            print(f"  - {m}")
        return 1
    print(f"\nTRACEABILITY OK: every mode with RPN >= {RPN_THRESHOLD} has >=1 covering test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(markdown="--markdown" in sys.argv))
