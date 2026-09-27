"""Rehearse every command the README tells a user to run.

`implementation.md` §6.1 requires a fresh-clone rehearsal and says "fix the
README, not your memory" -- a command in the README that does not work is a demo
failure. This walks the documented commands and reports any that fail, so the
README is checked rather than assumed.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

PY = sys.executable
README = Path(__file__).resolve().parent.parent / "README.md"

# Commands worth executing, in the order a new user would run them.
SMOKE = [
    ["-m", "src.cli", "inspect"],
    ["-m", "src.cli", "ask", "--retrieve-only", "expense ratio of HDFC Large Cap Fund?"],
    ["-m", "src.cli", "ask", "--explain", "exit load on HDFC Flexi Cap Fund"],
    ["-m", "evals.integrity"],
    ["scripts/eval_retrieval.py"],
    ["scripts/dump_index.py", "--no-vectors", "--out", "/tmp/_rehearsal_dump.txt"],
    ["scripts/build_disclaimer.py"],
    ["scripts/build_sample_qa.py"],
    ["scripts/list_models.py"],
]

# Commands that are allowed to exit non-zero, with the reason. A rehearsal that
# cries wolf over correct behaviour is as useless as one that stays silent.
EXPECTED_NONZERO = {
    "eval_retrieval": "exit 1 while the 3 documented retrieval gaps are open",
    "list_models": "exit 1 without ANTHROPIC_API_KEY; that is the script's job",
}

# Referenced but not executed: needs a browser/server, or mutates the index.
REFERENCED_ONLY = {
    "-m src.cli ingest": "mutates the index; covered by the full suite",
    "streamlit run src/app.py": "needs a browser",
}


def readme_commands() -> list[str]:
    text = README.read_text(encoding="utf-8")
    found: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(("python ", ".venv/bin/python ", "$ ")):
            cmd = line.removeprefix("$ ").removeprefix(".venv/bin/python ").removeprefix("python ")
            cmd = cmd.split("#")[0].strip()
            if cmd and cmd not in found:
                found.append(cmd)
    return found


def main() -> int:
    print("=" * 74)
    print("README REHEARSAL -- does every documented command actually run?")
    print("=" * 74)

    referenced = readme_commands()
    print(f"\n  {len(referenced)} commands referenced in README.md\n")

    failures: list[str] = []
    for argv in SMOKE:
        label = " ".join(argv)
        try:
            r = subprocess.run(
                [PY, *argv], capture_output=True, text=True, timeout=900
            )
        except subprocess.TimeoutExpired:
            failures.append(f"{label}  -> TIMEOUT")
            print(f"  TIMEOUT  {label}")
            continue
        if r.returncode == 0:
            print(f"  ok       {label}")
            continue
        expected = next(
            (why for frag, why in EXPECTED_NONZERO.items() if frag in label), None
        )
        if expected:
            print(f"  ok*      {label}  -> exit {r.returncode} ({expected})")
            continue
        note = REFERENCED_ONLY.get(label.split("--")[0].strip(), "")
        failures.append(f"{label}  -> exit {r.returncode} {note}")
        print(f"  FAIL     {label}  -> exit {r.returncode} {note}")

    print()
    if failures:
        print(f"  {len(failures)} README command(s) need fixing:")
        for f in failures:
            print(f"    - {f}")
        return 1
    print("  All executable README commands run.")
    print("  * = non-zero exit is the documented, expected behaviour for that command.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
