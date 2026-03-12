"""
Filter evaluation logs to extract only the useful result lines,
stripping out spam/error noise.
"""

import re
import sys
from pathlib import Path

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")

# Patterns that identify useful lines
USEFUL_PATTERNS = [
    r"Saving video:",
    r"Combined video saved to:",
    r"Success rate:",
    # stat line: task | policy | dataset | result (contains ANSI color codes and pipes)
    r"\x1b\[[0-9;]*m[^\x1b]+\x1b\[0m\s*\|\s*\x1b",
]

COMPILED = [re.compile(p) for p in USEFUL_PATTERNS]
SUCCESS_RATE_RE = re.compile(r"Success rate:.*?(\d+)/(\d+)\s*=>\s*([\d.]+)%.*current seed:\s*(\d+)")


def strip_ansi(line: str) -> str:
    return ANSI_ESCAPE.sub("", line)


def is_useful(line: str) -> bool:
    return any(p.search(line) for p in COMPILED)


def filter_log(path: Path) -> list[str]:
    try:
        text = path.read_text(errors="replace")
    except Exception as e:
        print(f"  [error reading {path.name}: {e}]", file=sys.stderr)
        return []

    useful = [line for line in text.splitlines() if is_useful(line)]
    return useful


def extract_final_accuracy(useful_lines: list[str]) -> str | None:
    """Return the last Success rate line, stripped of ANSI."""
    for line in reversed(useful_lines):
        if SUCCESS_RATE_RE.search(strip_ansi(line)):
            return strip_ansi(line).strip()
    return None


def main():
    log_dir = Path("logs")
    if not log_dir.exists():
        print(f"Directory '{log_dir}' not found.", file=sys.stderr)
        sys.exit(1)

    plain = not sys.stdout.isatty()  # strip ANSI when piped/redirected

    log_files = sorted(log_dir.glob("*.log"))
    if not log_files:
        print("No .log files found.", file=sys.stderr)
        sys.exit(1)

    print("Final accuracy summary")
    print("=" * 60)
    for log_path in log_files:
        useful_lines = filter_log(log_path)
        final = extract_final_accuracy(useful_lines)
        label = log_path.stem
        if final:
            print(f"{label}:\n  {final}")
        else:
            print(f"{label}:\n  (no result)")
    print()


if __name__ == "__main__":
    main()
