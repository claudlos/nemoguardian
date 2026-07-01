"""Grep-style secret / credential scanner for tracked files.

An offline CI guard that scans repository files for line shapes that look like
real secrets (cloud keys, provider API tokens, private-key headers, etc.) and
exits non-zero when any match is not covered by the allowlist.

Design notes:

* We scan the files returned by ``git ls-files`` (tracked files only) so that
  ``.venv``, build artifacts and untracked scratch files are ignored.
* The pattern set targets *high-signal* shapes (a fixed vendor prefix followed
  by enough entropy) to keep false positives low. Placeholder shapes used in
  ``.env.example`` / docs (``nmg_paste_your_key_here``, ``sk-...``) do not match.
* An allowlist covers the intentional secret-shaped fixtures used by the test
  suite plus any line carrying an inline ``# nosecret`` / ``pragma: allowlist
  secret`` marker.

Exit code is ``0`` when nothing suspicious is found and ``1`` on the first real
hit (all hits are printed to stderr).

Run directly::

    python scripts/secret_scan.py

or via ``make secret-scan``.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# name -> compiled pattern. Patterns intentionally demand a vendor prefix plus a
# run of high-entropy characters so that obvious placeholders do not trip them.
SECRET_PATTERNS: dict[str, re.Pattern[str]] = {
    "aws_access_key_id": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "github_token": re.compile(r"\bgh[pousr]_[0-9A-Za-z]{36,}\b"),
    "slack_token": re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{10,}\b"),
    "openai_key": re.compile(r"\bsk-[A-Za-z0-9]{40,}\b"),
    "stripe_live_secret": re.compile(r"\b[rs]k_live_[0-9A-Za-z]{20,}\b"),
    "google_api_key": re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),
    "huggingface_token": re.compile(r"\bhf_[0-9A-Za-z]{34,}\b"),
    "private_key_block": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----"),
}

# Inline markers that explicitly waive a line. The marker must be *comment
# anchored* (follow a ``#``) so an ordinary word that merely contains "nosec"
# as a substring — e.g. ``nanoseconds`` — does not silently disable detection.
_ALLOW_MARKER_RE = re.compile(
    r"#\s*(?:nosec(?:ret)?|pragma:\s*allowlist secret)\b"
)

# Path fragments whose files are allowed to contain secret-shaped test fixtures.
# Matched against the file's repo-relative POSIX path.
DEFAULT_ALLOWLIST_PATHS: tuple[str, ...] = (
    "tests/fixtures/",
    "tests/test_secret_scan",
    "tests/test_ci_tooling",
    # This scanner file itself contains the patterns above.
    "scripts/secret_scan.py",
)

_BINARY_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".zip", ".gz", ".whl",
    ".pt", ".bin", ".safetensors", ".woff", ".woff2", ".ttf",
}


@dataclass(frozen=True)
class SecretHit:
    """A single secret-shaped match."""

    path: Path
    line: int
    pattern: str
    snippet: str

    def describe(self, root: Path = ROOT) -> str:
        try:
            rel = self.path.relative_to(root)
        except ValueError:
            rel = self.path
        return f"{rel}:{self.line}: [{self.pattern}] {self.snippet}"


def _redact(match: str) -> str:
    """Redact the middle of a matched token so we never echo a live secret."""

    if len(match) <= 12:
        return match[:2] + "***"
    return f"{match[:6]}...{match[-4:]}"


def _is_allowlisted_path(rel_posix: str, allowlist: tuple[str, ...]) -> bool:
    """Match an allowlist entry as a path *prefix* (component-aligned).

    A bare-substring test would let an entry like ``tests/fixtures/`` waive an
    unrelated file such as ``evil/tests/fixtures_leak.txt``. We instead anchor
    each entry to the start of the repo-relative path and align it on path
    boundaries:

    * a trailing-slash entry (``tests/fixtures/``) matches that directory and
      anything beneath it;
    * a plain entry matches the exact path, any path beneath it, or a filename
      in the same directory that *starts with* the entry's last component
      (so ``tests/test_secret_scan`` covers ``tests/test_secret_scan.py``).
    """
    parts = rel_posix.split("/")
    for fragment in allowlist:
        frag = fragment.rstrip("/")
        if rel_posix in (fragment, frag):
            return True
        if fragment.endswith("/"):
            if rel_posix.startswith(fragment):
                return True
            continue
        if rel_posix.startswith(frag + "/"):
            return True
        frag_parts = frag.split("/")
        if (
            len(parts) == len(frag_parts)
            and parts[:-1] == frag_parts[:-1]
            and parts[-1].startswith(frag_parts[-1])
        ):
            return True
    return False


def scan_text(text: str, patterns: dict[str, re.Pattern[str]] | None = None) -> list[tuple[int, str, str]]:
    """Scan raw text; return ``(line_no, pattern_name, redacted_snippet)`` hits.

    Lines carrying an allowlist marker are skipped. This is the pure, offline,
    unit-testable core used by both the file scanner and the tests.
    """

    pats = patterns or SECRET_PATTERNS
    hits: list[tuple[int, str, str]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if _ALLOW_MARKER_RE.search(line):
            continue
        for name, pattern in pats.items():
            match = pattern.search(line)
            if match:
                hits.append((lineno, name, _redact(match.group(0))))
    return hits


def _tracked_files(root: Path) -> list[Path]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        # Fall back to a filesystem walk when git is unavailable.
        return [p for p in root.rglob("*") if p.is_file()]
    return [root / name for name in out.stdout.split("\0") if name]


def scan_files(
    files: list[Path],
    root: Path = ROOT,
    allowlist: tuple[str, ...] = DEFAULT_ALLOWLIST_PATHS,
) -> list[SecretHit]:
    """Scan the given files and return non-allowlisted secret hits."""

    hits: list[SecretHit] = []
    for path in files:
        if path.suffix.lower() in _BINARY_SUFFIXES:
            continue
        try:
            rel_posix = path.relative_to(root).as_posix()
        except ValueError:
            rel_posix = path.as_posix()
        if _is_allowlisted_path(rel_posix, allowlist):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for lineno, name, snippet in scan_text(text):
            hits.append(SecretHit(path=path, line=lineno, pattern=name, snippet=snippet))
    return hits


def scan_repository(
    root: Path = ROOT,
    allowlist: tuple[str, ...] = DEFAULT_ALLOWLIST_PATHS,
) -> list[SecretHit]:
    """Scan every tracked file in ``root``."""

    return scan_files(_tracked_files(root), root=root, allowlist=allowlist)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Files to scan (default: all git-tracked files).",
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)

    if args.paths:
        files = [p.resolve() for p in args.paths]
        hits = scan_files(files, root=args.root)
    else:
        hits = scan_repository(root=args.root)

    if hits:
        print(f"secret_scan: {len(hits)} potential secret(s) found:", file=sys.stderr)
        for hit in hits:
            print(f"  {hit.describe(args.root)}", file=sys.stderr)
        print(
            "secret_scan: add '# nosecret' to a false positive, or move test "
            "fixtures under tests/fixtures/.",
            file=sys.stderr,
        )
        return 1

    print("secret_scan: OK (no secret-shaped tokens in tracked files)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
