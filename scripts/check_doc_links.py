"""Validate that internal markdown links resolve to real files.

This is an offline CI guard: it walks ``README.md`` and every ``docs/*.md`` file,
extracts inline markdown links (``[text](target)``), and confirms that each
*internal* (relative / same-repo) target points at a file that actually exists.
External links (``http``/``https``/``mailto:`` etc.), pure in-page anchors
(``#section``) and empty targets are skipped -- verifying those would require
network access or a full markdown-heading model, which is out of scope for a
fast, deterministic link guard.

Exit code is ``0`` when every internal link resolves and ``1`` when at least one
broken link is found (the broken links are printed to stderr).

Run directly::

    python scripts/check_doc_links.py

or via ``make link-check``.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Inline markdown links: [label](target). The target capture stops at the first
# closing paren or whitespace so that "(target 'title')" forms degrade cleanly.
_LINK_RE = re.compile(r"\[[^\]]*\]\(\s*(<[^>]+>|[^)\s]+)")

# Schemes that point off-repo; we do not try to resolve these offline.
_EXTERNAL_SCHEMES = ("http://", "https://", "mailto:", "tel:", "ftp://")


@dataclass(frozen=True)
class BrokenLink:
    """A single internal markdown link that failed to resolve."""

    source: Path
    line: int
    target: str
    resolved: Path

    def describe(self, root: Path = ROOT) -> str:
        try:
            src = self.source.relative_to(root)
        except ValueError:
            src = self.source
        return f"{src}:{self.line}: broken link -> {self.target!r} (expected {self.resolved})"


def _is_external(target: str) -> bool:
    return target.lower().startswith(_EXTERNAL_SCHEMES)


def _strip_target(raw: str) -> str:
    """Normalise a captured link target (angle-bracket + fragment handling)."""

    target = raw.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1].strip()
    return target


def iter_link_targets(text: str) -> list[tuple[int, str]]:
    """Yield ``(line_number, raw_target)`` for every inline link in ``text``."""

    targets: list[tuple[int, str]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for match in _LINK_RE.finditer(line):
            targets.append((lineno, match.group(1)))
    return targets


def check_file(path: Path, root: Path = ROOT) -> list[BrokenLink]:
    """Return the broken internal links found in a single markdown file."""

    broken: list[BrokenLink] = []
    text = path.read_text(encoding="utf-8")
    for lineno, raw in iter_link_targets(text):
        target = _strip_target(raw)
        if not target or target.startswith("#") or _is_external(target):
            continue
        # Drop any in-page fragment or query suffix: docs/FOO.md#section -> docs/FOO.md
        path_part = target.split("#", 1)[0].split("?", 1)[0]
        if not path_part:
            continue
        if path_part.startswith("/"):
            resolved = (root / path_part.lstrip("/")).resolve()
        else:
            resolved = (path.parent / path_part).resolve()
        if not resolved.exists():
            broken.append(BrokenLink(source=path, line=lineno, target=target, resolved=resolved))
    return broken


def default_doc_files(root: Path = ROOT) -> list[Path]:
    """The README plus every ``docs/*.md`` file, sorted for stable output."""

    files: list[Path] = []
    readme = root / "README.md"
    if readme.exists():
        files.append(readme)
    docs_dir = root / "docs"
    if docs_dir.is_dir():
        files.extend(sorted(docs_dir.glob("*.md")))
    return files


def find_broken_links(files: list[Path], root: Path = ROOT) -> list[BrokenLink]:
    """Aggregate broken internal links across ``files``."""

    broken: list[BrokenLink] = []
    for path in files:
        broken.extend(check_file(path, root=root))
    return broken


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Markdown files to check (default: README.md + docs/*.md).",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Repository root used to resolve absolute-style '/foo' links.",
    )
    args = parser.parse_args(argv)

    files = [p.resolve() for p in args.paths] if args.paths else default_doc_files(args.root)
    if not files:
        print("check_doc_links: no markdown files to check", file=sys.stderr)
        return 0

    broken = find_broken_links(files, root=args.root)
    if broken:
        print(f"check_doc_links: {len(broken)} broken internal link(s):", file=sys.stderr)
        for link in broken:
            print(f"  {link.describe(args.root)}", file=sys.stderr)
        return 1

    print(f"check_doc_links: OK ({len(files)} file(s), all internal links resolve)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
