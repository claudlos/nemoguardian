"""Package build / metadata validation guard.

This backs the ``make build-check`` target. Its job is to catch packaging
regressions (a missing ``readme``/``license`` file, an unparseable
``pyproject.toml``, a package that will not be discovered by setuptools) *before*
they reach a real ``python -m build`` / PyPI upload.

By default it runs a fast, offline metadata validation that needs no network and
no extra build dependencies. When the optional ``build`` package is installed you
can pass ``--full`` to additionally invoke ``python -m build`` for a real sdist +
wheel build.

Exit code is ``0`` on success and ``1`` when validation fails.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PROJECT_FIELDS = ("name", "version", "description", "requires-python")


@dataclass
class MetadataReport:
    """Outcome of the offline metadata validation."""

    problems: list[str] = field(default_factory=list)
    checked: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def validate_metadata(root: Path = ROOT) -> MetadataReport:
    """Validate ``pyproject.toml`` metadata and referenced files, offline."""

    report = MetadataReport()
    pyproject = root / "pyproject.toml"
    if not pyproject.exists():
        report.problems.append("pyproject.toml is missing")
        return report

    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, OSError) as exc:
        report.problems.append(f"pyproject.toml is not parseable: {exc}")
        return report
    report.checked.append("pyproject.toml parses")

    build_system = data.get("build-system", {})
    if not build_system.get("requires"):
        report.problems.append("[build-system] requires is missing/empty")
    if not build_system.get("build-backend"):
        report.problems.append("[build-system] build-backend is missing")

    project = data.get("project")
    if not isinstance(project, dict):
        report.problems.append("[project] table is missing")
        return report

    for key in REQUIRED_PROJECT_FIELDS:
        if not project.get(key):
            report.problems.append(f"[project].{key} is missing/empty")
        else:
            report.checked.append(f"[project].{key} present")

    # readme file referenced by [project].readme must exist.
    readme = project.get("readme")
    readme_name = readme if isinstance(readme, str) else (readme or {}).get("file")
    if readme_name:
        if (root / readme_name).exists():
            report.checked.append(f"readme file '{readme_name}' exists")
        else:
            report.problems.append(f"readme file '{readme_name}' is missing")

    # license file(s) referenced by [project].license and license-files must exist.
    license_field = project.get("license")
    if isinstance(license_field, dict) and license_field.get("file"):
        lic = license_field["file"]
        if (root / lic).exists():
            report.checked.append(f"license file '{lic}' exists")
        else:
            report.problems.append(f"license file '{lic}' is missing")

    setuptools_cfg = data.get("tool", {}).get("setuptools", {})
    for lic in setuptools_cfg.get("license-files", []) or []:
        if (root / lic).exists():
            report.checked.append(f"license-files entry '{lic}' exists")
        else:
            report.problems.append(f"license-files entry '{lic}' is missing")

    # At least one discoverable package directory for the include globs.
    find_cfg = setuptools_cfg.get("packages", {})
    if isinstance(find_cfg, dict):
        find = find_cfg.get("find", {})
        includes = find.get("include", []) if isinstance(find, dict) else []
        for inc in includes:
            base = inc.rstrip("*").rstrip(".")
            if base and (root / base).is_dir():
                report.checked.append(f"package '{base}' discoverable")
            elif base:
                report.problems.append(f"package include '{inc}' matches no directory")

    return report


def run_full_build(root: Path = ROOT) -> int:
    """Invoke ``python -m build`` when the ``build`` package is available."""

    probe = subprocess.run(
        [sys.executable, "-c", "import build"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        print(
            "build_check: --full requested but the 'build' package is not "
            "installed (pip install build); skipping sdist/wheel build.",
            file=sys.stderr,
        )
        return 1
    print("build_check: running 'python -m build'...")
    completed = subprocess.run([sys.executable, "-m", "build"], cwd=root)
    return completed.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Also run 'python -m build' (requires the 'build' package).",
    )
    args = parser.parse_args(argv)

    report = validate_metadata(args.root)
    for note in report.checked:
        print(f"build_check: ok - {note}")
    if not report.ok:
        print(f"build_check: {len(report.problems)} metadata problem(s):", file=sys.stderr)
        for problem in report.problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    print("build_check: metadata OK")
    if args.full:
        return run_full_build(args.root)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
