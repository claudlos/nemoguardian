"""Tests for the CI-hardening helper scripts.

Covers the offline logic of the docs link-checker, the secret-pattern scanner
and the package metadata build-check. No network, GPU, secrets or SDKs required.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import build_check, check_doc_links, secret_scan

# --------------------------------------------------------------------------- #
# check_doc_links
# --------------------------------------------------------------------------- #


def test_link_checker_flags_broken_relative_link(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "real.md").write_text("# real\n", encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_text(
        "See [good](docs/real.md) and [bad](docs/missing.md).\n",
        encoding="utf-8",
    )

    broken = check_doc_links.check_file(readme, root=tmp_path)

    assert len(broken) == 1
    assert broken[0].target == "docs/missing.md"
    assert broken[0].line == 1
    assert "docs/missing.md" in broken[0].describe(tmp_path)


def test_link_checker_passes_when_all_resolve(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text("x", encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_text("[a](docs/a.md)\n", encoding="utf-8")

    assert check_doc_links.find_broken_links([readme], root=tmp_path) == []


def test_link_checker_skips_external_and_anchor_links(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        "[web](https://example.com/x) [anchor](#section) [mail](mailto:a@b.co)\n",
        encoding="utf-8",
    )

    assert check_doc_links.check_file(readme, root=tmp_path) == []


def test_link_checker_resolves_absolute_style_and_fragment(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "mod.py").write_text("pass\n", encoding="utf-8")
    readme = tmp_path / "README.md"
    # Absolute-style '/src/..' resolves against root; fragment is stripped.
    readme.write_text("[m](/src/mod.py#L1)\n", encoding="utf-8")

    assert check_doc_links.check_file(readme, root=tmp_path) == []


def test_link_checker_main_returns_nonzero_on_break(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("[x](nope.md)\n", encoding="utf-8")

    rc = check_doc_links.main([str(readme), "--root", str(tmp_path)])

    assert rc == 1
    assert "broken" in capsys.readouterr().err


def test_repo_docs_have_no_broken_links() -> None:
    files = check_doc_links.default_doc_files(check_doc_links.ROOT)
    assert files, "expected README + docs/*.md to exist"
    assert check_doc_links.find_broken_links(files, root=check_doc_links.ROOT) == []


# --------------------------------------------------------------------------- #
# secret_scan
# --------------------------------------------------------------------------- #

# Assembled at runtime so this source line is not itself a secret-shaped literal.
_FAKE_AWS_KEY = "AKIA" + "IOSFODNN7EXAMPLE1"[:16]


def test_secret_scan_detects_fake_aws_key() -> None:
    hits = secret_scan.scan_text(f"aws_key = '{_FAKE_AWS_KEY}'")

    assert len(hits) == 1
    assert hits[0][1] == "aws_access_key_id"
    # The redacted snippet must not echo the full token.
    assert _FAKE_AWS_KEY not in hits[0][2]


def test_secret_scan_detects_private_key_header() -> None:
    hits = secret_scan.scan_text("-----BEGIN RSA PRIVATE KEY-----")

    assert [h[1] for h in hits] == ["private_key_block"]


def test_secret_scan_clean_text_has_no_hits() -> None:
    clean = "NEMOGUARDIAN_API_KEY=nmg_paste_your_key_here\nsafe = 'hello world'\n"

    assert secret_scan.scan_text(clean) == []


def test_secret_scan_respects_inline_allow_marker() -> None:
    line = f"token = '{_FAKE_AWS_KEY}'  # nosecret"

    assert secret_scan.scan_text(line) == []


def test_secret_scan_files_honours_path_allowlist(tmp_path: Path) -> None:
    fixture = tmp_path / "tests" / "fixtures" / "creds.txt"
    fixture.parent.mkdir(parents=True)
    fixture.write_text(_FAKE_AWS_KEY, encoding="utf-8")
    leaked = tmp_path / "config.txt"
    leaked.write_text(_FAKE_AWS_KEY, encoding="utf-8")

    hits = secret_scan.scan_files([fixture, leaked], root=tmp_path)

    # Only the non-fixture file is reported.
    assert [h.path for h in hits] == [leaked]


def test_secret_scan_main_returns_nonzero_on_hit(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    leaked = tmp_path / "leaked.env"
    leaked.write_text(f"KEY={_FAKE_AWS_KEY}\n", encoding="utf-8")

    rc = secret_scan.main([str(leaked), "--root", str(tmp_path)])

    assert rc == 1
    assert "potential secret" in capsys.readouterr().err


def test_secret_scan_main_clean_returns_zero(tmp_path: Path) -> None:
    clean = tmp_path / "clean.txt"
    clean.write_text("nothing to see here\n", encoding="utf-8")

    assert secret_scan.main([str(clean), "--root", str(tmp_path)]) == 0


def test_repo_is_secret_clean() -> None:
    assert secret_scan.scan_repository(secret_scan.ROOT) == []


# --------------------------------------------------------------------------- #
# build_check
# --------------------------------------------------------------------------- #


def test_build_check_repo_metadata_is_valid() -> None:
    report = build_check.validate_metadata(build_check.ROOT)
    assert report.ok, report.problems
    assert report.checked


def test_build_check_flags_missing_pyproject(tmp_path: Path) -> None:
    report = build_check.validate_metadata(tmp_path)
    assert not report.ok
    assert any("pyproject.toml" in p for p in report.problems)


def test_build_check_flags_missing_readme_and_package(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        """
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "demo"
version = "0.0.1"
description = "demo"
requires-python = ">=3.10"
readme = "README.md"

[tool.setuptools.packages.find]
include = ["demo*"]
""".lstrip(),
        encoding="utf-8",
    )

    report = build_check.validate_metadata(tmp_path)

    assert not report.ok
    joined = " ".join(report.problems)
    assert "readme file 'README.md' is missing" in joined
    assert "package include 'demo*' matches no directory" in joined


def test_build_check_main_returns_nonzero_on_bad_root(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = build_check.main(["--root", str(tmp_path)])
    assert rc == 1
    assert "metadata problem" in capsys.readouterr().err


def test_build_check_main_ok_on_repo() -> None:
    assert build_check.main(["--root", str(build_check.ROOT)]) == 0
