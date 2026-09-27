"""Path portability.

Every path in Config must be anchored to PROJECT_ROOT, never to the process
working directory. This is not a style preference: under Render (and Docker, and
cron, and ``python /abs/path/script.py`` from anywhere) the CWD is not the repo
root, so a relative path silently writes somewhere the app never reads. The
symptom is nasty -- the UI reports "index not built" forever while the ingest
exits 0 having written the index to the wrong place.

The original bug was ``chroma_dir = "./chroma_db"``, the one CWD-relative path
among absolute ones. It passed every test for weeks because locally CWD happens
to equal PROJECT_ROOT. These tests deliberately change CWD.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from src.config import CONFIG, PROJECT_ROOT


class TestPathsAreAbsolute:
    @pytest.mark.parametrize(
        "field",
        [
            "chroma_dir",
            "sources_path",
            "documents_dir",
            "manifest_path",
            "cache_dir",
            "raw_html_dir",
        ],
    )
    def test_path_is_absolute(self, field):
        value = getattr(CONFIG, field)
        assert not value.startswith("."), (
            f"{field}={value!r} is relative; it would resolve against the process "
            "CWD, which is not the project root on Render"
        )
        assert Path(value).is_absolute(), f"{field}={value!r} is not absolute"

    @pytest.mark.parametrize(
        "field", ["chroma_dir", "sources_path", "manifest_path", "cache_dir"]
    )
    def test_path_is_inside_the_project(self, field):
        value = Path(getattr(CONFIG, field)).resolve()
        assert value.is_relative_to(PROJECT_ROOT), (
            f"{field} resolves to {value}, outside PROJECT_ROOT {PROJECT_ROOT}"
        )


class TestPathsSurviveACwdChange:
    def test_chroma_dir_does_not_move_when_cwd_changes(self, monkeypatch, tmp_path):
        before = Path(CONFIG.chroma_dir).resolve()
        monkeypatch.chdir(tmp_path)
        from src.config import _load

        assert Path(_load().chroma_dir).resolve() == before

    def test_index_is_reachable_from_an_unrelated_cwd(self, monkeypatch, tmp_path):
        """The real end-to-end version: open the collection from a foreign CWD."""
        from src.ingest.store import open_collection

        count_before = open_collection().count()
        monkeypatch.chdir(tmp_path)
        assert open_collection().count() == count_before
        assert count_before > 0, "fixture needs a built index"


class TestIndexGuardHonoursTheSamePaths:
    def test_ensure_index_sees_the_existing_index(self):
        """scripts/ensure_index.py must take the warm path, not re-ingest.

        If it re-ingests on every boot that is a five-page fetch per restart,
        and on Render it is the difference between a 1-second start and a
        60-second one.
        """
        result = subprocess.run(
            [sys.executable, "scripts/ensure_index.py"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=300,
        )
        assert result.returncode == 0, result.stderr
        assert "index present" in result.stdout, (
            f"expected the warm path, got: {result.stdout.strip()!r}"
        )
