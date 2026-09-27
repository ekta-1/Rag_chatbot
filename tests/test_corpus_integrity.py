"""Regression tests for the corpus-integrity bug found on 2026-09-27.

Groww renders a fund's size twice: once in the authoritative ``__NEXT_DATA__``
payload and the "Fund size (AUM)" stat card, and once in an auto-generated
summary paragraph. The generated sentence carried the AMC's house-level figure
(9,86,237 Cr) on all five scheme pages while ``aum`` sat in
``EXCLUDED_FIELDS``, so the only fund-size string in the corpus was the wrong
one -- retrievable at rank 1, with a valid citation, and wrong on every fund.

These tests pin both halves of the fix: the value must be right, and the stale
string must be unreachable.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from evals.integrity import authoritative_facts, check_all, check_scheme, corpus_facts
from src.config import CONFIG
from src.ingest.fetcher import cache_path_for
from src.ingest.sources import load_sources
from src.ingest.structured import EXCLUDED_FIELDS, _format, _indian_grouping, load_server_data

SCHEMES = [s.scheme_key for s in load_sources()]

# The wrong figure, kept as a literal so a regression names itself in the output.
STALE_HOUSE_AUM = "9,86,237"


class TestIndianGrouping:
    """Groww prints crores with Indian digit grouping, not thousands separators."""

    @pytest.mark.parametrize(
        "raw, expected",
        [
            (39933.3663, "39,933.37"),
            (113606.46602051, "1,13,606.47"),
            (15991.7823, "15,991.78"),
            (41890.8613, "41,890.86"),
            (107295.7919, "1,07,295.79"),
            (100000.0, "1,00,000.00"),
            (999.99, "999.99"),
        ],
    )
    def test_matches_page_formatting(self, raw, expected):
        assert _indian_grouping(raw) == expected

    def test_does_not_double_comma(self):
        # Regression: grouping was applied to an already-comma-formatted string,
        # producing "3,9,,933.37" for every value.
        for raw in (39933.3663, 113606.46602051, 107295.7919):
            assert ",," not in _indian_grouping(raw)


class TestAumIsIngested:
    def test_aum_no_longer_excluded(self):
        assert "aum" not in EXCLUDED_FIELDS, (
            "aum was excluded as a 'performance claim', but fund size is a static "
            "scheme attribute; excluding it left only the wrong generated value"
        )

    @pytest.mark.parametrize("scheme_key", SCHEMES)
    def test_corpus_aum_matches_payload(self, scheme_key):
        truth = authoritative_facts(scheme_key)["Fund size (AUM)"]
        stored = corpus_facts(scheme_key)["Fund size (AUM)"]
        assert stored, f"{scheme_key}: no AUM row in corpus"
        assert truth in stored, f"{scheme_key}: corpus {stored} != payload {truth}"

    @pytest.mark.parametrize("scheme_key", SCHEMES)
    def test_stale_house_aum_absent(self, scheme_key):
        path = Path(CONFIG.documents_dir) / f"{scheme_key}.json"
        text = path.read_text(encoding="utf-8")
        digits = STALE_HOUSE_AUM.replace(",", "")
        assert STALE_HOUSE_AUM not in text and digits not in text, (
            f"{scheme_key}: the AMC house-level AUM is back in the corpus"
        )

    def test_the_five_aums_are_distinct(self):
        # The original bug was one figure reused across five unrelated funds.
        values = [authoritative_facts(k)["Fund size (AUM)"] for k in SCHEMES]
        assert len(set(values)) == len(values), f"AUMs are not distinct: {values}"

    def test_known_large_cap_value(self):
        # Regression anchor: the figure the user corrected on 2026-09-27.
        assert authoritative_facts("large_cap")["Fund size (AUM)"] == "39,933.37 Cr"


class TestGeneratedSummaryIsStripped:
    def test_summary_sentence_removed(self):
        from src.ingest.extractor import _strip_generated_summary

        raw = (
            "HDFC Large Cap Fund Direct Growth is a Equity Mutual Fund Scheme. "
            "The fund currently has an Asset Under Management(AUM) of \u20b99,86,237 Cr "
            "and the Latest NAV as of 25 Sep 2026 is \u20b91,189.08. "
            "The HDFC Large Cap Fund Direct Growth is rated Very High risk."
        )
        out = _strip_generated_summary(raw)
        assert "9,86,237" not in out
        assert "1,189.08" not in out, "NAV must not survive via prose"
        # Surrounding prose is kept -- this is not a blanket paragraph drop.
        assert "rated Very High risk" in out
        assert "is a Equity Mutual Fund Scheme" in out

    def test_pattern_does_not_hardcode_the_wrong_number(self):
        # A pattern encoding "9,86,237" would silently stop working when Groww
        # edits the figure, so the pattern must key off the sentence shape.
        from src.ingest.extractor import GENERATED_SUMMARY_RE

        assert "9,86,237" not in GENERATED_SUMMARY_RE.pattern

    def test_strip_is_idempotent(self):
        from src.ingest.extractor import _strip_generated_summary

        raw = "Intro. The fund currently has an Asset Under Management(AUM) of X Cr and NAV Y. Outro."
        once = _strip_generated_summary(raw)
        assert _strip_generated_summary(once) == once

    def test_no_fragment_left_behind(self):
        # Regression: the sentence contains a NAV whose decimal point ("1,189.08")
        # is not a sentence end. A naive [^.]*? stopped there and left a bare
        # "08." in the corpus, which is how this was first found -- in generated
        # sample Q&A, as "…Direct Growth fund. 82. The HDFC…".
        from src.ingest.extractor import _strip_generated_summary

        raw = (
            "Intro sentence. The fund currently has an Asset Under Management(AUM) of "
            "\u20b99,86,237 Cr and the Latest NAV as of 25 Sep 2026 is \u20b91,189.08. "
            "Outro sentence."
        )
        out = _strip_generated_summary(raw)
        assert out.strip() == "Intro sentence. Outro sentence.", f"residue left: {out!r}"
        assert "08." not in out
        assert "09." not in out

    def test_corpus_has_no_summary_residue(self):
        import re as _re

        for scheme_key in SCHEMES:
            path = Path(CONFIG.documents_dir) / f"{scheme_key}.json"
            doc = json.loads(path.read_text(encoding="utf-8"))
            for block in doc.get("blocks", []):
                text = block.get("text", "")
                assert "Asset Under Management" not in text, (
                    f"{scheme_key}: generated summary survived extraction"
                )
                # A bare decimal fragment left where the sentence used to be.
                assert not _re.search(r"fund\.\s*\d{2}\.", text), (
                    f"{scheme_key}: decimal residue in {text[:120]!r}"
                )


class TestFundManagerResolution:
    """The flat ``fund_manager`` field is stale; ``fund_manager_details`` is not.

    Same family as the AUM bug: a denormalized convenience field was ingested
    instead of the structured source the page actually renders. For large_cap
    the flat field names 'Prashant Jain', who appears nowhere in that scheme's
    management section, and it was stale on four of the five schemes.
    """

    def test_reads_the_rendered_array_not_the_flat_field(self):
        from src.ingest.structured import _fund_manager_value

        data = {
            "fund_manager": "Prashant Jain",
            "fund_manager_details": [
                {"person_name": "Rahul Baijal", "date_from": "2022-07-28T18:30:00.000Z"},
                {"person_name": "Dhruv Muchhal", "date_from": "2023-06-21T18:30:00.000Z"},
            ],
        }
        value = _fund_manager_value(data)
        assert "Prashant Jain" not in value, "stale flat field leaked into the answer"
        assert "Rahul Baijal (since Jul 2022)" in value
        assert "Dhruv Muchhal (since Jun 2023)" in value

    def test_all_managers_are_kept(self):
        # Co-managed schemes are the norm here, not the exception: large_cap has
        # two and balanced_advantage has six. Reporting only the first would
        # understate who actually runs the money.
        from src.ingest.structured import _fund_manager_value

        data = {
            "fund_manager": "Stale Name",
            "fund_manager_details": [
                {"person_name": f"Manager {i}", "date_from": "2022-07-28T18:30:00.000Z"}
                for i in range(6)
            ],
        }
        value = _fund_manager_value(data)
        for i in range(6):
            assert f"Manager {i}" in value

    def test_falls_back_when_array_missing(self):
        from src.ingest.structured import _fund_manager_value

        assert _fund_manager_value({"fund_manager": "Someone"}) is None
        assert _fund_manager_value({"fund_manager_details": []}) is None
        assert _fund_manager_value({"fund_manager_details": [{}, {"person_name": "  "}]}) is None

    def test_duplicate_names_collapsed(self):
        from src.ingest.structured import _fund_manager_value

        data = {
            "fund_manager_details": [
                {"person_name": "Rahul Baijal", "date_from": "2022-07-28T18:30:00.000Z"},
                {"person_name": "Rahul Baijal", "date_from": "2022-07-28T18:30:00.000Z"},
            ]
        }
        assert _fund_manager_value(data).count("Rahul Baijal") == 1

    def test_corpus_manager_matches_page(self):
        for scheme_key in SCHEMES:
            truth = authoritative_facts(scheme_key)["Fund manager"]
            stored = corpus_facts(scheme_key)["Fund manager"]
            assert truth in stored, f"{scheme_key}: {stored} != {truth}"

    @pytest.mark.parametrize("scheme_key", SCHEMES)
    def test_no_stale_flat_name_in_corpus(self, scheme_key):
        from src.ingest.structured import _fund_manager_value, load_server_data

        path = Path(CONFIG.documents_dir) / f"{scheme_key}.json"
        text = path.read_text(encoding="utf-8")
        data = load_server_data(cache_path_for(scheme_key).read_text(encoding="utf-8", errors="replace"))
        flat = data.get("fund_manager")
        resolved = _fund_manager_value(data)
        if flat and resolved and flat not in resolved:
            assert flat not in text, f"{scheme_key}: stale flat manager {flat!r} still in corpus"

    def test_known_large_cap_managers(self):
        # Regression anchor for the question the user asked on 2026-09-27.
        value = authoritative_facts("large_cap")["Fund manager"]
        assert "Rahul Baijal" in value
        assert "Dhruv Muchhal" in value
        assert "Prashant Jain" not in value

    def test_month_year_formatting(self):
        from src.ingest.structured import _month_year

        assert _month_year("2022-07-28T18:30:00.000Z") == "Jul 2022"
        assert _month_year("2026-01-31T18:30:00.000Z") == "Jan 2026"
        assert _month_year("2023-12-01") == "Dec 2023"
        assert _month_year(None) is None
        assert _month_year("not-a-date") is None
        assert _month_year("2022-99-01") is None


class TestIntegrityGuard:
    def test_current_corpus_passes(self):
        assert check_all() == []

    @pytest.mark.parametrize("scheme_key", SCHEMES)
    def test_each_scheme_passes(self, scheme_key):
        assert check_scheme(scheme_key) == []

    def test_guard_catches_missing_fact(self, tmp_path):
        # Prove the guard fails when a fact the page states is absent, which is
        # precisely the original bug.
        issues = check_scheme("large_cap", documents_dir=tmp_path)
        assert any(i.kind == "missing" and i.fact == "Fund size (AUM)" for i in issues)

    def test_guard_catches_conflicting_values(self, tmp_path):
        # A corpus holding two different values for one label must be reported,
        # since the user cannot tell which is authoritative.
        (tmp_path / "large_cap.json").write_text(
            json.dumps(
                {
                    "blocks": [
                        {"text": "Fund size (AUM) | 39,933.37 Cr"},
                        {"text": "Fund size (AUM) | 9,86,237 Cr"},
                    ]
                }
            ),
            encoding="utf-8",
        )
        issues = check_scheme("large_cap", documents_dir=tmp_path)
        assert any(i.kind == "conflicting" and i.fact == "Fund size (AUM)" for i in issues)

    def test_norm_ignores_formatting_but_not_value(self):
        from evals.integrity import _norm

        assert _norm("\u20b939,933.37 Cr") == _norm("39,933.37 Cr")
        assert _norm("\u20b99,86,237 Cr") != _norm("39,933.37 Cr")
