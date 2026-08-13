"""The quantisation accuracy baseline.

What this guards: `Q4_K_M` is served because a 4 GB card cannot hold the f16
weights, and `profiles.py` has always warned that the quantisation is lossy and
that a quantised KV cache can damage exact-copy fields. The comparison below is
how that warning gets answered with a number.

The distinction the tests exist to protect is between a *wrong word* and a
*wrong identifier*. A differently phrased address is a difference; a
one-digit-different Aadhaar attributes a transaction to another person.
"""

from __future__ import annotations

import json

import pytest

from tools.baseline_check import EXACT_FIELDS, TEXT_FIELDS, _fields, compare


def _write(path, model, extraction, document="D1"):
    path.write_text(json.dumps({
        "model": model,
        "results": [{"document": document, "extraction": extraction}],
    }), encoding="utf-8")
    return path


DEED = {
    "seller_details": [{"name": "KRISHNAPPA", "father_name": "Ramaiah",
                        "pan_card_number": "ABCPK1234F",
                        "aadhaar_number": "663212345678",
                        "address": "Site 45, Anekal"}],
    "buyer_details": [{"name": "RAMESH", "aadhaar_number": "551298765432"}],
    "property_details": {"sale_consideration": "4500000",
                         "schedule_c_property_address": "Site 45, Anekal"},
    "document_details": {"transaction_date": "2025-06-14"},
}


class TestFlattening:
    def test_every_party_and_section_is_reachable(self):
        flat = _fields(DEED)
        assert flat["seller_details[0].pan_card_number"] == "ABCPK1234F"
        assert flat["buyer_details[0].aadhaar_number"] == "551298765432"
        assert flat["property_details.sale_consideration"] == "4500000"
        assert flat["document_details.transaction_date"] == "2025-06-14"

    def test_parties_are_indexed_so_two_sellers_do_not_collide(self):
        deed = {"seller_details": [{"name": "A"}, {"name": "B"}]}
        flat = _fields(deed)
        assert flat["seller_details[0].name"] == "A"
        assert flat["seller_details[1].name"] == "B"

    def test_an_unparseable_extraction_flattens_to_nothing(self):
        assert _fields(None) == {}


class TestTheIdentifierDistinction:
    def test_identifiers_are_classed_as_exact_copy(self):
        """These are the fields the KV-cache warning names."""
        assert "aadhaar_number" in EXACT_FIELDS
        assert "pan_card_number" in EXACT_FIELDS
        assert "sale_consideration" in EXACT_FIELDS
        assert "name" not in EXACT_FIELDS      # judged, but not as misattribution
        assert "name" in TEXT_FIELDS

    def test_identical_runs_agree_and_pass(self, tmp_path, capsys):
        _write(tmp_path / "b.json", "f16", DEED)
        _write(tmp_path / "c.json", "Q4_K_M", DEED)
        code = compare(tmp_path / "b.json", tmp_path / "c.json")
        assert code == 0
        assert "No exact-copy field disagreed" in capsys.readouterr().out

    def test_one_wrong_aadhaar_digit_fails_the_check(self, tmp_path, capsys):
        """The whole point. A near-miss here is a different person."""
        damaged = json.loads(json.dumps(DEED))
        damaged["seller_details"][0]["aadhaar_number"] = "663212345679"
        _write(tmp_path / "b.json", "f16", DEED)
        _write(tmp_path / "c.json", "Q4_K_M", damaged)

        code = compare(tmp_path / "b.json", tmp_path / "c.json")
        assert code == 1, "a corrupted Aadhaar passed the baseline"
        out = capsys.readouterr().out
        assert "aadhaar_number" in out
        assert "663212345679" in out

    def test_a_wrong_consideration_fails_the_check(self, tmp_path):
        damaged = json.loads(json.dumps(DEED))
        damaged["property_details"]["sale_consideration"] = "450000"
        _write(tmp_path / "b.json", "f16", DEED)
        _write(tmp_path / "c.json", "Q4_K_M", damaged)
        assert compare(tmp_path / "b.json", tmp_path / "c.json") == 1

    def test_a_differently_worded_address_does_not_fail_the_check(
            self, tmp_path, capsys):
        """Wording is reported in the text-field rate, but it does not
        misattribute anything, so it must not gate a release."""
        reworded = json.loads(json.dumps(DEED))
        reworded["seller_details"][0]["address"] = "Site No 45, Anekal Taluk"
        _write(tmp_path / "b.json", "f16", DEED)
        _write(tmp_path / "c.json", "Q4_K_M", reworded)

        assert compare(tmp_path / "b.json", tmp_path / "c.json") == 0
        assert "100.00%" not in capsys.readouterr().out.split("text fields")[1][:20]

    def test_a_document_missing_from_the_candidate_is_reported(self, tmp_path,
                                                               capsys):
        """Silently scoring 100% on a run that skipped half the sample would be
        the worst possible failure of a baseline."""
        _write(tmp_path / "b.json", "f16", DEED, document="D1")
        _write(tmp_path / "c.json", "Q4_K_M", DEED, document="D2")
        compare(tmp_path / "b.json", tmp_path / "c.json")
        assert "absent from the candidate run" in capsys.readouterr().out

    def test_an_unparseable_candidate_counts_as_disagreement(self, tmp_path):
        """A model that returned nothing has not agreed with the baseline."""
        _write(tmp_path / "b.json", "f16", DEED)
        _write(tmp_path / "c.json", "Q4_K_M", None)
        assert compare(tmp_path / "b.json", tmp_path / "c.json") == 1
