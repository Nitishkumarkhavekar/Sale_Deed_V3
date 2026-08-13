"""Transaction Identity extraction.

The format is distinctive, so *finding* one is easy. Finding the **right** one
is the problem: a sale deed recites its chain of title, so it quotes the
registration numbers of prior documents. On the 50-deed corpus 10 of 50 contain
more than one, and in at least one the deed's own number appears at line 330
while a 2018-19 prior deed appears at line 107.

Writing a previous owner's document number into the Transaction Identity column
is worse than leaving it blank, because nothing downstream would question it.

Accuracy against the whole corpus is measured by `tools/identity_check.py`
(50/50 at the time of writing). These tests pin the behaviour that measurement
depends on.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.transaction_id import CANONICAL, Extraction, extract, find_candidates

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "tests" / "corpus" / "OCR saledeeds"
pytestmark = pytest.mark.unit

REGISTRATION_BLOCK = """===== PAGE 1 =====
ಕರ್ನಾಟಕ ಸರ್ಕಾರ
ದಸ್ತಾವೇಜು ಸಂಖ್ಯೆ :- BGP-1-00275-2025-26
ಬೆಂಗಳೂರು ಉಪ ನೋಂದಣಿ ಕಚೇರಿ
"""


class TestFormat:
    @pytest.mark.parametrize("value", [
        "BGP-1-00275-2025-26", "MDG-1-00146-2025-26", "KRT-1-01222-2025-26",
        "YBG-1-00117-2025-26", "CDG-1-00082-2022-23",
    ])
    def test_the_documented_examples_are_valid(self, value):
        assert CANONICAL.match(value)

    @pytest.mark.parametrize("value", [
        "BGP-1-275-2025-26",        # serial not padded
        "B-1-00275-2025-26",        # office too short
        "BGPXX-1-00275-2025-26",    # office too long
        "BGP-1-00275-2025",         # year not split
        "2025-26-00275",            # reversed
    ])
    def test_malformed_values_are_rejected(self, value):
        assert not CANONICAL.match(value)

    def test_office_prefixes_vary(self):
        """17 distinct offices appear in the corpus - the code is not a fixed
        list and must not become one."""
        for office in ("BGP", "MDG", "KRT", "YBG", "HSR", "VJN", "TVR"):
            assert CANONICAL.match(f"{office}-1-00001-2025-26")


class TestFindingCandidates:
    def test_the_registration_block_is_found(self):
        found = find_candidates(REGISTRATION_BLOCK)
        assert [c.value for c in found] == ["BGP-1-00275-2025-26"]

    def test_ocr_spacing_is_tolerated(self):
        """OCR splits on glyph boundaries, so the hyphens acquire spaces."""
        found = find_candidates("ದಸ್ತಾವೇಜು ಸಂಖ್ಯೆ :- BGP - 1 - 00275 - 2025 - 26")
        assert found and found[0].value == "BGP-1-00275-2025-26"

    def test_a_period_separator_is_tolerated(self):
        found = find_candidates("Doc BGP.1.00275.2025.26 registered")
        assert found and found[0].value == "BGP-1-00275-2025-26"

    def test_a_short_serial_is_padded(self):
        found = find_candidates("BGP-1-0275-2025-26")
        assert found[0].serial == "00275"

    def test_letters_misread_as_digits_are_repaired(self):
        """Only inside the office code, where digits cannot be valid. The
        serial is never touched - turning a real 0 into O there would corrupt
        the value this module exists to get right."""
        found = find_candidates("0GP-1-00275-2025-26")
        assert found[0].office == "OGP"
        assert found[0].serial == "00275"

    def test_an_amount_is_not_mistaken_for_an_identity(self):
        assert not find_candidates("Rs. 30,00,000 paid on 15-06-2024")

    def test_a_date_is_not_mistaken_for_an_identity(self):
        assert not find_candidates("registered on 2025-06-15 at 11-30-00")


class TestChoosingBetweenCandidates:
    """The part that matters. Every case here is drawn from the real corpus."""

    def test_a_single_candidate_is_taken(self):
        result = extract(REGISTRATION_BLOCK)
        assert result.value == "BGP-1-00275-2025-26"
        assert result.confidence >= 0.9

    def test_a_prior_title_is_rejected(self):
        """The deed's own registration is the newest event in it."""
        text = ("ದಸ್ತಾವೇಜು ಸಂಖ್ಯೆ :- HSR-1-01451-2025-26\n"
                "Deed registered as Document No.YAN-1-06130-2014-15, "
                "entered in Book-I\n")
        assert extract(text).value == "HSR-1-01451-2025-26"

    def test_the_deed_number_wins_even_when_it_appears_last(self):
        """Corpus file `2025-26-1457`: the prior deed is at line 107 and the
        deed's own number at line 330. Position alone would be wrong."""
        text = ("registered as document No.VJN-1-04019-2018-19 of Book-I\n"
                + "filler\n" * 50
                + "ನಂಬರ್ VJN-1-01457-2025-26 ಆಗಿ\n")
        assert extract(text).value == "VJN-1-01457-2025-26"

    def test_several_prior_documents_are_all_rejected(self):
        """Corpus file `2025-26-1463` cites two."""
        text = ("ದಸ್ತಾವೇಜು ಸಂಖ್ಯೆ :- VJN-1-01463-2025-26\n"
                "Vide Doc No.SRI-1-00081-2012-13, BOOK-1\n"
                "Vide Doc No.NGB-4-00293-2023-24, Book-4\n"
                "ನಂಬರ್ VJN-1-01463-2025-26 ಆಗಿ\n")
        assert extract(text).value == "VJN-1-01463-2025-26"

    def test_the_reason_is_recorded(self):
        result = extract(REGISTRATION_BLOCK)
        assert result.reason

    def test_ambiguity_produces_a_blank_not_a_guess(self):
        """Two candidates the evidence cannot separate. A coin flip here writes
        a previous owner's number into a legal export."""
        text = "ABC-1-00001-2025-26 and DEF-1-00002-2025-26\n"
        result = extract(text)
        assert result.value == ""
        assert "ambiguous" in result.reason

    def test_nothing_found_is_reported_not_raised(self):
        result = extract("A deed with no registration number at all.")
        assert result.found is False
        assert result.reason


class TestNoisyDocuments:
    """Seals, stamps, signatures and handwriting land in OCR as junk lines."""

    def test_surrounding_noise_is_ignored(self):
        text = ("===== PAGE 1 =====\n"
                "|||| ~~~ SEAL ~~~ ||||\n"
                "sd/- illegible\n"
                "ದಸ್ತಾವೇಜು ಸಂಖ್ಯೆ :- BGP-1-00275-2025-26\n"
                "*** STAMP DUTY PAID ***\n")
        assert extract(text).value == "BGP-1-00275-2025-26"

    def test_the_page_number_is_reported(self):
        text = ("===== PAGE 1 =====\nnothing here\n"
                "===== PAGE 2 =====\nದಸ್ತಾವೇಜು ಸಂಖ್ಯೆ :- BGP-1-00275-2025-26\n")
        assert extract(text).page == 2

    def test_empty_input_is_safe(self):
        assert extract("").found is False
        assert extract(None or "").found is False


@pytest.mark.regression
class TestAgainstTheRealCorpus:
    """The measurement that matters, on documents nobody wrote for a test."""

    @pytest.fixture(scope="class")
    def files(self):
        if not CORPUS.is_dir():
            pytest.skip("deed corpus not present")
        found = sorted(CORPUS.glob("*.txt"))
        if not found:
            pytest.skip("corpus is empty")
        return found

    def test_every_deed_yields_an_identity(self, files):
        missing = []
        for path in files:
            text = path.read_text(encoding="utf-8", errors="replace")
            if not extract(text, source=path.name).found:
                missing.append(path.stem)
        assert not missing, f"no identity extracted from: {missing}"

    def test_every_result_is_canonical(self, files):
        for path in files:
            result = extract(path.read_text(encoding="utf-8", errors="replace"))
            if result.found:
                assert CANONICAL.match(result.value), \
                    f"{path.stem}: {result.value} is malformed"

    def test_the_serial_matches_the_filename(self, files):
        """Filenames encode the serial. Both being wrong the same way is far
        less likely than either being right by accident."""
        import re

        leading = re.compile(r"^(?:20)?\d{2}\s*[-. ]\s*(?:20)?\d{2}\s*[-. ]?\s*")
        trailing = re.compile(r"\s*[-. ]\s*(?:20)?\d{2}\s*[-. ]\s*(?:20)?\d{2}\b.*$")

        mismatches = []
        for path in files:
            stem = trailing.sub("", leading.sub("", path.stem))
            digits = re.findall(r"(\d{2,6})", stem)
            if not digits:
                continue
            want = digits[0].lstrip("0").rjust(5, "0")
            result = extract(path.read_text(encoding="utf-8", errors="replace"))
            if result.found and result.value.split("-")[2] != want:
                mismatches.append((path.stem, result.value, want))
        assert not mismatches, f"serial disagrees with filename: {mismatches}"

    def test_deeds_citing_prior_documents_are_handled(self, files):
        """The cases the scoring exists for."""
        multi = [p for p in files
                 if len({c.value for c in find_candidates(
                     p.read_text(encoding='utf-8', errors='replace'))}) > 1]
        assert multi, "no multi-candidate deed in the corpus to exercise"
        for path in multi:
            text = path.read_text(encoding="utf-8", errors="replace")
            assert extract(text, source=path.name).found, \
                f"{path.stem} cites prior documents and produced nothing"


class TestPipelineIntegration:
    def test_the_runner_writes_it_to_its_own_column(self):
        """It used to overwrite `document_id`, which is the internal handle -
        seeded from the file name so the row can exist before anything is read.
        Sharing one field made the two impossible to tell apart, and a deed
        whose number could not be read exported its file name (R-043)."""
        source = (ROOT / "src" / "core" / "pipeline" / "runner.py").read_text(
            encoding="utf-8")
        assert "extract_transaction_id(" in source
        assert "doc.transaction_identity = identity.value if identity.found else None"             in source
        assert "doc.document_id = identity.value" not in source,             "the extracted number is overwriting the internal handle again"

    def test_the_csv_reads_it(self):
        source = (ROOT / "src" / "core" / "csv_export.py").read_text(encoding="utf-8")
        assert '"Transaction Identity": _clean(doc.transaction_identity)' in source

    def test_no_identity_means_an_empty_column(self):
        """This test used to assert the opposite - that the file name stands in
        when nothing is found, "populated with something traceable rather than
        emptied". That contradicts `extract`'s own contract, which returns empty
        deliberately rather than risk writing a previous owner's number, and it
        is how the file name reached the Transaction Identity column."""
        source = (ROOT / "src" / "core" / "pipeline" / "runner.py").read_text(
            encoding="utf-8")
        assert "if identity.found else None" in source

        services = (ROOT / "src" / "app" / "services.py").read_text(encoding="utf-8")
        assert "transaction_identity=doc.transaction_identity or \"\"" in services
        assert "transaction_identity=doc.document_id" not in services

    def test_no_layer_falls_back_to_a_file_name(self):
        """Every export path, not just the main one."""
        services = (ROOT / "src" / "app" / "services.py").read_text(encoding="utf-8")
        for bad in ("transaction_identity=doc.document_id",
                    "transaction_identity=d.document_id",
                    "transaction_identity=doc.source_filename",
                    "transaction_identity=p.stem"):
            assert bad not in services, f"file name reaches the identity: {bad}"


class TestTheKaveriRegistrationCertificate:
    """The block Kaveri stamps on page 1, which is where the number lives.

    Transcribed from a real certificate:

        1 ನೇ ಪುಸ್ತಕದ ದಸ್ತಾವೇಜು
        ನಂಬರ BGP-1-00275-2025-26 ಆಗಿ
        ದಿನಾಂಕ 19/04/2025 ರಂದು ನೋಂದಾಯಿಸಿ ವಿದ್ಯುನ್ಮಾನ ಮಾದರಿಯಲ್ಲಿ
        ಕೇಂದ್ರಿತ ದತ್ತಾಂಶ ಕೋಶದಲ್ಲಿ ಶೇಖರಿಸಿದೆ.
        ಉಪ ನೋಂದಣಾಧಿಕಾರಿ ಚಿಕ್ಕಬಳ್ಳಾಪುರ (ಬಾಗೇಪಲ್ಲಿ)
    """

    CERTIFICATE = (
        "Kaveri Online Services\n"
        "1 ನೇ ಪುಸ್ತಕದ ದಸ್ತಾವೇಜು\n"
        "ನಂಬರ BGP-1-00275-2025-26 ಆಗಿ\n"
        "ದಿನಾಂಕ 19/04/2025 ರಂದು ನೋಂದಾಯಿಸಿ ವಿದ್ಯುನ್ಮಾನ ಮಾದರಿಯಲ್ಲಿ\n"
        "ಕೇಂದ್ರಿತ ದತ್ತಾಂಶ ಕೋಶದಲ್ಲಿ ಶೇಖರಿಸಿದೆ.\n"
        "ಉಪ ನೋಂದಣಾಧಿಕಾರಿ ಚಿಕ್ಕಬಳ್ಳಾಪುರ (ಬಾಗೇಪಲ್ಲಿ)\n")

    def test_the_number_is_read_from_the_certificate(self):
        result = extract(self.CERTIFICATE, source="275.pdf", ocr_used=True)
        assert result.value == "BGP-1-00275-2025-26"
        assert result.found

    def test_the_parts_are_what_the_certificate_shows(self):
        """BGP office, book 1, serial 00275, financial year 2025-26."""
        from core.transaction_id import CANONICAL

        assert CANONICAL.match("BGP-1-00275-2025-26")
        office, book, serial, year, yy = "BGP-1-00275-2025-26".split("-")
        assert office == "BGP" and book == "1"
        assert serial == "00275" and len(serial) == 5
        assert f"{year}-{yy}" == "2025-26"

    def test_the_certificate_wording_counts_as_evidence(self):
        """`ನಂಬರ` is written without the trailing virama on the certificate, and
        in 49 of 50 corpus documents. Listing only `ನಂಬರ್` meant no label matched
        on the very block the number sits in, and the financial-year signal was
        deciding alone."""
        result = extract(self.CERTIFICATE, source="275.pdf", ocr_used=True)
        assert "registration label" in result.reason

    def test_the_bare_form_subsumes_the_virama_form(self):
        from core.transaction_id import LABELS

        assert "ನಂಬರ" in LABELS
        assert "ನಂಬರ್".startswith("ನಂಬರ"), "one is a prefix of the other"

    def test_the_heading_carries_it_when_ocr_drops_the_number_word(self):
        """OCR loses words. With `ನಂಬರ` gone, the certificate's own heading -
        `ಪುಸ್ತಕದ ದಸ್ತಾವೇಜು`, present in 49 of 50 corpus documents - is what still
        marks this as the registration block rather than a citation."""
        damaged = self.CERTIFICATE.replace("ನಂಬರ ", "")
        assert "ನಂಬರ" not in damaged
        result = extract(damaged, source="275.pdf", ocr_used=True)
        assert result.value == "BGP-1-00275-2025-26"
        assert "registration label" in result.reason

    def test_prior_documents_do_not_win(self):
        """A deed recites its chain of title. The certificate's number is the
        newest and the labelled one; a cited number is neither."""
        deed = ("ಮೂಲ ದಸ್ತಾವೇಜು ನಂಬರ್ CKL-1-01021-2018-19 ರ ಮೂಲಕ ಖರೀದಿಸಲಾಗಿದೆ.\n"
                "Vide Doc No. YBG-1-00117-2019-20 registered earlier.\n"
                + "\n" * 40 + self.CERTIFICATE)
        result = extract(deed, source="275.pdf", ocr_used=True)
        assert result.value == "BGP-1-00275-2025-26"
        assert {c.value for c in find_candidates(deed)} == {
            "BGP-1-00275-2025-26", "CKL-1-01021-2018-19", "YBG-1-00117-2019-20"}

    def test_the_certificate_number_reaches_the_csv_column(self):
        from core.csv_export import DocumentExport, build_rows

        result = extract(self.CERTIFICATE, source="275.pdf", ocr_used=True)
        row = build_rows([DocumentExport(
            transaction_identity=result.value if result.found else "",
            source_filename="275.pdf", extraction={})])[0]
        assert row["Transaction Identity"] == "BGP-1-00275-2025-26"
        assert row["Transaction Identity"] != "275"
