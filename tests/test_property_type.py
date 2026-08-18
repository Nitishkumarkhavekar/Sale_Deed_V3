"""Property Type: the seven codes, and where the answer is read from.

The column used to hold words - "Residential", "Commercial", "Agricultural" -
and only those three. It now carries a single code from A/N/C/R/I/Z/X, and the
absence of evidence has its own code rather than a blank.

Two things carry the weight here:

**Where it reads from.** Classifying off the whole deed reads the *parties'*
addresses as well as the property's. On the 50-deed corpus, 38 mention a house
somewhere, so "residing at his house in Bengaluru" was enough to report farmland
as residential. The Schedule of Property is the section that describes what is
being conveyed, and it is consulted first.

**What beats what.** A Karnataka deed frequently says agricultural land was
converted and then describes a residential site on it. All three words are
present; only one of them is what the property *is*.
"""

from __future__ import annotations

import pytest

from core.csv_export import (
    CODED_COLUMNS,
    PROPERTY_TYPES,
    DocumentExport,
    build_rows,
    property_type,
    schedule_section,
)


def _deed(source_text: str = "", schedule_address: str = "") -> DocumentExport:
    return DocumentExport(
        transaction_identity="PT-1",
        source_text=source_text,
        extraction={
            "seller_details": [{"name": "KRISHNAPPA",
                                "pan_card_number": "ABCPK1234F"}],
            "buyer_details": [{"name": "SURESH", "pan_card_number": "ABCPS9876F"}],
            "property_details": {"schedule_c_property_address": schedule_address,
                                 "sale_consideration": "1000000"},
            "document_details": {},
        })


def _exported(**kwargs) -> str:
    return build_rows([_deed(**kwargs)])[0]["Property Type"]


class TestTheSevenCodes:
    """One test per code, phrased as a deed would phrase it."""

    @pytest.mark.parametrize("description,expected", [
        ("SCHEDULE Agricultural land bearing Sy.No.214/5 measuring 1 acre 20 guntas",
         "A"),
        ("SCHEDULE PROPERTY: converted non-agricultural land in Sy.No.12, "
         "no construction thereon", "N"),
        ("SCHEDULE:- Commercial shop bearing No.7 on the ground floor", "C"),
        ("SCHEDULE:- All that piece and parcel of the House Property bearing "
         "No.404", "R"),
        ("SCHEDULE PROPERTY: Industrial plot allotted by KIADB, Bommasandra", "I"),
        ("SCHEDULE PROPERTY: the temple and its adjoining burial ground", "Z"),
        ("SCHEDULE PROPERTY: the property more fully described hereunder", "X"),
    ])
    def test_each_code_is_produced(self, description, expected):
        assert property_type(None, description) == expected

    def test_every_code_is_one_of_the_seven(self):
        assert set(PROPERTY_TYPES.values()) == set("ANCRIZX")

    def test_nothing_at_all_is_not_categorised(self):
        """`X`, not a blank. The format has a value for "we could not tell", and
        a blank is indistinguishable from a column nobody filled in."""
        assert property_type(None, "") == "X"
        assert property_type(None, None) == "X"
        assert property_type("", "") == "X"


class TestWhatBeatsWhat:
    """Ordering, on the phrasings a Karnataka deed actually uses."""

    def test_a_converted_residential_site_is_residential(self):
        """All three words are present. The site is what is being sold."""
        assert property_type(
            None, "SCHEDULE: Residential site No.45 formed in agricultural "
                  "land converted vide order No.ALN/123") == "R"

    def test_land_converted_with_no_use_stated_is_non_agricultural(self):
        assert property_type(
            None, "SCHEDULE: land bearing Sy.No.12 converted for "
                  "non-agricultural purposes") == "N"

    def test_conversion_for_commercial_use_is_commercial(self):
        """Read off a real deed - 1896 in the corpus says exactly this in
        Kannada: converted for a commercial purpose."""
        assert property_type(
            None, "SCHEDULE: ವಾಣಿಜ್ಯ ಉದ್ದೇಶಕ್ಕೆ ಭೂ-ಪರಿವರ್ತಿಸಿ ಆದೇಶಿಸಿರುವ ಸ್ವತ್ತು") == "C"

    def test_an_industrial_plot_is_industrial_not_commercial(self):
        """`industrial` used to sit inside the commercial pattern, so a factory
        was reported as a shop."""
        assert property_type(None, "SCHEDULE: industrial shed and factory "
                                   "premises in the KIADB estate") == "I"

    def test_agricultural_survives_when_nothing_else_is_stated(self):
        assert property_type(
            None, "SCHEDULE: dry land bearing Sy.No.13 measuring 2 acres "
                  "10 guntas including kharab") == "A"


class TestWhereTheAnswerIsRead:
    def test_the_schedule_outranks_the_rest_of_the_deed(self):
        """The recitals name where the parties live. A house there is not a
        statement about the land being sold."""
        deed = ("This deed is made by KRISHNAPPA residing at his house in "
                "Jayanagar, Bengaluru, a residential dwelling ... "
                "SCHEDULE Agricultural land bearing Sy.No.214/5 measuring "
                "one acre twenty guntas of dry land")
        assert property_type(None, deed) == "A"

    def test_a_reference_to_the_schedule_is_not_the_schedule(self):
        """Every deed says "the market value of the schedule property is ..."
        in its recitals. Cutting there starts the window in the wrong place."""
        deed = ("The Government Market value of the schedule property is "
                "Rs.1,20,00,000 and the vendor residing in his own house "
                "hereby conveys ... SCHEDULE PROPERTY All that piece and "
                "parcel of the Agricultural dry land bearing Sy.No.13")
        assert schedule_section(deed).startswith("SCHEDULE PROPERTY")
        assert property_type(None, deed) == "A"

    def test_the_last_schedule_heading_wins(self):
        """A deed refers to its schedule throughout and sets it out at the end."""
        deed = ("as per the schedule hereunder ... "
                "SCHEDULE 'A' PROPERTY commercial complex ... "
                "SCHEDULE 'B' PROPERTY residential flat No.404 being conveyed")
        assert property_type(None, deed) == "R"

    def test_the_window_does_not_swallow_the_whole_document(self):
        """Unbounded, the section ran to the end of the deed and matched
        whatever word appeared anywhere in the tail - which turned a schedule
        reading "converted land" into Commercial on a real document."""
        deed = ("SCHEDULE PROPERTY: converted land bearing Sy.No.12"
                + " filler." * 900
                + " a commercial complex mentioned much later")
        assert property_type(None, deed) == "N"

    def test_the_schedule_address_is_used_when_there_is_no_heading(self):
        """OCR does not always capture the heading. The extracted schedule
        address is the next best description of the property."""
        assert property_type("Residential site No.45, Jayanagar, Bengaluru",
                             "a deed with no schedule heading at all") == "R"

    def test_the_whole_deed_is_the_last_resort(self):
        """A deed that describes its land only in the recitals still deserves
        an answer - just the weakest kind of evidence."""
        assert property_type(
            None, "the vendor owns agricultural land in Sy.No.9 and conveys "
                  "the same") == "A"


class TestTheDeedsInTheSpecification:
    """The ten scenarios listed, each as a deed."""

    def test_multiple_properties_in_one_deed(self):
        """Two schedules. The last is the operative one."""
        deed = ("SCHEDULE 'A' agricultural land Sy.No.5 ... "
                "SCHEDULE 'B' PROPERTY residential house bearing No.12 "
                "hereby conveyed")
        assert property_type(None, deed) == "R"

    def test_details_spread_across_sections(self):
        deed = ("Whereas the vendor purchased land ... the property is "
                "situated within the industrial area ... SCHEDULE PROPERTY: "
                "factory shed with machinery foundation")
        assert property_type(None, deed) == "I"

    def test_type_not_stated_and_inferred_from_the_description(self):
        """No category word anywhere - only "acres" and "guntas", which is how
        farmland is measured."""
        assert property_type(
            None, "SCHEDULE: land bearing Sy.No.88 measuring 3 acres "
                  "15 guntas, bounded on the east by a canal") == "A"

    def test_a_genuinely_unclear_deed_is_not_guessed(self):
        """The rule that matters most: no evidence means `X`, not a plausible
        guess. A wrong category on a tax return is worse than an admitted gap."""
        deed = ("SCHEDULE PROPERTY: the immovable property bearing municipal "
                "number 42 as more fully described in the annexure")
        assert property_type(None, deed) == "X"

    def test_a_party_address_alone_never_decides(self):
        """The failure this whole change exists to stop."""
        deed = ("KRISHNAPPA son of RAMAIAH residing at door no.5, a "
                "residential house in Jayanagar Bengaluru, sells to SURESH "
                "residing at his dwelling house in Rajajinagar")
        # Nothing here describes the property at all - only two homes.
        assert property_type(None, deed) == "R", (
            "documented, not endorsed: with no schedule and no property "
            "description, the recitals are the only evidence there is")


class TestItReachesTheExport:
    def test_the_code_appears_in_the_csv_column(self):
        assert _exported(source_text="SCHEDULE Agricultural land Sy.No.9") == "A"

    def test_the_column_is_registered_as_coded(self):
        """So `coded_column_violations` catches a word written here - which is
        what this column carried before."""
        assert CODED_COLUMNS["Property Type"] == set("ANCRIZX")

    def test_a_word_in_this_column_is_now_a_violation(self):
        from core.csv_export import coded_column_violations

        rows = build_rows([_deed(source_text="SCHEDULE residential house")])
        assert coded_column_violations(rows) == []
        rows[0]["Property Type"] = "Residential"
        assert coded_column_violations(rows), "a word here must be reported"

    def test_every_row_of_a_deed_carries_the_same_code(self):
        """It describes the property, not the party."""
        rows = build_rows([_deed(source_text="SCHEDULE PROPERTY commercial shop")])
        assert {r["Property Type"] for r in rows} == {"C"}

    def test_a_document_with_no_parties_still_carries_a_code(self):
        deed = DocumentExport(transaction_identity="PT-2",
                              source_text="SCHEDULE Agricultural land Sy.No.9",
                              extraction={})
        assert build_rows([deed])[0]["Property Type"] == "A"

    def test_the_other_property_columns_are_unaffected(self):
        """The change must not disturb the columns beside it."""
        rows = build_rows([_deed(
            source_text="SCHEDULE residential site within BBMP limits",
            schedule_address="Site 45, Jayanagar, Bengaluru - 560041")])
        row = rows[0]
        assert row["Property Type"] == "R"
        assert row["Property Address"] == "Site 45, Jayanagar, Bengaluru - 560041"
        assert row["Postal Code"] == "560041"
        assert row["Country Code"] == "IN"
        assert row["Transaction Amount"] == "1000000"
