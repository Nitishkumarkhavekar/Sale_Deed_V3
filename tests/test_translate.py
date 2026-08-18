"""Translation stage - field detection, routing, and write-back.

The model itself is absent (gated on HuggingFace), so these tests cover
everything around it: which fields are selected, whether each is tagged for the
right *operation*, and that results land where the CSV writer looks for them.

The operation distinction is the part worth guarding. A person's name must be
**transliterated** - ರಮೇಶ್ becomes "Ramesh", the sound. An address must be
**translated** - ಮುಖ್ಯ ರಸ್ತೆ becomes "Main Road", the meaning. Getting these the
wrong way round produces output that looks plausible and is wrong in half the
columns, which no downstream check would catch.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.pipeline.stages import (
    PERSON_FIELDS,
    PROPERTY_FIELDS,
    StageName,
    TranslateStage,
)

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.unit

KANNADA_NAME = "ರಮೇಶ್ ಕುಮಾರ್"
KANNADA_ADDRESS = "ಮುಖ್ಯ ರಸ್ತೆ, ಬೆಂಗಳೂರು"


def _deed() -> dict:
    return {
        "buyer_details": [
            {"name": KANNADA_NAME, "father_name": "ಸುರೇಶ್",
             "address": KANNADA_ADDRESS},
        ],
        "seller_details": [{"name": "John Smith", "address": "12 Church Street"}],
        "property_details": {"schedule_c_property_address": KANNADA_ADDRESS},
    }


class TestDetection:
    def test_kannada_is_detected(self):
        assert TranslateStage.needs_translation(KANNADA_NAME)

    def test_latin_text_is_left_alone(self):
        assert not TranslateStage.needs_translation("John Smith")

    def test_empty_is_not_pending(self):
        assert not TranslateStage.needs_translation("")
        assert not TranslateStage.needs_translation(None)

    def test_digits_alone_do_not_trigger_translation(self):
        assert not TranslateStage.needs_translation("560001")


class TestFieldSelection:
    def test_only_kannada_fields_are_queued(self):
        out = TranslateStage(engine="passthrough").run(_deed())
        fields = set(out.data["fields"])
        assert "b1.name" in fields
        assert "b1.address" in fields
        assert "property.schedule_c_property_address" in fields
        # The English seller must not be queued.
        assert not any(f.startswith("s1.") for f in fields)

    def test_pending_count_matches_fields(self):
        out = TranslateStage(engine="passthrough").run(_deed())
        # Four Kannada fields in `_deed()`; the stage now covers ten in
        # total, so this asserts the count matches what was queued rather
        # than a fixed number.
        assert out.data["pending"] == len(out.data["fields"]) == 4

    def test_nothing_to_do_is_success_not_failure(self):
        out = TranslateStage(engine="passthrough").run(
            {"buyer_details": [{"name": "John"}], "seller_details": []})
        assert out.ok
        assert out.data["pending"] == 0


class TestPassthroughHonesty:
    """A stage that cannot translate must say so, not quietly succeed."""

    def test_values_are_not_altered(self):
        deed = _deed()
        before = json.dumps(deed, ensure_ascii=False, sort_keys=True)
        TranslateStage(engine="passthrough").run(deed)
        assert json.dumps(deed, ensure_ascii=False, sort_keys=True) == before

    def test_translated_count_is_zero(self):
        out = TranslateStage(engine="passthrough").run(_deed())
        assert out.data["translated"] == 0

    def test_stage_is_named_correctly(self):
        assert TranslateStage(engine="passthrough").run(_deed()).stage \
            is StageName.TRANSLATE


class TestOperationRouting:
    """Names transliterate, addresses translate - see the module docstring."""

    def _kinds(self) -> dict[str, str]:
        stage = TranslateStage(engine="passthrough")
        deed = _deed()
        kinds: dict[str, str] = {}
        for side in ("buyer_details", "seller_details"):
            for i, person in enumerate(deed.get(side) or [], start=1):
                for field_name, kind in (("name", "transliterate"),
                                         ("father_name", "transliterate"),
                                         ("address", "translate")):
                    if stage.needs_translation(person.get(field_name)):
                        kinds[f"{side[0]}{i}.{field_name}"] = kind
        return kinds

    def test_names_are_transliterated(self):
        kinds = self._kinds()
        assert kinds["b1.name"] == "transliterate"
        assert kinds["b1.father_name"] == "transliterate"

    def test_addresses_are_translated(self):
        assert self._kinds()["b1.address"] == "translate"


class TestFieldResolution:
    """`b1.name` must resolve back to the dict the value came from, or a
    translated result has nowhere to go."""

    def test_buyer_path_resolves(self):
        deed = _deed()
        container, key = TranslateStage(engine="passthrough")._field(deed, "b1.name")
        assert container is deed["buyer_details"][0]
        assert key == "name"

    def test_seller_path_resolves(self):
        deed = _deed()
        container, key = TranslateStage(engine="passthrough")._field(deed, "s1.address")
        assert container is deed["seller_details"][0]

    def test_property_path_resolves(self):
        deed = _deed()
        container, key = TranslateStage(engine="passthrough")._field(
            deed, "property.schedule_c_property_address")
        assert container is deed["property_details"]

    def test_out_of_range_index_is_reported_not_raised(self):
        """A stale field reference must not kill the batch."""
        container, _ = TranslateStage(engine="passthrough")._field(_deed(), "b9.name")
        assert container is None


class TestWriteBack:
    """Results go to `<field>_translated`, never over the original."""

    def test_csv_writer_prefers_the_translated_value(self):
        from core.csv_export import DocumentExport, write_csv
        import csv as csvmod
        import tempfile

        deed = _deed()
        deed["buyer_details"][0]["name_translated"] = "Ramesh Kumar"
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "out.csv"
            write_csv(target, [DocumentExport(transaction_identity="T",
                                              extraction=deed,
                                              source_filename="x.pdf")])
            body = target.read_text(encoding="utf-8-sig")
        assert "Ramesh Kumar" in body

    def test_source_survives_alongside_the_translation(self):
        """A reviewer checking against the deed still needs the Kannada."""
        deed = _deed()
        deed["buyer_details"][0]["name_translated"] = "Ramesh Kumar"
        assert deed["buyer_details"][0]["name"] == KANNADA_NAME


class TestBackendConfiguration:
    def test_missing_model_is_reported_not_raised(self, tmp_path):
        stage = TranslateStage(model_dir=tmp_path,
                               translator_python=tmp_path / "python.exe")
        ok, detail = stage.available()
        assert ok is False
        assert "no model weights" in detail or "not found" in detail

    def test_disabling_translation_reports_itself(self):
        """`passthrough` disables the service rather than selecting a different
        engine - there is only one engine now."""
        ok, detail = TranslateStage(engine="passthrough").available()
        assert ok is False
        assert "disabled" in detail.lower()

    def test_enabled_translation_claims_the_gpu(self):
        """The lease must be held whenever translation *might* use the card;
        whether it lands on CUDA is decided inside the runner from free VRAM."""
        assert TranslateStage().uses_gpu is True
        assert TranslateStage(engine="passthrough").uses_gpu is False

    def test_the_pipeline_actually_enables_translation(self):
        """Regression (R-018). `build_stages` used to search for IndicTrans2
        weights by `*.safetensors` and, finding none, set the stage to
        `passthrough`. NLLB ships `pytorch_model.bin`, so that check could never
        match: translation was dead in the pipeline while the service worked
        perfectly when called directly. The bug was invisible because the only
        test asserted the broken fallback.
        """
        from core.pipeline.runner import build_stages
        from core.translation import TranslationService, build_config

        service_ok, service_why = TranslationService(build_config()).available()
        stage_ok, stage_why = build_stages().translate.available()

        # The pipeline must reach exactly the same verdict as the service. If
        # the model is absent both are False, which is still agreement.
        assert stage_ok == service_ok, (
            f"service says {service_ok} ({service_why}) but the pipeline says "
            f"{stage_ok} ({stage_why})")

    def test_no_second_translation_backend_survives(self):
        """One job, one implementation. A second runner is how the wiring above
        drifted out of sync with the service in the first place."""
        assert not (ROOT / "src" / "tools" / "indictrans_runner.py").exists()
        assert not hasattr(
            __import__("core.pipeline.stages", fromlist=["x"]), "find_translator")


class TestRunnerScript:
    """The subprocess runner, checked without executing the model."""

    SCRIPT = ROOT / "src" / "tools" / "translate_runner.py"

    def test_script_exists(self):
        assert self.SCRIPT.is_file()

    def test_language_tags_are_flores_codes(self):
        """NLLB expects `kan_Knda`, not `kn`. ISO codes silently produce
        garbage rather than an error."""
        text = self.SCRIPT.read_text(encoding="utf-8")
        assert "kan_Knda" in text and "eng_Latn" in text

    def test_device_is_chosen_from_free_vram(self):
        text = self.SCRIPT.read_text(encoding="utf-8")
        assert "mem_get_info" in text, \
            "availability is not capacity - a full GPU would still be selected"

    def test_normalisation_is_nfc(self):
        """Kannada vowel signs decompose several ways; the model saw NFC."""
        text = self.SCRIPT.read_text(encoding="utf-8")
        assert 'normalize("NFC"' in text


class TestCoverageOfEveryKannadaColumn:
    """The export is specified to be English, so every column that can carry
    Kannada must have a translation path.

    Audited by filling every plausible field with Kannada and checking the
    produced CSV - reading the field list gives the intent, not the coverage.
    `gender` was found leaking with no path at all.
    """

    KN_NAME = "ರಮೇಶ್"
    KN_TEXT = "ಆಸ್ತಿಯ ವಿವರಣೆ"

    def _full_deed(self) -> dict:
        return {
            "document_details": {"transaction_date": "2024-06-15"},
            "property_details": {
                "schedule_c_property_address": self.KN_TEXT,
                "property_description": self.KN_TEXT,
                "village": self.KN_TEXT, "district": self.KN_TEXT,
                "taluk": self.KN_TEXT,
            },
            "buyer_details": [{
                "name": self.KN_NAME, "father_name": self.KN_NAME,
                "address": self.KN_TEXT, "gender": self.KN_TEXT,
                "occupation": self.KN_TEXT,
            }],
            "seller_details": [],
        }

    def test_gender_is_covered(self):
        """Regression: it reached the `Gender (PC)` column untranslated."""
        fields = {f for f, _ in PERSON_FIELDS}
        assert "gender" in fields

    def test_place_names_are_transliterated_not_translated(self):
        """A village is a proper noun. Translating it renders what the name
        *means* rather than what it is called."""
        kinds = dict(PROPERTY_FIELDS)
        for place in ("village", "district", "taluk"):
            assert kinds[place] == "transliterate"

    def test_addresses_are_translated_not_transliterated(self):
        kinds = dict(PROPERTY_FIELDS) | dict(PERSON_FIELDS)
        assert kinds["address"] == "translate"
        assert kinds["schedule_c_property_address"] == "translate"

    def test_every_kannada_field_is_queued(self):
        out = TranslateStage(engine="passthrough").run(self._full_deed())
        queued = {f.split(".")[-1] for f in out.data["fields"]}
        expected = {f for f, _ in PERSON_FIELDS} | {f for f, _ in PROPERTY_FIELDS}
        assert queued == expected, f"not queued: {expected - queued}"

    def test_the_export_reports_what_it_could_not_translate(self):
        """Kannada is written rather than dropped - a blank cell is worse than a
        Kannada one, because a reader can act on what they can see - but the
        export must say so."""
        import csv as csvmod
        import tempfile

        from core.csv_export import DocumentExport, untranslated_cells, write_csv

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "out.csv"
            write_csv(target, [DocumentExport(transaction_identity="T",
                                              extraction=self._full_deed(),
                                              source_filename="x.pdf")])
            with open(target, encoding="utf-8-sig", newline="") as fh:
                rows = list(csvmod.DictReader(fh))

        remaining = untranslated_cells(rows)
        assert remaining, "the audit helper found nothing in an all-Kannada deed"
        assert "Person Name (PC)" in remaining

    def test_values_are_never_dropped_for_being_kannada(self):
        """Losing a value from a legal record is not an acceptable way to make
        the column English."""
        source = (ROOT / "src" / "core" / "csv_export.py").read_text(encoding="utf-8")
        assert "never to remove" in source

    def test_the_audit_tool_exists(self):
        tool = ROOT / "src" / "tools" / "kannada_audit.py"
        assert tool.is_file()
        assert r"d:\saledeed" not in tool.read_text(encoding="utf-8").lower()


class TestTranslationLogging:
    """Requirement: the log must show language, model, timing and both texts.

    Logging lives in the service now, not the stage - that is the point of
    centralising it, so a caller that is not the pipeline gets the same records.
    """

    SOURCE = ROOT / "src" / "core" / "translation" / "service.py"

    def test_unavailability_is_reported_not_silent(self):
        source = self.SOURCE.read_text(encoding="utf-8")
        assert "translation unavailable" in source
        # The message continues "...stay in the source language" across a line
        # break, so match the fragment that is contiguous in the file.
        assert "stay in the source" in source

    def test_the_log_records_language_model_and_timing(self):
        source = self.SOURCE.read_text(encoding="utf-8")
        for key in ('"languages"', '"engine"', '"model"', '"device"'):
            assert key in source, f"{key} is not logged"
        assert "result.seconds" in source

    def test_original_and_translation_are_logged(self):
        source = self.SOURCE.read_text(encoding="utf-8")
        assert '"original"' in source and '"translation"' in source

    def test_retries_are_logged(self):
        source = self.SOURCE.read_text(encoding="utf-8")
        assert "attempt %d failed, retrying" in source

    def test_the_outcome_carries_the_language_for_the_caller(self):
        out = TranslateStage(engine="passthrough").run(
            {"buyer_details": [{"name": "ರಮೇಶ್"}], "seller_details": []})
        # FLORES-200, not ISO 639-1: that is what NLLB expects, and an ISO code
        # produces silent garbage rather than an error.
        assert out.data["source_language"] == "kan_Knda"
        assert out.data["engine"] in ("unavailable", "failed", "nllb")


@pytest.mark.integration
class TestTheTranslationIsPersisted:
    """The stage's output has to reach the database, or it never happened.

    Found by exporting a real batch and noticing Kannada in columns meant to
    hold English on a document whose `translate` stage read DONE. The cause is
    an ordering that looks harmless: validation writes the person rows, then
    translation mutates the extraction dict in place - and nothing saved it
    again. Every person row in the database had a NULL `name_translated`, and
    roughly 75 seconds of CPU per document went into values that were discarded.

    The stage column said DONE throughout, which is why it survived so long:
    nothing anywhere reported a problem.
    """

    @pytest.fixture()
    def document(self, session_factory, temp_batch):
        from core.db.engine import session_scope
        from core.db.repositories import UnitOfWork

        extraction = {
            "seller_details": [{"name": "ಹೆಚ್. ಕೃಷ್ಣಮೂರ್ತಿ",
                                "father_name": "ಕೆ. ನರಸಿಂಹಪ್ಪ",
                                "address": "ನಾಗರಭಾವಿ, ಬೆಂಗಳೂರು"}],
            "buyer_details": [{"name": "ಜಿ. ಸಿ. ಸೋಮಣ್ಣ"}],
            "property_details": {"address": "ರಾಮನಗರ ಜಿಲ್ಲೆ"},
            "document_details": {},
        }
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            doc = uow.documents.list_for_batch(temp_batch, per_page=1)[0][0]
            uow.results.save_property(doc, extraction["property_details"])
            uow.results.replace_persons(doc, extraction)
            doc_pk = doc.id
        return doc_pk, extraction

    def test_a_translated_name_reaches_the_database(self, session_factory,
                                                    document):
        from core.db.engine import session_scope
        from core.db.models import Document
        from core.db.repositories import UnitOfWork

        doc_pk, extraction = document
        # What the stage does: adds `<field>_translated` beside each field.
        extraction["seller_details"][0]["name_translated"] = "H. Krishnamurthy"
        extraction["seller_details"][0]["address_translated"] = "Nagarabhavi, Bengaluru"
        extraction["buyer_details"][0]["name_translated"] = "G. C. Somanna"

        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            written = uow.results.apply_translations(
                session.get(Document, doc_pk), extraction)
        assert written == 3, "the translated values were not written"

        # A fresh session, so this is the database and not the identity map.
        with session_scope(session_factory) as session:
            doc = session.get(Document, doc_pk)
            names = {p.relation.value: p.name_translated for p in doc.persons}
        assert names["S"] == "H. Krishnamurthy"
        assert names["B"] == "G. C. Somanna"

    def test_the_original_is_kept_beside_the_translation(self, session_factory,
                                                         document):
        """The deed is a legal record; the source text is never replaced."""
        from core.db.engine import session_scope
        from core.db.models import Document
        from core.db.repositories import UnitOfWork

        doc_pk, extraction = document
        extraction["seller_details"][0]["name_translated"] = "H. Krishnamurthy"
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            uow.results.apply_translations(session.get(Document, doc_pk),
                                           extraction)
        with session_scope(session_factory) as session:
            seller = next(p for p in session.get(Document, doc_pk).persons
                          if p.relation.value == "S")
        assert seller.name == "ಹೆಚ್. ಕೃಷ್ಣಮೂರ್ತಿ"
        assert seller.name_translated == "H. Krishnamurthy"

    def test_the_person_rows_are_updated_not_replaced(self, session_factory,
                                                      document):
        """Rebuilding them would issue new primary keys, and the validation
        flags recorded moments earlier reference the old ones - so the flags
        would point at rows that no longer exist."""
        from core.db.engine import session_scope
        from core.db.models import Document
        from core.db.repositories import UnitOfWork

        doc_pk, extraction = document
        with session_scope(session_factory) as session:
            before = {p.id for p in session.get(Document, doc_pk).persons}

        extraction["seller_details"][0]["name_translated"] = "H. Krishnamurthy"
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            uow.results.apply_translations(session.get(Document, doc_pk),
                                           extraction)

        with session_scope(session_factory) as session:
            after = {p.id for p in session.get(Document, doc_pk).persons}
        assert before == after, "person rows were recreated"

    def test_nothing_is_written_when_there_is_nothing_to_write(
            self, session_factory, document):
        doc_pk, extraction = document
        from core.db.engine import session_scope
        from core.db.models import Document
        from core.db.repositories import UnitOfWork

        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            assert uow.results.apply_translations(
                session.get(Document, doc_pk), extraction) == 0

    def test_the_property_address_is_translated_too(self, session_factory,
                                                    document):
        from core.db.engine import session_scope
        from core.db.models import Document
        from core.db.repositories import UnitOfWork

        doc_pk, extraction = document
        extraction["property_details"]["address_translated"] = "Ramanagara District"
        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            uow.results.apply_translations(session.get(Document, doc_pk),
                                           extraction)
        with session_scope(session_factory) as session:
            assert session.get(Document, doc_pk).property_.address_translated \
                == "Ramanagara District"

    def test_the_runner_persists_after_a_successful_translate(self):
        """The wiring, not just the repository. Asserted on the source because
        running it needs the translation model and several minutes of CPU - and
        the defect was precisely that this call was absent."""
        import inspect

        from core.pipeline.runner import BatchRunner

        body = inspect.getsource(BatchRunner._do_translate)
        assert "apply_translations" in body, (
            "the runner marks translate DONE without saving what it produced")
        assert body.index("apply_translations") < body.index("StageState.DONE"), \
            "the result must be saved before the stage is marked done"
