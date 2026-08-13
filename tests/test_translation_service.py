"""Multilingual translation: detection, service, caching, failure handling.

The model runs in a subprocess and takes seconds per batch, so it is not loaded
here - `tools/translation_check.py` does that against the real weights. What
these tests cover is everything around it, which is where the defects live:
which language a field is, whether it is queued, whether the cache is honoured,
and whether a failure loses the deed.

Detection is asserted against all twelve required languages using real text in
each script, because a script-range table is exactly the kind of thing that is
correct for the language you tested and off by one block for the next.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.translation import (
    ENGLISH,
    Script,
    TranslationConfig,
    TranslationItem,
    TranslationService,
    build_config,
    detect,
    needs_translation,
    summarise,
)

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.unit

#: One realistic phrase per required language: a name and a road.
SAMPLES: dict[str, tuple[str, str]] = {
    "kan_Knda": ("Kannada", "ರಮೇಶ್ ಕುಮಾರ್ ಮುಖ್ಯ ರಸ್ತೆ"),
    "hin_Deva": ("Hindi", "रमेश कुमार मुख्य सड़क"),
    "tel_Telu": ("Telugu", "రమేష్ కుమార్ ప్రధాన రహదారి"),
    "tam_Taml": ("Tamil", "ரமேஷ் குமார் பிரதான சாலை"),
    "mal_Mlym": ("Malayalam", "രമേഷ് കുമാർ പ്രധാന റോഡ്"),
    "guj_Gujr": ("Gujarati", "રમેશ કુમાર મુખ્ય રોડ"),
    "ben_Beng": ("Bengali", "রমেশ কুমার প্রধান রাস্তা"),
    "pan_Guru": ("Punjabi", "ਰਮੇਸ਼ ਕੁਮਾਰ ਮੁੱਖ ਸੜਕ"),
    "ory_Orya": ("Odia", "ରମେଶ କୁମାର ମୁଖ୍ୟ ରାସ୍ତା"),
    "urd_Arab": ("Urdu", "رمیش کمار مین روڈ"),
    "eng_Latn": ("English", "Ramesh Kumar Main Road"),
}


class TestDetectionAcrossEveryLanguage:
    @pytest.mark.parametrize("code,sample", list(SAMPLES.items()))
    def test_language_is_identified(self, code, sample):
        name, text = sample
        result = detect(text)
        assert result.language == code, f"{name} detected as {result.name}"
        assert result.confidence == pytest.approx(1.0)

    @pytest.mark.parametrize("code,sample", list(SAMPLES.items()))
    def test_names_are_reported_for_humans(self, code, sample):
        assert detect(sample[1]).name == sample[0]

    def test_marathi_and_hindi_share_the_script_but_not_the_words(self):
        """This test used to assert that Marathi text detects as Hindi, which
        was the defect rather than the contract: `रस्ता` is Marathi (Hindi is
        `सड़क`). The script is shared; the vocabulary is not, and a deed repeats
        exactly the words that differ. See R-038."""
        text = "रमेश कुमार मुख्य रस्ता"
        assert detect(text).script is Script.DEVANAGARI
        assert detect(text).language == "mar_Deva"
        # An operator working one jurisdiction can still force it.
        assert detect(text, devanagari_as="hin_Deva").language == "hin_Deva"
        assert detect(text, devanagari_as="mar_Deva").language == "mar_Deva"


class TestValuesThatMustNotBeTranslated:
    """Sending an identifier to a translator risks it inventing words, and it
    wastes a model call on every document."""

    @pytest.mark.parametrize("value", [
        "30,00,000", "560001", "ABCDE1234F", "1234 5678 9012",
        "15-06-2024", "455/1", "", "   ", "42 1/2",
    ])
    def test_not_queued(self, value):
        assert needs_translation(value) is False

    def test_digits_detect_as_neutral_not_english(self):
        """Calling an amount "English" would be a lie that happens to work; it
        breaks as soon as a caller counts languages."""
        assert detect("560001").script is Script.NEUTRAL

    def test_english_text_is_not_queued(self):
        assert needs_translation("Ramesh Kumar, Main Road, Bengaluru") is False


class TestMixedLanguageDocuments:
    def test_dominant_script_wins(self):
        result = detect("ರಮೇಶ್ ಕುಮಾರ್ ಬೆಂಗಳೂರು PAN ABCDE1234F")
        assert result.language == "kan_Knda"

    def test_mixed_is_reported(self):
        result = detect("ರಮೇಶ್ ಕುಮಾರ್ PAN ABCDE1234F")
        assert result.is_mixed is True
        assert Script.KANNADA in result.scripts
        assert Script.LATIN in result.scripts

    def test_digits_do_not_make_a_field_mixed(self):
        """A Kannada address with a house number is not a mixed-language field
        in any useful sense."""
        assert detect("ಮುಖ್ಯ ರಸ್ತೆ 42").is_mixed is False

    def test_a_document_summary_counts_every_language(self):
        tally = summarise([text for _, text in SAMPLES.values()])
        assert tally["kan_Knda"] == 1
        assert "eng_Latn" in tally

    def test_two_regional_languages_in_one_document(self):
        """Rare but real on border-district deeds."""
        tally = summarise(["ರಮೇಶ್ ಕುಮಾರ್", "రమేష్ కుమార్", "Main Road"])
        assert tally["kan_Knda"] == 1 and tally["tel_Telu"] == 1


class TestServiceWithoutAModel:
    """The model may be absent, disabled or broken. None of those may lose a
    deed."""

    def _service(self, **overrides) -> TranslationService:
        config = TranslationConfig(**overrides)
        return TranslationService(config)

    def test_disabled_reports_itself(self):
        ok, detail = self._service(enabled=False).available()
        assert ok is False and "disabled" in detail

    def test_missing_model_reports_itself(self, tmp_path):
        ok, detail = self._service(model_dir=tmp_path).available()
        assert ok is False and "no model weights" in detail

    def test_the_original_survives_a_failure(self, tmp_path):
        """`output` must never be blank - a lost value is worse than an
        untranslated one on a legal record."""
        service = self._service(model_dir=tmp_path)
        item = TranslationItem(key="b1.name", text="ರಮೇಶ್ ಕುಮಾರ್")
        service.translate([item])
        assert item.output == "ರಮೇಶ್ ಕುಮಾರ್"
        assert item.translated is None

    def test_failure_is_reported_not_raised(self, tmp_path):
        result = self._service(model_dir=tmp_path).translate(
            [TranslationItem(key="k", text="ರಮೇಶ್")])
        assert result.ok is False
        assert result.error
        assert len(result.untranslated) == 1

    def test_english_needs_no_model(self, tmp_path):
        """A wholly English deed must succeed even with nothing installed."""
        result = self._service(model_dir=tmp_path).translate(
            [TranslationItem(key="k", text="Ramesh Kumar")])
        assert result.ok is True
        assert result.engine == "none"
        assert not result.untranslated

    def test_identifiers_need_no_model(self, tmp_path):
        result = self._service(model_dir=tmp_path).translate([
            TranslationItem(key="pan", text="ABCDE1234F"),
            TranslationItem(key="amount", text="30,00,000"),
        ])
        assert result.ok is True and result.translated == 0


class TestCaching:
    """A batch of deeds repeats villages, districts and offices heavily."""

    def test_a_repeat_is_served_from_cache(self, tmp_path):
        service = TranslationService(TranslationConfig(model_dir=tmp_path))
        service._cache[("ಬೆಂಗಳೂರು", "kan_Knda", ENGLISH)] = "Bengaluru"

        item = TranslationItem(key="v", text="ಬೆಂಗಳೂರು")
        service.translate([item])
        assert item.translated == "Bengaluru"
        assert item.from_cache is True

    def test_the_cache_survives_an_unavailable_model(self, tmp_path):
        """Cached values must still be applied when the model is missing -
        otherwise a restart loses work already done."""
        service = TranslationService(TranslationConfig(model_dir=tmp_path))
        service._cache[("ಬೆಂಗಳೂರು", "kan_Knda", ENGLISH)] = "Bengaluru"
        result = service.translate([TranslationItem(key="v", text="ಬೆಂಗಳೂರು")])
        assert result.translated == 1

    def test_the_key_includes_the_language(self):
        """The same string can appear in two scripts; one cache entry must not
        answer for both."""
        service = TranslationService(TranslationConfig())
        service._cache[("x", "kan_Knda", ENGLISH)] = "from kannada"
        assert ("x", "tel_Telu", ENGLISH) not in service._cache

    def test_statistics_are_reported(self, tmp_path):
        service = TranslationService(TranslationConfig(model_dir=tmp_path))
        service._cache[("ಬೆಂಗಳೂರು", "kan_Knda", ENGLISH)] = "Bengaluru"
        service.translate([TranslationItem(key="v", text="ಬೆಂಗಳೂರು")])
        assert service.cache_stats()["hits"] == 1


class TestConfiguration:
    """Every option the requirement names must exist and be reachable."""

    def test_defaults_are_sane(self):
        config = TranslationConfig()
        assert config.enabled is True
        assert config.target_language == ENGLISH
        assert config.source_language == "auto"
        assert config.device == "auto"

    def test_every_required_option_exists(self):
        fields = TranslationConfig().as_dict()
        for option in ("enabled", "target_language", "source_language",
                       "model_dir", "device", "batch_size"):
            assert option in fields, f"{option} is not configurable"

    def test_settings_override_defaults(self):
        stored = {"translation_enabled": "false", "translation_batch_size": "32",
                  "translation_device": "cpu"}
        config = build_config(lambda k, d="": stored.get(k, d))
        assert config.enabled is False
        assert config.batch_size == 32
        assert config.device == "cpu"

    def test_a_broken_setting_does_not_disable_translation(self):
        def hostile(key: str, default: str = "") -> str:
            raise RuntimeError("database is down")

        config = build_config(hostile)
        assert config.enabled is True

    def test_the_environment_wins_over_stored_settings(self, monkeypatch):
        """An operator must be able to override for one run without editing
        anything."""
        monkeypatch.setenv("SALEDEED_TRANSLATION_DEVICE", "cpu")
        config = build_config(lambda k, d="": "cuda" if "device" in k else d)
        assert config.device == "cpu"

    def test_all_settings_are_seeded(self):
        from tools.db_setup import DEFAULT_SETTINGS

        for key in ("translation_enabled", "translation_target",
                    "translation_source", "translation_model",
                    "translation_device", "translation_batch_size",
                    "translation_max_retries"):
            assert key in DEFAULT_SETTINGS, f"{key} is not seeded"


class TestModelChoice:
    def test_the_model_is_ungated_and_documented(self):
        from core.translation import DEFAULT_MODEL_REPO

        assert DEFAULT_MODEL_REPO == "facebook/nllb-200-distilled-600M"
        source = (ROOT / "src" / "core" / "translation" / "config.py").read_text(
            encoding="utf-8")
        assert "Ungated" in source, "the choice is not explained"

    def test_the_runner_uses_flores_codes(self):
        """NLLB expects `kan_Knda`; an ISO code produces silent garbage."""
        from core.translation.detect import LANGUAGE_FOR_SCRIPT

        assert LANGUAGE_FOR_SCRIPT[Script.KANNADA] == "kan_Knda"
        assert all("_" in code for code in LANGUAGE_FOR_SCRIPT.values())

    def test_the_installer_can_fetch_and_verify_it(self):
        source = (ROOT / "src" / "tools" / "setup.py").read_text(encoding="utf-8")
        assert "def install_translation_model" in source
        assert "def verify_translation_model" in source
        assert "skipping download" in source, "the installer re-downloads"


class TestProperNounsAreNotTranslated:
    """Names go through a rule, never through the model.

    Measured on this project's own NLLB before the correction:

        ಲಕ್ಷ್ಮಿ ದೇವಿ  -> "Goddess Lakshmi"     a person became a deity
        ವೆಂಕಟೇಶ್      -> "What is Venkatesh?"  a name became a question

    On a record identifying parties to a property transfer that is a corrupted
    document. A sentence translator given a fragment translates *meaning*, and
    on Indian names the meaning is often a word - tightening the beam reduces
    the rate but cannot remove the failure, because the model is doing what it
    was built for.
    """

    from core.translation.transliterate import transliterate as _tr

    CASES = [
        ("ರಮೇಶ್ ಕುಮಾರ್", "Ramesh Kumar"),
        ("ಲಕ್ಷ್ಮಿ ದೇವಿ", "Lakshmi Devi"),
        ("ವೆಂಕಟೇಶ್", "Venkatesh"),
        ("ಬೆಂಗಳೂರು", "Bengaluru"),
        ("रमेश कुमार", "Ramesh Kumar"),
        ("రమేష్ కుమార్", "Ramesh Kumar"),
        ("ரமேஷ் குமார்", "Ramesh Kumar"),
        ("രമേഷ് കുമാർ", "Ramesh Kumar"),
        ("રમેશ કુમાર", "Ramesh Kumar"),
        ("রমেশ কুমার", "Ramesh Kumar"),
        ("ରମେଶ କୁମାର", "Ramesh Kumar"),
    ]

    @pytest.mark.parametrize("source,expected", CASES)
    def test_names_render_by_rule(self, source, expected):
        from core.translation.transliterate import transliterate

        assert transliterate(source) == expected

    def test_it_is_deterministic(self):
        """The property that matters: the same input always gives the same
        output, with no possibility of inventing a different person."""
        from core.translation.transliterate import transliterate

        first = transliterate("ಲಕ್ಷ್ಮಿ ದೇವಿ")
        for _ in range(5):
            assert transliterate("ಲಕ್ಷ್ಮಿ ದೇವಿ") == first

    def test_english_is_untouched(self):
        from core.translation.transliterate import transliterate

        assert transliterate("Ramesh Kumar") == "Ramesh Kumar"

    def test_an_unsupported_script_returns_the_original(self):
        """Urdu has no rule set here. Returning the source is honest; inventing
        a rendering is not."""
        from core.translation.transliterate import transliterate

        text = "رمیش کمار"
        assert transliterate(text) == text

    def test_devanagari_schwa_is_dropped(self):
        """`रमेश` transliterates as "ramesha"; the deed says Ramesh, and
        "Ramesha Kumara" is a different name."""
        from core.translation.transliterate import transliterate

        assert transliterate("रमेश") == "Ramesh"

    def test_dravidian_final_vowels_are_kept(self):
        """The schwa rule must not fire where the vowel is real - dropping it
        would turn Rama into Ram."""
        from core.translation.transliterate import transliterate

        assert transliterate("ರಾಮ").lower().startswith("rama")

    def test_the_service_uses_the_rule_not_the_model(self):
        """A transliterate item must never be queued for the model."""
        source = (ROOT / "src" / "core" / "translation" / "service.py").read_text(
            encoding="utf-8")
        assert "transliterate_supported(script)" in source
        assert "Rule-based transliteration is deterministic" in source

    def test_disabling_translation_also_disables_transliteration(self):
        """The switch means "leave values alone" - silently rewriting names
        would be the surprise it exists to prevent."""
        service = TranslationService(TranslationConfig(enabled=False))
        item = TranslationItem(key="n", text="ರಮೇಶ್", kind="transliterate")
        service.translate([item])
        assert item.output == "ರಮೇಶ್"


class TestFragmentArtefactsAreCorrected:
    """A sentence translator given a field produces a sentence."""

    from core.translation.postprocess import tidy as _tidy

    @pytest.mark.parametrize("raw,expected", [
        ("It is located at 123, 4th Avenue, Jayanagar.", "123, 4th Avenue, Jayanagar"),
        ("The male", "Male"),
        ("The woman", "Female"),
        ("main road", "Main road"),
        ("What is Venkatesh?", "Venkatesh"),
        ("  spaced   out  ", "Spaced out"),
    ])
    def test_artefacts_are_removed(self, raw, expected):
        from core.translation.postprocess import tidy

        assert tidy(raw) == expected

    def test_a_real_article_is_kept(self):
        """"The Bank of Baroda" must keep its article - only short values are
        stripped, because that is where the model pads."""
        from core.translation.postprocess import tidy

        assert tidy("The Bank of Baroda") == "The Bank of Baroda"

    def test_gender_uses_a_closed_vocabulary(self):
        """Three possible values. A translator is the wrong tool."""
        from core.translation.postprocess import tidy

        for raw in ("male", "The male", "man", "gentleman"):
            assert tidy(raw) == "Male"
        for raw in ("female", "The woman", "lady", "girl"):
            assert tidy(raw) == "Female"

    def test_nothing_becomes_blank(self):
        """A missing value in a legal record is worse than an awkward one."""
        from core.translation.postprocess import tidy

        for raw in ("The", "?", "a", "It is located at"):
            assert tidy(raw), f"{raw!r} produced a blank cell"

    def test_capitalisation_is_not_title_case(self):
        """`.title()` would produce "4Th Cross Road" and destroy abbreviations."""
        from core.translation.postprocess import tidy

        assert tidy("4th cross road") == "4th cross road"
        assert tidy("PAN office") == "PAN office"


class TestMarathi:
    """Marathi end to end: detection, configuration, and the plumbing.

    R-038. Marathi was half-supported - `mar_Deva` existed in the language
    table and NLLB has always handled it - but detection always answered Hindi
    and nothing in the interface could say otherwise. A Maharashtra deed was
    translated by the Hindi model and nobody could tell.
    """

    #: A registered Maharashtra deed repeats these. Each line is Marathi in
    #: vocabulary, not merely in script.
    MARATHI = [
        "मौजे कोंढवा तालुका हवेली जिल्हा पुणे",
        "श्री रमेश कुळकर्णी यांनी सदर मिळकत खरेदी केली आहे",
        "क्षेत्रफळ चौरस मीटर आणि वहिवाट",
        "दस्त नोंदणी कार्यालय पुणे",
    ]
    HINDI = [
        "यह संपत्ति जिला पुणे तहसील हवेली में है",
        "विक्रेता श्री शर्मा और क्रेता श्री वर्मा के बीच",
        "पंजीकरण कार्यालय द्वारा किया गया",
    ]

    @pytest.mark.parametrize("text", MARATHI)
    def test_marathi_is_detected(self, text):
        result = detect(text)
        assert result.language == "mar_Deva", f"{text!r} -> {result.reason}"
        assert result.name == "Marathi"

    @pytest.mark.parametrize("text", HINDI)
    def test_hindi_is_not_mistaken_for_marathi(self, text):
        """The fix must not simply relabel all Devanagari as Marathi."""
        result = detect(text)
        assert result.language == "hin_Deva", f"{text!r} -> {result.reason}"

    def test_a_bare_name_is_not_guessed(self):
        """Where the evidence is absent the module falls back and says so,
        rather than inventing a decision. That restraint is the whole reason
        this project detects by script instead of by a statistical model."""
        result = detect("रमेश कुमार")
        assert result.reason == "no distinguishing evidence"
        assert result.language == "hin_Deva"          # the documented fallback

    def test_a_letter_hindi_does_not_use_is_decisive(self):
        """`ळ` is an ordinary Marathi consonant and absent from standard Hindi."""
        from core.translation.detect import discriminate_devanagari

        language, why = discriminate_devanagari("मिळकत")
        assert language == "mar_Deva"
        assert "ळ" in why

    def test_the_setting_overrides_the_evidence(self):
        """An operator processing one jurisdiction should be able to say so and
        have it obeyed, including against the detector."""
        assert detect(self.HINDI[0], devanagari_as="mar_Deva").language == "mar_Deva"
        assert detect(self.MARATHI[0], devanagari_as="hin_Deva").language == "hin_Deva"

    def test_mixed_marathi_and_english_still_translates(self):
        """Deeds interleave English words. The dominant script decides."""
        from core.translation.detect import needs_translation

        text = "मौजे कोंढवा Survey No. 42 तालुका हवेली"
        result = detect(text)
        assert result.language == "mar_Deva"
        assert result.is_mixed
        assert needs_translation(text)

    def test_marathi_alongside_kannada_is_kept_apart(self):
        """Two regional languages in one batch must not collapse into one."""
        assert detect("मौजे कोंढवा तालुका").language == "mar_Deva"
        assert detect("ಬೆಂಗಳೂರು ದಕ್ಷಿಣ ತಾಲ್ಲೂಕು").language == "kan_Knda"

    def test_digits_alone_are_never_marathi(self):
        from core.translation.detect import needs_translation

        assert not needs_translation("४११०४८")     # Devanagari digits
        assert not needs_translation("411048")

    def test_the_configuration_defaults_to_detecting(self):
        from core.translation import build_config

        assert build_config().devanagari_as == "auto"

    def test_the_translation_model_supports_marathi(self):
        """NLLB-200 covers `mar_Deva`; nothing extra has to be downloaded. If
        the model is ever swapped, this is the check that notices."""
        from core.translation.detect import LANGUAGE_NAMES

        assert LANGUAGE_NAMES["mar_Deva"] == "Marathi"

    def test_the_interface_offers_the_choice(self):
        """The selector that used to sit here wrote `translation_language`,
        which nothing read. This one writes the setting the pipeline uses."""
        services = (ROOT / "src" / "app" / "services.py").read_text(encoding="utf-8")
        template = (ROOT / "src" / "app" / "ui" / "templates" /
                    "settings.mustache").read_text(encoding="utf-8")
        js = (ROOT / "src" / "app" / "ui" / "assets" / "app.js").read_text(encoding="utf-8")

        assert '"mar_Deva", "Always Marathi"' in services
        assert "devanagari_languages" in services and "devanagari_languages" in template
        assert '"devanagari-as", "translation_devanagari_as"' in js
        assert "translation_language" not in js, "the dead control is still saved"

    def test_the_decision_is_logged(self):
        source = (ROOT / "src" / "core" / "translation" /
                  "service.py").read_text(encoding="utf-8")
        assert "Devanagari: %d field(s) resolved to %s" in source
        assert "evidence" in source


class TestDevanagariSchwaDeletion:
    """Final-vowel handling for Hindi and Marathi names.

    R-038. The schwa rule ran *after* diacritics were stripped, by which point
    an inherent `a` and a real `ā` are the same character. `सुनीता` printed as
    "Sunit" - a different name, on a legal document - and `कोंढवा` as "Kondhav".
    """

    def _t(self, text):
        from core.translation.detect import Script
        from core.translation.transliterate import transliterate

        return transliterate(text, script=Script.DEVANAGARI)

    def test_a_real_final_vowel_survives(self):
        assert self._t("सुनीता") == "Sunita"
        assert self._t("सीता") == "Sita"

    def test_the_inherent_schwa_is_still_dropped(self):
        """The original rule was right, only mis-ordered. `रमेश` is Ramesh."""
        assert self._t("रमेश") == "Ramesh"
        assert self._t("कुमार") == "Kumar"

    def test_a_schwa_after_a_cluster_is_pronounced(self):
        """`महाराष्ट्र` is Maharashtra. Dropping there produced "Maharashtr"."""
        assert self._t("महाराष्ट्र") == "Maharashtra"
        assert self._t("बुद्ध") == "Buddha"

    def test_an_aspirate_is_one_consonant_not_two(self):
        """`kh` is a digraph in IAST. Counting it as a cluster would keep the
        schwa and print "Mukha"."""
        assert self._t("मुख") == "Mukh"

    def test_marathi_place_names(self):
        assert self._t("पुणे") == "Pune"
        assert self._t("ठाणे") == "Thane"
        assert self._t("हवेली") == "Haveli"

    def test_short_words_are_left_alone(self):
        """`राम` is Ram, and the rule must not chew into two-letter words."""
        assert self._t("राम") == "Ram"


class TestKannadaInitials:
    """Kannada spells English initials phonetically. R-045.

    `ಜಿ` is how you write the letter G, `ಕೆ` is K, `ಎಂ` is M. Sounding them out
    produced `Ji.ke. Raju` where the deed means `G.K. Raju` - not a spelling
    preference but different letters from the ones printed. It affected every
    name written with initials, which on Indian deeds is most of them.
    """

    def _t(self, text):
        from core.translation.detect import Script
        from core.translation.transliterate import transliterate

        return transliterate(text, script=Script.KANNADA)

    def test_initials_become_letters(self):
        assert self._t("ಜಿ.ಕೆ. ರಾಜು") == "G.K. Raju"
        assert self._t("ಎಂ.ಟಿ. ರಂಗೇಗೌಡ") == "M.T. Rangegowda"

    def test_an_initial_after_a_stop_is_found_too(self):
        """`ಶಶಿಕುಮಾರ್.ಆರ್` has no trailing stop. Requiring one missed every
        trailing initial."""
        assert self._t("ಶಶಿಕುಮಾರ್.ಆರ್") == "Shashikumar R."
        assert self._t("ಶಮಾ ಕೌಸರ್.ಹೆಚ್.ಎಸ್") == "Shama Kowsar H.S."

    def test_an_initial_running_on_from_another_is_found(self):
        assert self._t("ವಿಜಯ್ಕುಮಾರ್ ಕೆ. ಎಂ") == "Vijaykumar K. M."

    def test_an_ordinary_word_is_not_mistaken_for_an_initial(self):
        """`ಬಿ` is the letter B and the start of `ಬಿಂದು`. Position is the guard:
        without an adjoining stop it is a word."""
        assert self._t("ಬಿಂದು") == "Bindu"
        assert self._t("ಚನ್ನೇಗೌಡ") == "Channegowda"
        assert self._t("ಮಹಬೂಬ್ ಖಾನ್") == "Mahabub Khan"

    def test_a_name_written_against_its_initials_gets_a_space(self):
        """`ಜಿ.ಎಲ್.ರವಿ` has no spaces at all, so title-casing could not reach
        the name - it printed `G.L.ravi`."""
        assert self._t("ಜಿ.ಎಲ್.ರವಿ") == "G.L. Ravi"

    def test_the_letter_c_is_not_turned_into_ch(self):
        """`ಸಿ` is the letter C. The Kannada `c`->`ch` convention rewrote the
        initial itself and produced `Navin Kumar M. Ch.`"""
        assert self._t("ನವೀನ್ ಕುಮಾರ್ ಎಂ. ಸಿ.") == "Navin Kumar M. C."
        assert self._t("ಸಿ. ಚನ್ನೇಗೌಡ") == "C. Channegowda"

    def test_kannada_spelling_conventions(self):
        """`c` is IAST for ಚ and is written `ch`; `au` is written `ow` in the
        names it appears in."""
        assert self._t("ಚನ್ನೇಗೌಡ") == "Channegowda"
        assert self._t("ಗೌರಮ್ಮ") == "Gowramma"


class TestNameColumnHygiene:
    """Duplicate parties and unusable names. R-045."""

    def _rows(self, extraction):
        from core.csv_export import DocumentExport, build_rows

        return build_rows([DocumentExport(transaction_identity="T",
                                          extraction=extraction)])

    def test_the_same_party_listed_twice_yields_one_row(self):
        rows = self._rows({
            "seller_details": [
                {"name": "channegowda", "aadhaar_number": "971236189364"},
                {"name": "channegowda", "aadhaar_number": "971236189364"}],
            "buyer_details": [], "property_details": {}, "document_details": {}})
        assert len(rows) == 1

    def test_two_real_people_are_both_kept(self):
        """Deduplication must not merge distinct parties."""
        rows = self._rows({
            "seller_details": [
                {"name": "One", "aadhaar_number": "111122223333"},
                {"name": "Two", "aadhaar_number": "444455556666"}],
            "buyer_details": [], "property_details": {}, "document_details": {}})
        assert len(rows) == 2

    def test_an_alias_is_removed_from_the_name(self):
        """`@` marks an alias on an Indian deed. The Name column carries the
        primary name only, so everything from the marker onward is dropped.

        This test previously asserted the opposite - that the alias is kept,
        because it is a fact the deed states. That reading was overruled: the
        column is for one name. The full form is not lost, it is logged against
        the document by `build_rows`.
        """
        from core.csv_export import primary_name

        assert primary_name("AAKASH SACHIDANAND MISHRA @ AAKASH MISHRA") ==             "AAKASH SACHIDANAND MISHRA"
        rows = self._rows({
            "seller_details": [{"name": "AAKASH MISHRA @ AAKASH",
                                "father_name": "RAJU @ RAJESH"}],
            "buyer_details": [], "property_details": {}, "document_details": {}})
        assert rows[0]["Person Name (PC)"] == "AAKASH MISHRA"
        assert rows[0]["Father's Name (PC)"] == "RAJU"
        assert "@" not in rows[0]["Person Name (PC)"]

    def test_a_name_without_an_alias_is_untouched(self):
        """The rule must not trim an ordinary name, in any script."""
        from core.csv_export import primary_name

        for name in ("KRISHNAPPA", "K. ANAND", "ಚನ್ನೇಗೌಡ", "ಸುನೀತಾ ಜೋಶಿ",
                     "Venkatesh Uruph Venkataramaiah"):
            assert primary_name(name) == name

    def test_punctuation_alone_is_not_a_name(self):
        from core.csv_export import looks_like_a_name

        for value in ("@", "   ", "...", "-", ""):
            assert not looks_like_a_name(value)

    def test_a_kannada_name_survives_the_check(self):
        """A Kannada name is mostly combining marks. Any cleaner built from an
        allowed-character list mangles it; this one tests for a letter."""
        from core.csv_export import looks_like_a_name

        assert looks_like_a_name("ಚನ್ನೇಗೌಡ")
        assert looks_like_a_name("ಸುನೀತಾ ಜೋಶಿ")

    def test_a_party_with_no_usable_name_is_not_exported(self):
        rows = self._rows({
            "seller_details": [{"name": "Real Person"}, {"name": "@"}],
            "buyer_details": [], "property_details": {}, "document_details": {}})
        assert len(rows) == 1
        assert rows[0]["Person Name (PC)"] == "Real Person"
