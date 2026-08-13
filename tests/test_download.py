"""Data View download: one chosen batch, to a chosen location.

The fault this guards against is quiet: an export that writes the wrong batch,
or mixes two, produces a plausible file that is wrong. Nothing about it looks
like a failure.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from core.db.engine import session_scope
from core.db.models import BatchState
from core.db.repositories import UnitOfWork

pytestmark = pytest.mark.integration


def _batch(uow: UnitOfWork, name: str, identities: list[str]):
    user = uow.users.get_or_create("download_test")
    batch = uow.batches.create(name, user, len(identities), 1024 * len(identities))
    # Out of QUEUED at once: `batches.create` enforces a four-batch queue cap,
    # so on a machine with real batches waiting these fixtures would trip the
    # cap rather than test anything. Batch state is irrelevant to an export.
    uow.batches.set_state(batch, BatchState.COMPLETED)
    docs = uow.documents.add_many(batch, [
        {"document_id": ident, "source_filename": f"{ident}.pdf",
         "page_count": 2, "size_bytes": 1024} for ident in identities])
    for doc, ident in zip(docs, identities):
        doc.transaction_identity = ident
        uow.results.save_property(doc, {
            "schedule_c_property_address": f"Site 1, {name} Town, Bengaluru",
            "sale_consideration": "100000"})
        uow.results.replace_persons(doc, {
            "seller_details": [{"name": f"SELLER OF {name}"}],
            "buyer_details": [{"name": f"BUYER OF {name}"}]})
    uow.flush()
    return batch


@pytest.fixture()
def two_batches(session_factory):
    made = {}
    with session_scope(session_factory) as session:
        uow = UnitOfWork(session)
        a = _batch(uow, "AlphaBatch", ["AAA-1-00001-2025-26", "AAA-1-00002-2025-26"])
        b = _batch(uow, "BetaBatch", ["BBB-1-00001-2025-26"])
        made["a"], made["b"] = a.id, b.id
    yield made
    with session_scope(session_factory) as session:
        uow = UnitOfWork(session)
        for key in ("a", "b"):
            batch = uow.batches.get(made[key])
            if batch is not None:
                session.delete(batch)


def _read(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


class TestOnlyTheSelectedBatchIsWritten:
    def test_selecting_batch_a_writes_only_batch_a(self, app_service, two_batches):
        result = app_service.export_view({"batch_id": two_batches["a"]})
        rows = _read(result["path"])
        identities = {r["Transaction Identity"] for r in rows}
        assert identities == {"AAA-1-00001-2025-26", "AAA-1-00002-2025-26"}
        assert not any(i.startswith("BBB") for i in identities), "batches mixed"
        assert result["batch_name"] == "AlphaBatch"

    def test_selecting_batch_b_writes_only_batch_b(self, app_service, two_batches):
        result = app_service.export_view({"batch_id": two_batches["b"]})
        identities = {r["Transaction Identity"] for r in _read(result["path"])}
        assert identities == {"BBB-1-00001-2025-26"}
        assert result["batch_name"] == "BetaBatch"

    def test_no_party_of_another_batch_appears(self, app_service, two_batches):
        """Rows are per party, so a mixing bug shows up in the names first."""
        rows = _read(app_service.export_view({"batch_id": two_batches["a"]})["path"])
        names = {r["Person Name (PC)"] for r in rows}
        assert names == {"SELLER OF AlphaBatch", "BUYER OF AlphaBatch"}
        assert not any("BetaBatch" in n for n in names)

    def test_the_identifier_is_carried_from_the_request(self, app_service,
                                                        two_batches):
        """A string id, as the select element sends it - not an int."""
        result = app_service.export_view({"batch_id": str(two_batches["b"])})
        assert result["batch_id"] == two_batches["b"]
        assert {r["Transaction Identity"] for r in _read(result["path"])} == {
            "BBB-1-00001-2025-26"}


class TestTheChosenLocation:
    def test_the_file_is_written_where_the_operator_chose(self, app_service,
                                                          two_batches, tmp_path):
        target = tmp_path / "picked" / "alpha.csv"
        target.parent.mkdir()
        result = app_service.export_view(
            {"batch_id": two_batches["a"], "destination": str(target)})
        assert Path(result["path"]) == target
        assert target.is_file()
        assert len(_read(target)) == 4          # two documents, two parties each

    def test_a_directory_receives_the_generated_name(self, app_service,
                                                     two_batches, tmp_path):
        result = app_service.export_view(
            {"batch_id": two_batches["a"], "destination": str(tmp_path)})
        written = Path(result["path"])
        assert written.parent == tmp_path
        assert written.suffix == ".csv"
        assert "AlphaBatch" in written.name

    def test_a_name_without_a_suffix_becomes_a_csv(self, app_service,
                                                   two_batches, tmp_path):
        result = app_service.export_view(
            {"batch_id": two_batches["a"], "destination": str(tmp_path / "mine")})
        assert Path(result["path"]).suffix == ".csv"

    def test_no_destination_keeps_the_previous_behaviour(self, app_service,
                                                         two_batches):
        """The existing Download CSV button passes no destination and must go
        on writing to the exports folder."""
        from app.services import EXPORT_DIR

        result = app_service.export_view({"batch_id": two_batches["a"]})
        assert Path(result["path"]).parent == Path(EXPORT_DIR)


class TestTheSaveDialog:
    def test_cancelling_is_reported_rather_than_raising(self, app_service):
        """An empty path means the operator cancelled. That is a decision, not
        a failure, and must not surface as an error."""
        app_service.save_picker = lambda suggested="": ""
        assert app_service.pick_save_path() == {"path": "", "cancelled": True}

    def test_a_chosen_path_comes_back(self, app_service):
        app_service.save_picker = lambda suggested="": r"C:\chosen\out.csv"
        assert app_service.pick_save_path("out.csv") == {
            "path": r"C:\chosen\out.csv", "cancelled": False}

    def test_the_suggested_name_reaches_the_dialog(self, app_service):
        seen = {}
        app_service.save_picker = lambda suggested="": seen.setdefault("name", suggested)
        app_service.pick_save_path("AlphaBatch.csv")
        assert seen["name"] == "AlphaBatch.csv"

    def test_the_dialog_runs_on_the_gui_thread(self):
        """A QFileDialog built on a pool worker terminates the process on
        Windows - no exception, nothing in the log. This list is the only thing
        preventing it."""
        from app.ui.bridge import _GUI_THREAD

        assert "pick_save_path" in _GUI_THREAD


class TestTheDownloadSelector:
    def test_every_batch_is_offered_with_its_document_count(self, app_service,
                                                            two_batches):
        model = app_service._data_view({"batch_id": two_batches["a"]})
        offered = {b["name"]: b["document_count"] for b in model["batches"]}
        assert offered.get("AlphaBatch") == 2
        assert offered.get("BetaBatch") == 1

    def test_the_chosen_batch_is_preselected(self, app_service, two_batches):
        model = app_service._data_view({"batch_id": two_batches["b"]})
        chosen = [b for b in model["batches"] if b["selected"]]
        assert [b["name"] for b in chosen] == ["BetaBatch"]

    def test_with_nothing_chosen_the_newest_batch_is_preselected(self,
                                                                 app_service,
                                                                 two_batches):
        """The selector and the table below must never disagree about which
        batch is in view."""
        model = app_service._data_view({})
        chosen = [b for b in model["batches"] if b["selected"]]
        assert len(chosen) == 1


class TestTheDownloadCardIsUsable:
    """The selector must identify a batch unambiguously, and the Download
    button must not act until one is chosen. R-052."""

    def test_each_option_carries_name_id_count_state_and_date(self, app_service,
                                                              two_batches):
        model = app_service._data_view({})
        alpha = next(b for b in model["batches"] if b["name"] == "AlphaBatch")
        assert f"(ID {two_batches['a']})" in alpha["label"]
        assert "AlphaBatch" in alpha["label"]
        assert "2 document(s)" in alpha["label"]
        assert alpha["state"] in alpha["label"]
        assert alpha["created_at"] in alpha["label"]

    def test_two_batches_with_the_same_name_stay_distinguishable(self,
                                                                 app_service,
                                                                 two_batches):
        """The id is in the label precisely so a repeated name is still
        unambiguous."""
        model = app_service._data_view({})
        labels = [b["label"] for b in model["batches"]]
        assert len(set(labels)) == len(labels)

    def test_the_page_offers_an_unselected_placeholder(self, app_service,
                                                       two_batches):
        """Defaulting to a batch and downloading it is how an operator ends up
        with the wrong file and no reason to suspect it."""
        html = app_service.render_page("data", {})
        head = html[html.find('id="download-batch"'):]
        assert 'Select a batch' in head[:400]
        assert '<option value="">' in head[:400]

    def test_the_download_button_starts_disabled(self, app_service, two_batches):
        html = app_service.render_page("data", {})
        button = html[html.find('id="btn-download-batch"'):][:200]
        assert "disabled" in button

    def test_choosing_a_location_is_its_own_control(self, app_service,
                                                    two_batches):
        html = app_service.render_page("data", {})
        assert 'id="btn-choose-location"' in html
        assert 'id="download-location"' in html
