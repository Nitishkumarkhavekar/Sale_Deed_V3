"""The AI server's HTTP surface, against a real server.

`server.py` sat at 28% and `engines/mock.py` at 0%, for the same reason: both
were only ever exercised by a live GPU stack, so neither the routing, the
status codes, nor the queue's behaviour under bad input had a test that could
run anywhere.

A real `ThreadingHTTPServer` with `MockEngine` fixes both. No GPU, no model, no
subprocess - but genuine sockets, genuine JSON, genuine status codes. That
matters because the defects this catches are boundary defects: a wrong status
code, an exception class leaking into a response, a malformed body reaching a
worker. None of those are visible from inside Python.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import replace
import urllib.error
import urllib.request

import pytest

from ai_server.engines.mock import MockEngine
from ai_server.profiles import Profile, QUANT_LADDER
from ai_server.resources import ResourceGovernor
from ai_server.server import AiServer, make_http_server


def _profile():
    return Profile(
        device="cpu", quant=QUANT_LADDER[-1], n_ctx=16384, kv_type="q8_0",
        n_gpu_layers=0, n_parallel=1, n_threads=4, gpu_uuid=None,
        weight_bytes=1, kv_bytes=1, overhead_bytes=1, budget_bytes=10,
        reason="test")


class _FixedGovernor(ResourceGovernor):
    """The real governor with its admission decision pinned.

    A subclass rather than a stand-in: `AiServer.start` calls `start`, the queue
    calls `plan`, and a hand-rolled object silently lacks whichever of those is
    reached next - turning a genuine result into an AttributeError inside the
    test. Only the one decision under test is overridden.

    It exists because the real decision is resource-dependent, and correctly so:
    these tests failed intermittently with 503 whenever the rest of the suite
    happened to be holding memory. The server was behaving properly; the test was
    asserting something untrue of a busy machine. Pressure has its own test at the
    bottom of this file.
    """

    def __init__(self, *, admit: bool, reason: str = "pinned for tests") -> None:
        super().__init__()
        self._admit = admit
        self._reason = reason

    def plan(self):
        return replace(super().plan(), admit_new_work=self._admit,
                       reason=self._reason)


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    """A real server on a real port, served by the mock engine."""
    model_dir = tmp_path_factory.mktemp("model")
    app = AiServer(MockEngine(), _profile(), model_dir,
                   governor=_FixedGovernor(admit=True))
    app.start()
    # Port 0 lets the OS choose, so a developer already running the real
    # server on 8077 does not see this suite fail for the wrong reason.
    httpd = make_http_server(app, "127.0.0.1", 0)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        httpd.shutdown()
        app.stop()


def _get(base, path, timeout=10):
    try:
        with urllib.request.urlopen(f"{base}{path}", timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def _post(base, path, payload, timeout=30, raw=None):
    body = raw if raw is not None else json.dumps(payload).encode()
    req = urllib.request.Request(f"{base}{path}", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


class TestTheReadOnlyRoutes:
    def test_health_reports_the_engine(self, server):
        status, body = _get(server, "/health")
        assert status == 200
        assert body["engine"]["engine"] == "mock"
        assert "ready" in body

    def test_hardware_answers(self, server):
        status, body = _get(server, "/hardware")
        assert status == 200
        assert isinstance(body, dict)

    def test_profile_answers_even_without_a_checkpoint(self, server):
        """It used to 500 when config.json was absent - the diagnostic failing
        at exactly the moment it is worth calling."""
        status, body = _get(server, "/profile")
        assert status == 200
        assert isinstance(body, dict)
        # The ladder could not be computed here, and the response says so
        # rather than pretending or dying.
        assert "ladder" in body

    def test_the_queue_summary_answers(self, server):
        status, body = _get(server, "/jobs")
        assert status == 200
        assert "queued_depth" in body


class TestRoutingRefusesCleanly:
    def test_an_unknown_route_is_404(self, server):
        status, body = _get(server, "/no-such-route")
        assert status == 404
        assert "error" in body

    def test_path_traversal_is_refused(self, server):
        """Not served, not resolved, not 500 - just not a route."""
        status, _ = _get(server, "/../../etc/passwd")
        assert status == 404

    def test_an_unknown_job_is_404(self, server):
        status, body = _get(server, "/jobs/does-not-exist")
        assert status == 404
        assert "error" in body


class TestExtractionOverHttp:
    def test_a_deed_is_accepted_and_completes(self, server):
        status, body = _post(server, "/extract", {
            "ocr_text": "Sale deed. Seller KRISHNAPPA. PAN ABCPK1234F.",
            "document_id": "http-1"})
        # 202: submission is asynchronous by design so a 1000-file batch never
        # blocks the caller.
        assert status == 202
        job_id = body["job_id"]

        deadline = time.time() + 30
        while time.time() < deadline:
            status, job = _get(server, f"/jobs/{job_id}")
            assert status == 200
            if job["state"] in ("done", "failed"):
                break
            time.sleep(0.2)
        assert job["state"] == "done", job
        assert job["result"]
        assert job["document_id"] == "http-1"

    def test_a_batch_returns_one_id_per_deed(self, server):
        status, body = _post(server, "/extract/batch", {
            "documents": [{"ocr_text": f"deed {n}", "document_id": str(n)}
                          for n in range(3)]})
        assert status in (200, 202)
        assert len(body["job_ids"]) == 3
        assert len(set(body["job_ids"])) == 3, "job ids collided"


class TestBadInputIsRejectedNotCrashed:
    @pytest.mark.parametrize("payload,expected", [
        ({}, 400),
        ({"ocr_text": ""}, 400),
        ({"ocr_text": "   "}, 400),
        ({"ocr_text": 123}, 400),
        ({"ocr_text": []}, 400),
        ({"ocr_text": None}, 400),
    ])
    def test_a_bad_extract_body_is_400(self, server, payload, expected):
        status, body = _post(server, "/extract", payload)
        assert status == expected
        assert body.get("error")

    def test_malformed_json_is_400_not_500(self, server):
        status, body = _post(server, "/extract", None, raw=b"{not json at all")
        assert status == 400
        assert body.get("error")

    def test_no_response_leaks_an_exception_class(self, server):
        """The rule the project already had, enforced at the boundary: an
        operator must never be shown a Python exception type."""
        for payload in ({"ocr_text": 123}, {}, {"ocr_text": []}):
            _status, body = _post(server, "/extract", payload)
            message = str(body.get("error", ""))
            for leak in ("Traceback", "AttributeError", "TypeError",
                         "object has no"):
                assert leak not in message, f"{leak!r} leaked: {message}"

    def test_the_model_route_requires_a_boolean(self, server):
        for value in ("banana", 1, None, []):
            status, body = _post(server, "/model", {"loaded": value})
            assert status == 400, f"{value!r} was accepted"
            assert "true or false" in str(body.get("error", ""))


class TestConcurrency:
    def test_many_simultaneous_requests_all_succeed(self, server):
        """ThreadingHTTPServer plus a queue: nothing here may deadlock, and no
        request may be answered with another's result."""
        results: list[tuple] = []

        def hit(n):
            results.append(_post(server, "/extract", {
                "ocr_text": f"deed number {n}", "document_id": f"c{n}"}))

        threads = [threading.Thread(target=hit, args=(n,)) for n in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        assert len(results) == 12
        assert all(status == 202 for status, _ in results)
        ids = [body["job_id"] for _, body in results]
        assert len(set(ids)) == 12, "two requests were given the same job id"


class TestTheMockEngineItself:
    def test_it_returns_schema_shaped_json(self):
        from ai_server.engines.base import ExtractionRequest

        engine = MockEngine()
        engine.start()
        try:
            result = engine.generate(ExtractionRequest(
                ocr_text="Seller KRISHNAPPA PAN ABCPK1234F Aadhaar 663212345678",
                prompt="extract"))
            parsed = json.loads(result.text)
            assert "seller_details" in parsed or "buyer_details" in parsed
        finally:
            engine.stop()

    def test_it_can_be_told_to_fail(self):
        """`fail_every` injects the failure a real model actually produces -
        malformed output - rather than an exception. That is the harder case
        for the pipeline to handle, and the one worth simulating."""
        from ai_server.engines.base import ExtractionRequest, FinishReason

        engine = MockEngine(fail_every=1)
        engine.start()
        try:
            result = engine.generate(ExtractionRequest(ocr_text="x", prompt="p"))
            with pytest.raises(json.JSONDecodeError):
                json.loads(result.text)
            assert result.finish_reason is FinishReason.LENGTH
            assert result.metadata.get("injected_failure")
        finally:
            engine.stop()

    def test_health_is_answerable_before_and_after_start(self):
        engine = MockEngine()
        assert engine.health().ready is False
        engine.start()
        assert engine.health().ready is True
        engine.stop()
        assert engine.health().ready is False


class TestAdmissionUnderPressure:
    """The behaviour the fixture above deliberately holds constant.

    A 503 from `/extract` is not a fault - it is the governor declining to pile
    more work onto a machine that is already short of memory, which is what stops
    a batch from dying with a CUDA OOM halfway through. Worth its own test
    precisely because the other tests suppress it.
    """

    @pytest.fixture()
    def strained(self, tmp_path):
        app = AiServer(MockEngine(), _profile(), tmp_path,
                       governor=_FixedGovernor(
                           admit=False, reason="only 200 MiB free"))
        app.start()
        httpd = make_http_server(app, "127.0.0.1", 0)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{httpd.server_address[1]}"
        finally:
            httpd.shutdown()
            app.stop()

    def test_work_is_refused_with_503_not_500(self, strained):
        status, body = _post(strained, "/extract",
                             {"ocr_text": "a deed", "document_id": "p-1"})
        assert status == 503
        assert body.get("error")

    def test_the_refusal_says_it_is_worth_retrying(self, strained):
        """The distinction a caller acts on: retry shortly, or give up."""
        _status, body = _post(strained, "/extract",
                              {"ocr_text": "a deed", "document_id": "p-2"})
        assert body.get("retry") is True

    def test_the_refusal_explains_itself(self, strained):
        _status, body = _post(strained, "/extract",
                              {"ocr_text": "a deed", "document_id": "p-3"})
        message = str(body.get("error", ""))
        # The reason, which is the actionable half. The pressure *label* comes
        # from the real snapshot and so depends on the machine running the
        # test - asserting a particular one would pin the test to this laptop.
        assert "200 MiB free" in message
        assert "system pressure is" in message

    def test_read_only_routes_still_answer_under_pressure(self, strained):
        """Refusing new work must not make the server undiagnosable - /health is
        exactly what an operator reaches for when work is being refused."""
        for route in ("/health", "/jobs", "/hardware"):
            status, _ = _get(strained, route)
            assert status == 200, route
