import json

from bench.adapters.base import PermanentError, Question, Request, Response, TransientError
from bench.runner import load_results, measure_rtt, run_requests

Q = Question(type="predicate", name="q", instructions="?")


def reqs(n):
    return [Request(f"c{i}", f"texto {i}", Q) for i in range(n)]


class Fake:
    name = "fake"

    def __init__(self, script=None, rtt=12.0):
        self.script = script or {}  # case_id -> lista de exceções a lançar antes de responder
        self.calls = []
        self.rtt = rtt

    def decide(self, req):
        self.calls.append(req.case_id)
        pending = self.script.get(req.case_id, [])
        if pending:
            raise pending.pop(0)
        return Response(case_id=req.case_id, provider=self.name, answer=0.9,
                        probabilities={"true": 0.9, "false": 0.1}, latency_ms=5.0)

    def ping(self):
        return self.rtt


def test_runs_all_and_writes_jsonl(tmp_path):
    out = tmp_path / "fake.jsonl"
    stats = run_requests(Fake(), reqs(3), out, warmup=0, sleep=lambda s: None)
    assert stats == {"ok": 3, "failed": 0, "skipped": 0}
    assert set(load_results(out)) == {"c0", "c1", "c2"}


def test_warmup_calls_are_discarded(tmp_path):
    a = Fake()
    run_requests(a, reqs(2), tmp_path / "f.jsonl", warmup=3, sleep=lambda s: None)
    assert a.calls == ["c0", "c0", "c0", "c0", "c1"]
    assert len((tmp_path / "f.jsonl").read_text().splitlines()) == 2


def test_resume_skips_successes_and_retries_failures(tmp_path):
    out = tmp_path / "f.jsonl"
    run_requests(Fake({"c1": [PermanentError("x")]}), reqs(3), out, warmup=0, sleep=lambda s: None)
    assert load_results(out)["c1"].error.startswith("PermanentError")
    a = Fake()
    stats = run_requests(a, reqs(3), out, warmup=0, sleep=lambda s: None)
    assert stats == {"ok": 1, "failed": 0, "skipped": 2} and a.calls == ["c1"]
    assert load_results(out)["c1"].error is None


def test_transient_retries_with_backoff_then_succeeds(tmp_path):
    sleeps = []
    a = Fake({"c0": [TransientError("429"), TransientError("503")]})
    stats = run_requests(a, reqs(1), tmp_path / "f.jsonl", warmup=0, backoff=1.0, sleep=sleeps.append)
    assert stats["ok"] == 1 and sleeps == [1.0, 2.0]


def test_transient_exhausted_is_recorded(tmp_path):
    out = tmp_path / "f.jsonl"
    a = Fake({"c0": [TransientError("429")] * 4})
    stats = run_requests(a, reqs(1), out, warmup=0, max_retries=4, sleep=lambda s: None)
    assert stats["failed"] == 1 and "TransientError" in load_results(out)["c0"].error


def test_permanent_error_recorded_without_retry(tmp_path):
    a = Fake({"c0": [PermanentError("422")]})
    run_requests(a, reqs(1), tmp_path / "f.jsonl", warmup=0, sleep=lambda s: None)
    assert a.calls == ["c0"]


def test_resume_after_truncated_line(tmp_path):
    out = tmp_path / "f.jsonl"
    good = Response(case_id="c0", provider="fake", answer=0.5).to_dict()
    out.write_text(json.dumps(good) + "\n" + '{"case_id": "c1", "prov', encoding="utf-8")
    run_requests(Fake(), reqs(2), out, warmup=0, sleep=lambda s: None)
    results = load_results(out)
    assert set(results) == {"c0", "c1"} and results["c1"].error is None


def test_measure_rtt():
    assert measure_rtt(Fake(rtt=7.0), n=3) == [7.0, 7.0, 7.0]
    assert measure_rtt(Fake(rtt=None), n=3) == []
