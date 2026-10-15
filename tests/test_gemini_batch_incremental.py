"""A Gemini batch cell can be submitted in parts: a one-scenario probe first,
then the rest, and the probe's request is never sent (or paid for) twice."""
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _submit_args(source, out, n, dry_run=False):
    return SimpleNamespace(dataset_root=source, out_dir=out, split="nuscenes", condition="video",
                           scenario="scenario_1", scenarios=n, offset=0,
                           display_name="t", dry_run=dry_run)


@pytest.fixture
def batch(tmp_path, monkeypatch):
    from scripts import run_gemini_batch as gb
    from tests.test_evaluation_campaign import make_protocol_dataset
    jobs = []

    def create_job(key, model, display_name, file_name):
        jobs.append(file_name)
        return f"batches/{len(jobs)}"

    monkeypatch.setattr(gb, "_gemini_key", lambda: "k")
    monkeypatch.setattr(gb, "_upload", lambda key, raw: f"files/{len(jobs) + 1}")
    monkeypatch.setattr(gb, "_create_job", create_job)
    source = make_protocol_dataset(tmp_path / "data", n=3)
    return gb, source, tmp_path / "cell", jobs


def _state(out):
    return json.loads((out / "batch_state.json").read_text())


def test_a_later_submit_sends_only_the_scenarios_not_yet_submitted(batch):
    gb, source, out, jobs = batch
    gb.submit(_submit_args(source, out, 1))
    first = _state(out)
    assert [j["keys"] for j in first["jobs"]] == [["scenario_1"]]

    gb.submit(_submit_args(source, out, 3))
    state = _state(out)
    assert [j["keys"] for j in state["jobs"]] == [["scenario_1"], ["scenario_2", "scenario_3"]]
    assert [j["chunk"] for j in state["jobs"]] == [0, 1]
    assert state["scenarios"] == ["scenario_1", "scenario_2", "scenario_3"]
    assert state["mapping"]["scenario_1"] == first["mapping"]["scenario_1"]
    assert set(state["mapping"]) == {"scenario_1", "scenario_2", "scenario_3"}
    sent = [json.loads(l)["key"] for l in (out / "batch_input_1.jsonl").read_text().splitlines()]
    assert sent == ["scenario_2", "scenario_3"]

    gb.submit(_submit_args(source, out, 3))          # nothing new: no job, no upload
    assert len(_state(out)["jobs"]) == 2 and len(jobs) == 2


def test_a_submitted_scenario_whose_prompt_changed_is_refused(batch):
    gb, source, out, jobs = batch
    gb.submit(_submit_args(source, out, 1))
    state = _state(out)
    state["mapping"]["scenario_1"] = "0" * 64
    (out / "batch_state.json").write_text(json.dumps(state))
    with pytest.raises(RuntimeError, match="scenario_1.*changed"):
        gb.submit(_submit_args(source, out, 3))
    assert len(jobs) == 1


def test_the_panel_estimate_counts_only_new_requests(tmp_path, monkeypatch, capsys):
    from scripts import run_api_panel as P
    names = [Path(f"scenario_{i}") for i in (1, 2, 3)]
    monkeypatch.setattr(P, "choose", lambda split, n: names[:n] if n else names)
    cell = tmp_path / P.GEMINI / "carla_video"
    cell.mkdir(parents=True)
    (cell / "batch_state.json").write_text(json.dumps({"mapping": {"scenario_1": "x"}, "jobs": [{}]}))
    calls = []
    monkeypatch.setattr(P.gemini_batch, "submit", lambda args: calls.append(args))
    P.cmd_gemini(SimpleNamespace(action="submit", out_root=tmp_path, splits=["carla"],
                                 conditions=["video"], scenarios=None, confirm=False))
    assert "2 new request(s)" in capsys.readouterr().out and not calls
    P.cmd_gemini(SimpleNamespace(action="submit", out_root=tmp_path, splits=["carla"],
                                 conditions=["video"], scenarios=None, confirm=True))
    assert len(calls) == 1 and calls[0].scenarios == 3
    P.cmd_gemini(SimpleNamespace(action="submit", out_root=tmp_path, splits=["carla"],
                                 conditions=["video"], scenarios=1, confirm=True))
    assert len(calls) == 1                           # the probe is already submitted


class _FakeService:
    """Batch jobs whose state and per-request output a test sets by hand."""

    def __init__(self, gb, monkeypatch):
        self.jobs = {}
        monkeypatch.setattr(gb, "_gemini_key", lambda: "k")
        monkeypatch.setattr(gb, "_upload", lambda key, raw: f"files/{len(self.jobs) + 1}")
        monkeypatch.setattr(gb, "_create_job", self.create)
        monkeypatch.setattr(gb, "_job", lambda name: {
            "metadata": {"state": self.jobs[name]["state"]},
            "response": {"responsesFile": name}})
        monkeypatch.setattr(gb, "_http", lambda url, data, headers, method="POST": "\n".join(
            json.dumps(line) for line in self.jobs[url.split("/v1beta/")[1].split(":")[0]]["out"]
        ).encode())

    def create(self, key, model, display_name, file_name):
        name = f"batches/{len(self.jobs) + 1}"
        self.jobs[name] = {"state": "BATCH_STATE_PENDING", "out": []}
        return name

    def finish(self, name, state="BATCH_STATE_SUCCEEDED", answered=(), errored=()):
        answer = {"candidates": [{"content": {"parts": [{"text": "{}"}]}, "finishReason": "STOP"}],
                  "modelVersion": "gemini-2.5-pro",
                  "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1}}
        self.jobs[name] = {"state": state, "out": (
            [{"key": k, "response": answer} for k in answered]
            + [{"key": k, "error": {"code": 429, "message": "spend cap"}} for k in errored])}


def test_only_requests_without_an_answer_are_retried_and_fetch_joins_every_job(tmp_path, monkeypatch):
    from scripts import run_gemini_batch as gb
    from tests.test_evaluation_campaign import make_protocol_dataset
    service = _FakeService(gb, monkeypatch)
    for name in ("score_cell", "export_tables"):
        monkeypatch.setattr(gb, name, lambda *a, **k: None)
    monkeypatch.setattr(gb, "collect_cell", lambda *a, **k: {"metrics": {}})
    source = make_protocol_dataset(tmp_path / "data", n=3)
    out = tmp_path / "cell"
    gb.submit(_submit_args(source, out, 3))
    service.finish("batches/1", answered=["scenario_1"], errored=["scenario_2"])   # scenario_3 missing

    with pytest.raises(RuntimeError, match="2 request"):
        gb.fetch(SimpleNamespace(out_dir=out))
    assert set(json.loads((out / "batch_errors.json").read_text())) == {"scenario_2", "scenario_3"}

    gb.retry(SimpleNamespace(out_dir=out))
    state = _state(out)
    assert [j["keys"] for j in state["jobs"]] == [["scenario_1", "scenario_2", "scenario_3"],
                                                  ["scenario_2", "scenario_3"]]
    assert state["jobs"][1]["retry"] is True
    with pytest.raises(RuntimeError, match="not finished"):
        gb.retry(SimpleNamespace(out_dir=out))       # never resend while a job still runs

    service.finish("batches/2", answered=["scenario_2", "scenario_3"])
    gb.retry(SimpleNamespace(out_dir=out))           # nothing left: no new job
    assert len(service.jobs) == 2
    gb.fetch(SimpleNamespace(out_dir=out))
    cell = out / "cells" / "nuscenes"
    assert sorted(p.parent.name for p in cell.glob("*/scenario_*/run_manifest.json")) == [
        "scenario_1", "scenario_2", "scenario_3"]


def test_a_failed_or_expired_job_is_retried_whole(tmp_path, monkeypatch):
    from scripts import run_gemini_batch as gb
    from tests.test_evaluation_campaign import make_protocol_dataset
    service = _FakeService(gb, monkeypatch)
    source = make_protocol_dataset(tmp_path / "data", n=2)
    out = tmp_path / "cell"
    gb.submit(_submit_args(source, out, 2))
    service.finish("batches/1", state="BATCH_STATE_EXPIRED")
    assert gb.status(SimpleNamespace(out_dir=out)) == "BATCH_STATE_EXPIRED"
    gb.retry(SimpleNamespace(out_dir=out))
    assert [j["keys"] for j in _state(out)["jobs"]][1] == ["scenario_1", "scenario_2"]
    assert gb.status(SimpleNamespace(out_dir=out)) == "RUNNING"


def test_the_panel_retries_only_with_confirmation(tmp_path, monkeypatch, capsys):
    from scripts import run_api_panel as P
    cell = tmp_path / P.GEMINI / "carla_video"
    cell.mkdir(parents=True)
    (cell / "batch_state.json").write_text(json.dumps({"mapping": {"a": "x", "b": "y"}, "jobs": [{}]}))
    monkeypatch.setattr(P.gemini_batch, "_collect",
                        lambda out, state: ({"a": {}}, {"b": "error"}, []))
    calls = []
    monkeypatch.setattr(P.gemini_batch, "retry", lambda args: calls.append(args.out_dir))
    kw = dict(action="retry", out_root=tmp_path, splits=["carla"], conditions=["video"], scenarios=None)
    P.cmd_gemini(SimpleNamespace(confirm=False, **kw))
    assert "1 request(s) without an answer" in capsys.readouterr().out and not calls
    P.cmd_gemini(SimpleNamespace(confirm=True, **kw))
    assert calls == [cell]


def test_the_panel_prices_a_probe_without_scoring_it(tmp_path, monkeypatch, capsys):
    from scripts import run_api_panel as P
    cell = tmp_path / P.GEMINI / "nuscenes_video"
    cell.mkdir(parents=True)
    (cell / "batch_state.json").write_text(json.dumps({"mapping": {"a": "x", "b": "y"}, "jobs": [{}]}))
    answer = {"usageMetadata": {"promptTokenCount": 16000, "candidatesTokenCount": 14000,
                                "thoughtsTokenCount": 10000},
              "candidates": [{"finishReason": "STOP"}]}
    monkeypatch.setattr(P.gemini_batch, "_collect",
                        lambda out, state: ({"a": answer, "b": answer}, {}, []))
    P.cmd_gemini(SimpleNamespace(action="usage", out_root=tmp_path, splits=["nuscenes"],
                                 conditions=["video"], scenarios=None, confirm=False))
    out = capsys.readouterr().out
    rest = len(P.choose("nuscenes", None)) - 2
    per = (16000 * 0.625 + 24000 * 5.0) / 1e6
    assert "2 answer(s), 0 failed" in out and "finish {'STOP': 2}" in out
    assert f"${per:.3f}/request" in out and f"{rest} not yet submitted, about ${rest * per:.2f}" in out
    assert not list(cell.glob("cells"))                # nothing fetched or scored


def test_a_large_request_file_gets_time_to_upload(monkeypatch):
    """The upload is one sendall whose timeout covers the whole transfer, so a
    175 MB CARLA file on a shared uplink needs more than the per-call default."""
    from scripts import run_gemini_batch as gb

    class Started:
        headers = {"X-Goog-Upload-URL": "https://upload.example/x"}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(gb.urllib.request, "urlopen", lambda req, timeout: Started())
    seen = []

    def http(url, data, headers, method="POST", timeout=None):
        seen.append(timeout)
        return json.dumps({"file": {"name": "files/x"}}).encode()

    monkeypatch.setattr(gb, "_http", http)
    assert gb._upload("k", b"x" * 10) == "files/x"
    gb._upload("k", b"x" * 200_000_000)
    assert seen[0] == gb.api_backend.TIMEOUT_S
    assert seen[1] >= 200_000_000 / 100_000          # allows a 100 kB/s uplink or slower


def test_an_upload_that_breaks_loses_only_its_own_file(tmp_path, monkeypatch):
    """Job files are uploaded one by one and recorded as each job is created,
    so a broken upload costs only that file: the next submit sends the rest."""
    from scripts import run_gemini_batch as gb
    from tests.test_evaluation_campaign import make_protocol_dataset
    service = _FakeService(gb, monkeypatch)
    monkeypatch.setattr(gb, "CHUNK_LIMIT", 1)          # one request per job file
    uploads = []

    def upload(key, raw):
        uploads.append(raw)
        if len(uploads) == 2:
            raise gb.urllib.error.URLError(BrokenPipeError(32, "Broken pipe"))
        return f"files/{len(uploads)}"

    monkeypatch.setattr(gb, "_upload", upload)
    source = make_protocol_dataset(tmp_path / "data", n=3)
    out = tmp_path / "cell"
    with pytest.raises(gb.urllib.error.URLError):
        gb.submit(_submit_args(source, out, 3))
    assert [j["keys"] for j in _state(out)["jobs"]] == [["scenario_1"]]
    gb.submit(_submit_args(source, out, 3))
    assert [j["keys"] for j in _state(out)["jobs"]] == [["scenario_1"], ["scenario_2"], ["scenario_3"]]


def test_job_files_stay_small_enough_to_upload_on_a_slow_link():
    from scripts import run_gemini_batch as gb
    assert gb.CHUNK_LIMIT <= 500_000_000


def test_a_cell_keeps_its_prompt_variant_from_submit_to_fetch(batch):
    gb, source, out, jobs = batch
    args = _submit_args(source, out, 1)
    args.prompt_variant = "recipe_withheld"
    gb.submit(args)
    assert _state(out)["prompt_variant"] == "recipe_withheld"
    line = json.loads((out / "batch_input_0.jsonl").read_text().splitlines()[0])
    text = json.dumps(line["request"])
    assert "Motion parameters: not supplied" in text and "Published motion profiles" not in text
    with pytest.raises(ValueError, match="variant"):
        gb.submit(_submit_args(source, out, 2))       # the default `full` into an arm-B cell


def test_a_cell_freezes_its_system_prompt_and_generation_config(batch):
    gb, source, out, jobs = batch
    gb.submit(_submit_args(source, out, 1))
    state = _state(out)
    fp = state["request_fingerprint"]
    assert fp["system_sha256"] and fp["generation_config"] == {"temperature": 0.0, "maxOutputTokens": 65536}
    line = json.loads((out / "batch_input_0.jsonl").read_text().splitlines()[0])
    gb._check_fingerprint(state, line["request"], "fetch")          # unchanged: accepted
    edited = json.loads(json.dumps(line["request"]))
    edited["systemInstruction"]["parts"][0]["text"] += "\nA new rule."
    with pytest.raises(RuntimeError, match="system prompt or generation config changed"):
        gb._check_fingerprint(state, edited, "fetch")
    edited = json.loads(json.dumps(line["request"])); edited["generationConfig"]["temperature"] = 0.7
    with pytest.raises(RuntimeError, match="changed"):
        gb._check_fingerprint(state, edited, "fetch")
    assert gb._check_fingerprint({"mapping": {}}, edited, "fetch") is None   # a pre-check state is accepted


def test_retry_refuses_a_changed_system_prompt(batch, monkeypatch):
    gb, source, out, jobs = batch
    gb.submit(_submit_args(source, out, 2))
    monkeypatch.setattr(gb, "_collect", lambda out_, state: ({"scenario_1": {}}, {"scenario_2": "error"}, []))
    state = _state(out)
    state["request_fingerprint"] = dict(state["request_fingerprint"], system_sha256="0" * 64)
    (out / "batch_state.json").write_text(json.dumps(state))
    with pytest.raises(RuntimeError, match="system prompt or generation config changed"):
        gb.retry(SimpleNamespace(out_dir=out))
    assert len(jobs) == 1                                   # nothing was resent


def test_verify_records_the_fingerprint_of_a_cell_submitted_before_the_check(batch):
    gb, source, out, jobs = batch
    gb.submit(_submit_args(source, out, 2))
    state = _state(out); fingerprint = state.pop("request_fingerprint")
    (out / "batch_state.json").write_text(json.dumps(state))
    gb.verify(SimpleNamespace(out_dir=out, rebuild=True))   # rebuilds scenario_1 through the current code
    assert _state(out)["request_fingerprint"] == fingerprint
    lines = [json.loads(l) for l in (out / "batch_input_0.jsonl").read_text().splitlines()]
    for rec in lines:
        rec["request"]["systemInstruction"]["parts"][0]["text"] += "\nA rule that was never sent."
    (out / "batch_input_0.jsonl").write_text("\n".join(json.dumps(r) for r in lines) + "\n")
    with pytest.raises(RuntimeError, match="state's fingerprint differs"):
        gb.verify(SimpleNamespace(out_dir=out, rebuild=False))
    state = _state(out); state.pop("request_fingerprint")
    (out / "batch_state.json").write_text(json.dumps(state))
    with pytest.raises(RuntimeError, match="current code builds another system instruction"):
        gb.verify(SimpleNamespace(out_dir=out, rebuild=True))
    lines[0]["request"]["systemInstruction"]["parts"][0]["text"] = "different again"
    (out / "batch_input_0.jsonl").write_text("\n".join(json.dumps(r) for r in lines) + "\n")
    with pytest.raises(RuntimeError, match="one system instruction"):
        gb.verify(SimpleNamespace(out_dir=out, rebuild=False))


def test_verify_checks_every_saved_line_so_a_retry_under_another_prompt_is_caught(batch):
    gb, source, out, jobs = batch
    gb.submit(_submit_args(source, out, 1))
    line = json.loads((out / "batch_input_0.jsonl").read_text().splitlines()[0])
    resent = json.loads(json.dumps(line))
    resent["request"]["systemInstruction"]["parts"][0]["text"] += "\nA rule the retry added."
    (out / "batch_input_1.jsonl").write_text(json.dumps(resent) + "\n")        # same scenario, other system text
    with pytest.raises(RuntimeError, match="one system instruction"):
        gb.verify(SimpleNamespace(out_dir=out, rebuild=False))
    resent = json.loads(json.dumps(line))
    resent["request"]["contents"][0]["parts"][0]["text"] += " changed"          # same scenario, other prompt text
    (out / "batch_input_1.jsonl").write_text(json.dumps(resent) + "\n")
    with pytest.raises(RuntimeError, match="prompt digest differs"):
        gb.verify(SimpleNamespace(out_dir=out, rebuild=False))


def test_a_probe_fetched_alone_can_be_fetched_again_with_the_rest_of_the_cell(tmp_path, monkeypatch):
    from scripts import run_gemini_batch as gb
    from tests.test_evaluation_campaign import make_protocol_dataset
    service = _FakeService(gb, monkeypatch)
    for name in ("score_cell", "export_tables"):
        monkeypatch.setattr(gb, name, lambda *a, **k: None)
    monkeypatch.setattr(gb, "collect_cell", lambda *a, **k: {"metrics": {}})
    source = make_protocol_dataset(tmp_path / "data", n=3)
    out = tmp_path / "cell"
    gb.submit(_submit_args(source, out, 1))
    service.finish("batches/1", answered=["scenario_1"])
    gb.fetch(SimpleNamespace(out_dir=out))                   # the probe, scored on its own
    identity = json.loads((out / "cells" / "nuscenes" / "gemini-2.5-pro_video" / "campaign_identity.json").read_text())
    gb.submit(_submit_args(source, out, 3))
    service.finish("batches/2", answered=["scenario_2", "scenario_3"])
    gb.fetch(SimpleNamespace(out_dir=out))                   # the whole cell: a new identity, not a conflict
    cell = out / "cells" / "nuscenes" / "gemini-2.5-pro_video"
    assert sorted(p.parent.name for p in cell.glob("scenario_*/run_manifest.json")) == [
        "scenario_1", "scenario_2", "scenario_3"]
    assert json.loads((cell / "campaign_identity.json").read_text())["frozen_sha256"] != identity["frozen_sha256"]


def test_an_answer_that_spent_the_cap_on_thoughts_is_recorded_empty(monkeypatch):
    """Gemini can spend the whole output cap on thoughts it does not return:
    the candidate then has no parts. That is the model's empty answer at the
    cap (recorded, not retried); a candidate without parts that did not stop at
    the cap is still a malformed response."""
    import pytest
    from analysis.backends import api
    usage = {"promptTokenCount": 17869, "candidatesTokenCount": 0, "thoughtsTokenCount": 65533}
    capped = {"candidates": [{"content": {"role": "model"}, "finishReason": "MAX_TOKENS", "index": 0}],
              "usageMetadata": usage, "modelVersion": "gemini-2.5-pro"}
    monkeypatch.setattr(api, "_post_json", lambda *a, **k: capped)
    result = api.gemini_generate_result("p", [], "gemini-2.5-pro", api_key="k", max_output_tokens=65536)
    assert result.final_text == "" and result.raw_text == ""
    assert result.metadata["finish_reason"] == "MAX_TOKENS" and result.metadata["thinking_tokens"] == 65533
    broken = {"candidates": [{"content": {"role": "model"}, "finishReason": "STOP"}], "usageMetadata": usage}
    monkeypatch.setattr(api, "_post_json", lambda *a, **k: broken)
    with pytest.raises(api.ApiError, match="unexpected Gemini response shape"):
        api.gemini_generate_result("p", [], "gemini-2.5-pro", api_key="k")
