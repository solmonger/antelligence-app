"""S3 worker backends: logprob extraction, identity, cache replay, d1 request shape, Nous verbalised parse. No network."""
from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
from pathlib import Path

import httpx
import numpy as np
import pytest

from antelligence.kernel.decision import Answer, DecisionError, DecisionRequest, Question
from antelligence.providers.decision_backends import (CachedDecision, D1Backend, LlamaCppBackend, NousFWBackend,
                                                      VLLMLogprobBackend, answer_schema, extract_label_probs,
                                                      patch_to_images)
from tests.v4_1.workers.helpers import (BONSAI, BRIEF, DEFAULT_CHOSEN, QWEN9, Recorder, chat_reply, default_tops,
                                        make_request, split_tokens)

REPO = Path(__file__).resolve().parents[3]


def run(coro):
    return asyncio.run(coro)


def vllm(rec: Recorder, **kw) -> VLLMLogprobBackend:
    return VLLMLogprobBackend("http://pod:8100", *QWEN9, runtime_version="0.30.0", api_key="key-123", retries=0,
                              transport=rec.transport(), **kw)


def served(hf_rev=QWEN9) -> str:
    return f"{hf_rev[0]}@{hf_rev[1]}"


def vllm_recorder(request, *, model=None, version="0.30.0", cites=None, tops=None, chosen=None) -> Recorder:
    reply = chat_reply(request, chosen or DEFAULT_CHOSEN, tops or default_tops(), model or served(), cites)
    return Recorder(lambda r: httpx.Response(200, json=reply), version={"version": version})


# --- logprob extraction -----------------------------------------------------------------------------------------


def test_logprob_extraction_single_token_labels():
    req = make_request(served())
    content = json.dumps(DEFAULT_CHOSEN)
    reply = chat_reply(req, DEFAULT_CHOSEN, default_tops(), "m")
    probs, info = extract_label_probs(req, content, reply["choices"][0]["logprobs"]["content"])
    # target: offered labels A,B,C,S,X (D is not offered and is dropped); renormalised over the offered mass
    t = probs["target"]
    assert set(t) == {"A", "B", "C", "S", "X"}
    assert t["B"] == pytest.approx(0.6 / 0.99) and t["A"] == pytest.approx(0.3 / 0.99)
    assert sum(t.values()) == pytest.approx(1.0)
    assert info["residual"]["target"] == pytest.approx(0.01)  # 1 - offered mass: the D token and the unlogged tail
    assert probs["release"] == {"yes": pytest.approx(0.8 / 0.95), "no": pytest.approx(0.15 / 0.95)}
    a = probs["amount"]  # digits follow a quote: one token each
    assert set(a) == {"1", "2", "3", "4", "5"} and a["3"] == pytest.approx(0.5)
    assert info["labels"] == DEFAULT_CHOSEN


def test_missing_labels_get_a_floor_and_degrade(caplog):
    req = make_request(served())
    tops = default_tops()
    tops["target"] = {"B": 0.7, "A": 0.2}  # C, S, X absent from the top list: residual 0.1 over 3 labels
    reply = chat_reply(req, DEFAULT_CHOSEN, tops, "m")
    entries = reply["choices"][0]["logprobs"]["content"]
    with caplog.at_level("WARNING"):
        probs, info = extract_label_probs(req, json.dumps(DEFAULT_CHOSEN), entries)
    floor = min(0.1 / 3, 0.2)  # residual / n_missing, capped at the smallest found probability
    t = probs["target"]
    assert info["degraded"] == ["target"] and "absent from the top" in caplog.text
    assert t["C"] == t["S"] == t["X"] == pytest.approx(floor / (0.9 + 3 * floor))
    assert t["B"] > t["A"] > t["C"] and sum(t.values()) == pytest.approx(1.0)  # a seen label always outranks an unseen one
    # a tiny found mass caps the floor at the smallest found probability
    tops["target"] = {"B": 0.02}
    probs2, _ = extract_label_probs(req, json.dumps(DEFAULT_CHOSEN), chat_reply(req, DEFAULT_CHOSEN, tops, "m")["choices"][0]["logprobs"]["content"])
    assert probs2["target"]["A"] == pytest.approx(probs2["target"]["B"]) and probs2["target"]["B"] == pytest.approx(1 / 5)
    # nothing degraded when every label is seen; the reply flag follows
    full = {"target": {"B": 0.5, "A": 0.2, "C": 0.1, "S": 0.1, "X": 0.1}, "release": {"yes": 0.8, "no": 0.2},
            "amount": {"3": 0.4, "1": 0.15, "2": 0.15, "4": 0.15, "5": 0.15}}
    rec = vllm_recorder(req, tops=full)
    assert run(vllm(rec).decide(req)).degraded is False
    rec = vllm_recorder(req, tops={**full, "target": {"B": 0.7, "A": 0.2}})
    resp = run(vllm(rec).decide(req))
    assert resp.degraded is True and resp.answers["target"].value == "B"


def test_cache_roundtrips_new_flags(tmp_path):
    req = make_request(served())
    rec = vllm_recorder(req, tops={**default_tops(), "target": {"B": 0.7, "A": 0.2}})
    first = run(CachedDecision(vllm(rec), tmp_path).decide(req))
    again = run(CachedDecision(vllm(Recorder(lambda r: httpx.Response(500))), tmp_path, offline=True).decide(req))
    assert first.degraded and again.degraded and again.reasoning_tokens == 0 and again.alarm == {}


def test_golden_pre_v41_request_hash():
    """A pre-v4.1 DecisionRequest hashes exactly as it did on the integration base (bee3ab9): cache keys survive."""
    q = Question
    req = DecisionRequest("jev-1.13.0", "Round 3 of 24. Payload left: 50%.", seed=7, questions={
        "target": q("choice", "Where next?", ("A", "B", "S", "X"), {"A": "region A", "B": "region B", "S": "stay", "X": "explore"}),
        "release": q("noul", "Release?", criteria={"true": "release", "false": "keep"}),
        "amount": q("score", "How much?", ("1", "2", "3", "4", "5"))})
    assert req.request_hash == "22f4f64eb9f6782f5ef4b900648ab6ac315315503281cd646825b76b7b783212"
    assert "images" not in req.payload() and "brief_ids" not in req.payload()


def test_logprob_extraction_rejects_multi_token_labels():
    req = make_request(served())
    content = '{"target": "B", "release": "yes", "amount": " 3"}'  # leading-space digit = two tokens on Qwen/Bonsai (L2)
    tokens = ['{"target": "', "B", '", "release": "', "yes", '", "amount": "', " ", "3", '"}']
    entries = [{"token": t, "logprob": -0.1, "top_logprobs": [{"token": t, "logprob": -0.1}]} for t in tokens]
    with pytest.raises(DecisionError, match="amount"):
        extract_label_probs(req, content, entries)
    with pytest.raises(DecisionError, match="reassemble"):
        extract_label_probs(req, content.replace("yes", "no"), entries)
    with pytest.raises(DecisionError, match="no per-token logprobs"):
        extract_label_probs(req, content, None)


def test_logprob_extraction_with_whitespace_and_compact_json():
    req = make_request(served())
    for content in ('{"target":"A","release":"no","amount":"5"}', '{ "target" : "A", "release" : "no", "amount" : "5" }'):
        tokens = split_tokens(content, req)
        entries = [{"token": t, "logprob": -0.2, "top_logprobs": [{"token": t, "logprob": -0.2}]} for t in tokens]
        probs, info = extract_label_probs(req, content, entries)
        assert info["labels"] == {"target": "A", "release": "no", "amount": "5"}
        assert probs["target"]["A"] == max(probs["target"].values()) and info["degraded"]  # others unseen -> floored


# --- real tokenizers (cached only) ------------------------------------------------------------------------------

TOKENIZERS = {
    "qwen3.5-9b": ("RedHatAI/Qwen3.5-9B-FP8-dynamic", "790f0576d2d77dd5322aa0603a470bd9e3a3d1f6"),
    "qwen3.5-4b": ("RedHatAI/Qwen3.5-4B-FP8-dynamic", "397b7ba47a99b3221ebfc0cfc5a279118cb733ad"),
    "bonsai-8b": ("prism-ml/Ternary-Bonsai-8B-unpacked", "ac20f03fc62e872399218b659c8e949dfca05769"),
}


@pytest.mark.parametrize("name", sorted(TOKENIZERS))
def test_answer_json_tokenises_to_single_token_labels(name):
    """The exact JSON the schema forces, tokenised by the real tokenizer: every answer value is one token."""
    try:
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer
    except ImportError:
        pytest.skip("huggingface_hub/tokenizers not installed")
    repo, rev = TOKENIZERS[name]
    try:
        path = hf_hub_download(repo, "tokenizer.json", revision=rev, local_files_only=True)
    except Exception:
        pytest.skip(f"{name}: tokenizer.json for {repo}@{rev[:8]} not in the HF cache")
    tok = Tokenizer.from_file(path)
    req = make_request("m", labels=("A", "B", "C", "D", "E", "F"))
    labels = {"target": ["A", "F", "S", "X"], "release": ["yes", "no"], "amount": ["1", "3", "5"]}
    for sep in (", ", ","):
        for tgt in labels["target"]:
            for rel in labels["release"]:
                for amt in labels["amount"]:
                    content = json.dumps({"target": tgt, "release": rel, "amount": amt}, separators=(sep, ": " if sep == ", " else ":"))
                    enc = tok.encode(content, add_special_tokens=False)
                    tokens = [content[a:b] for a, b in enc.offsets]
                    assert "".join(tokens) == content
                    entries = [{"token": t, "logprob": -0.1, "top_logprobs": [{"token": t, "logprob": -0.1}]} for t in tokens]
                    _, info = extract_label_probs(req, content, entries)  # raises if any label is not one token
                    assert info["labels"] == {"target": tgt, "release": rel, "amount": amt}, (name, content)


def test_schema_enums_are_strings_in_question_order():
    req = make_request(served(), brief_ids=BRIEF)
    schema = answer_schema(req, req.brief_ids)
    assert list(schema["properties"]) == ["target", "release", "amount", "cites"]
    assert schema["properties"]["target"]["enum"] == ["A", "B", "C", "S", "X"]
    assert schema["properties"]["amount"]["enum"] == ["1", "2", "3", "4", "5"]
    assert schema["properties"]["release"]["enum"] == ["yes", "no"]
    assert schema["additionalProperties"] is False and schema["required"] == list(schema["properties"])


# --- vLLM / llama.cpp backends ----------------------------------------------------------------------------------


def test_vllm_backend_end_to_end():
    req = make_request(served(), brief_ids=BRIEF)
    rec = vllm_recorder(req, cites=[BRIEF[1][:8], BRIEF[0][:8]])
    resp = run(vllm(rec).decide(req))
    body = json.loads(rec.posts[0].content)
    assert body["temperature"] == 0.0 and body["logprobs"] is True and body["top_logprobs"] == 20
    assert body["model"] == served() and body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["response_format"]["type"] == "json_schema"
    assert rec.posts[0].url.path == "/v1/chat/completions" and rec.posts[0].headers["authorization"] == "Bearer key-123"
    assert resp.prob_method == "logprobs"
    assert resp.identity == {"served_model": QWEN9[0], "revision": QWEN9[1], "runtime": "vllm", "runtime_version": "0.30.0"}
    assert resp.cites == (BRIEF[0], BRIEF[1])  # short ids mapped back to full ids, sorted
    assert resp.answers["target"].value == "B" and resp.answers["release"].p_yes == pytest.approx(0.8 / 0.95)
    assert resp.answers["amount"].value == pytest.approx(sum(i * p for i, p in enumerate([0.03, 0.25, 0.5, 0.2, 0.02])) / 1.0, abs=0.02)
    assert set(resp.timing) == {"server_ms", "network_ms", "queue_ms"} and resp.usage["tokens_in"] == 120


def test_identity_mismatch_raises():
    req = make_request(served())
    # wrong revision in the served name
    rec = vllm_recorder(req, model=f"{QWEN9[0]}@{'0' * 40}")
    with pytest.raises(DecisionError, match="identity mismatch: revision"):
        run(vllm(rec).decide(req))
    # wrong model
    rec = vllm_recorder(req, model="RedHatAI/Qwen3.5-4B-FP8-dynamic@" + QWEN9[1])
    with pytest.raises(DecisionError, match="identity mismatch: served_model"):
        run(vllm(rec).decide(req))
    # wrong runtime version
    rec = vllm_recorder(req, version="0.17.1")
    with pytest.raises(DecisionError, match="runtime_version"):
        run(vllm(rec).decide(req))
    # a llama.cpp server answering where vLLM is pinned: /version has no "version"
    rec = vllm_recorder(req)
    rec.version = {"build_info": "b1"}
    with pytest.raises(DecisionError, match="does not look like vLLM"):
        run(vllm(rec).decide(req))
    # d1: the shim reports a different revision
    d1req = make_request("LiquidAI/d1-3B")
    answers = _d1_reply(d1req)
    answers["identity"]["revision"] = "f" * 40
    d1 = D1Backend("http://pod:8102", retries=0, transport=httpx.MockTransport(lambda r: httpx.Response(200, json=answers)))
    with pytest.raises(DecisionError, match="identity mismatch: revision"):
        run(d1.decide(d1req))
    # Nous served a different model
    nreq = make_request("deepseek/deepseek-v4.1-flash")
    nous = _nous(nreq, model="deepseek/other")
    with pytest.raises(DecisionError, match="identity mismatch: served_model"):
        run(nous.decide(nreq))


def test_cache_replay_identical(tmp_path):
    req = make_request(served(), brief_ids=BRIEF)
    rec = vllm_recorder(req, cites=[BRIEF[2][:8]])
    backend = vllm(rec)
    cached = CachedDecision(backend, tmp_path)
    first = run(cached.decide(req))
    n_posts = len(rec.posts)
    second = run(cached.decide(req))
    assert not first.cached and second.cached and len(rec.posts) == n_posts == 1
    for field in ("answers", "identity", "timing", "cites", "prob_method", "served_model", "request_hash",
                  "input_tokens", "output_tokens", "cost_usd"):
        assert getattr(first, field) == getattr(second, field), field
    assert first.accounting() == second.accounting()
    # a fresh process (new CachedDecision, offline) replays the same bytes from disk and never calls the backend
    offline = CachedDecision(vllm(Recorder(lambda r: httpx.Response(500))), tmp_path, offline=True)
    third = run(offline.decide(req))
    assert third.cached and third.answers == first.answers and third.identity == first.identity
    # identity is part of the request hash only through the model name; images change the hash
    assert make_request(served(), images=("QUJD",)).request_hash != make_request(served()).request_hash


def test_llamacpp_backend_logprobs_prefill_and_onehot_fallback():
    model = f"{BONSAI[0]}@{BONSAI[1]}"
    req = make_request(model)
    reply = chat_reply(req, DEFAULT_CHOSEN, default_tops(), model)
    rec = Recorder(lambda r: httpx.Response(200, json=reply), props={"build_info": "b7000-abc"})
    be = LlamaCppBackend("http://mac:8080", *BONSAI, runtime_version="b7000-abc", prefill_empty_think=True, retries=0,
                         transport=rec.transport())
    resp = run(be.decide(req))
    body = json.loads(rec.posts[0].content)
    assert body["messages"][-1] == {"role": "assistant", "content": "<think>\n\n</think>\n\n"}
    assert body["response_format"]["json_schema"]["schema"]["properties"]["target"]["enum"] == ["A", "B", "C", "S", "X"]
    assert resp.identity["runtime"] == "llama.cpp" and resp.identity["runtime_version"] == "b7000-abc"
    assert resp.prob_method == "logprobs"
    # no logprobs from the build: fails by default, one-hot (flagged) when allowed
    bare = json.loads(json.dumps(reply))
    bare["choices"][0].pop("logprobs")
    mk = lambda **kw: LlamaCppBackend("http://mac:8080", *BONSAI, retries=0, **kw,
                                      transport=Recorder(lambda r: httpx.Response(200, json=bare), props={"build_info": "b1"}).transport())
    with pytest.raises(DecisionError, match="no per-token logprobs"):
        run(mk().decide(req))
    one = run(mk(onehot_fallback=True).decide(req))
    assert one.prob_method == "onehot" and one.answers["target"].probabilities["B"] == 1.0


# --- d1 ---------------------------------------------------------------------------------------------------------


def _d1_reply(req):
    answers = {}
    for qid, q in req.questions.items():
        if q.type == "noul":
            answers[qid] = {"type": "noul", "noul": 0.7}
        elif q.type == "choice":
            probs = {o: (0.6 if i == 0 else 0.4 / (len(q.options) - 1)) for i, o in enumerate(q.options)}
            answers[qid] = {"type": "choice", "choice": q.options[0], "confidence": 0.6, "probabilities": probs}
        else:
            answers[qid] = {"type": "score", "score": 2.0, "confidence": 0.5,
                            "probabilities": {"0": 0.1, "1": 0.2, "2": 0.5, "3": 0.1, "4": 0.1}}
    return {"model": "LiquidAI/d1-3B",
            "identity": {"served_model": "LiquidAI/d1-3B", "revision": "051bcc464b01b9f92942b364d9586b0ef5912432",
                         "runtime": "transformers", "runtime_version": "5.14.0"},
            "answers": answers, "usage": {"input_tokens": 80, "output_tokens": 0}, "timing": {"server_ms": 41, "queue_ms": 2}}


def test_d1_backend_request_shape_with_images():
    patch = np.random.default_rng(0).random((64, 64, 2)).astype(np.float32)
    flair, t1c = patch_to_images(patch)
    for img in (flair, t1c):
        assert base64.b64decode(img)[:8] == b"\x89PNG\r\n\x1a\n"
    assert (flair, t1c) == patch_to_images(patch) and flair != t1c  # deterministic, channels differ
    req = make_request("LiquidAI/d1-3B", images=(flair, t1c))
    reply = _d1_reply(req)
    rec = Recorder(lambda r: httpx.Response(200, json=reply))
    be = D1Backend("http://pod:8102", runtime_version="5.14.0", retries=0, api_key="tok", transport=rec.transport())
    resp = run(be.decide(req))
    post = rec.posts[0]
    body = json.loads(post.content)
    assert post.url.path == "/v1/systemone" and post.headers["authorization"] == "Bearer tok"
    assert set(body) == {"model", "state", "questions", "images"} and body["images"] == [flair, t1c]
    assert body["model"] == "LiquidAI/d1-3B" and set(body["questions"]) == {"target", "release", "amount"}
    assert body["questions"]["target"]["type"] == "choice" and list(body["questions"]["target"]["criteria"]) == ["A", "B", "C", "S", "X"]
    assert body["questions"]["amount"] == {"type": "score", "instructions": req.questions["amount"].instructions,
                                           "criteria": ["1", "2", "3", "4", "5"]}
    assert "brief_ids" not in body  # the read receipt is not part of the wire
    assert resp.prob_method == "system_one" and resp.cites == () and resp.timing["server_ms"] == 41
    assert resp.identity["revision"] == "051bcc464b01b9f92942b364d9586b0ef5912432"
    # no images -> no key (pre-v4.1 hashes unchanged)
    assert "images" not in make_request("LiquidAI/d1-3B").payload()

    # the shim accepts exactly this body and returns what the backend parses
    spec = importlib.util.spec_from_file_location("v41_d1_server", REPO / "scripts" / "v41_d1_server.py")
    shim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shim)

    class FakeModel:
        seen = None

        def system_one_batch(self, requests):
            FakeModel.seen = requests
            return [{"answers": reply["answers"], "usage": reply["usage"]}]

    identity = reply["identity"]
    out = shim.handle_request(FakeModel(), identity, body)
    state, questions, images = FakeModel.seen[0]
    assert state == req.state and questions == body["questions"] and len(images) == 2 and images[0].size == (64, 64)
    assert out["identity"] == identity and out["answers"] == reply["answers"] and out["timing"]["queue_ms"] == 0
    for bad in ({**body, "model": "other"}, {**body, "images": ["not base64!"]}, {**body, "questions": {}}):
        with pytest.raises(shim.BadRequest):
            shim.handle_request(FakeModel(), identity, bad)


# --- Nous / DeepSeek V4.1 Flash ---------------------------------------------------------------------------------


def _nous(req, *, content=None, model="deepseek/deepseek-v4.1-flash", finish="stop", thinking=None, reasoning_tokens=None):
    content = content if content is not None else json.dumps({
        "target": {"A": 0.5, "B": 0.3, "C": 0.1, "S": 0.05, "X": 0.05}, "release": 0.9,
        "amount": {"0": 0.0, "1": 0.1, "2": 0.6, "3": 0.2, "4": 0.1}, "cites": [BRIEF[0][:8], "deadbeef"]})
    reply = {"model": model, "choices": [{"finish_reason": finish, "message": {"content": content}}],
             "usage": {"prompt_tokens": 300, "completion_tokens": 60}}
    if reasoning_tokens is not None:
        reply["usage"]["completion_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
    rec = Recorder(lambda r: httpx.Response(200, json=reply))
    be = NousFWBackend("https://nous.test/v1", api_key="nk", retries=0, transport=rec.transport(),
                       **({"thinking_param": thinking} if thinking is not None else {}))
    be.rec = rec
    return be


def test_nous_fw_verbalised_parse():
    req = make_request("deepseek/deepseek-v4.1-flash", brief_ids=BRIEF)
    be = _nous(req)
    resp = run(be.decide(req))
    body = json.loads(be.rec.posts[0].content)
    assert be.rec.posts[0].url.path == "/v1/chat/completions" and body["reasoning"] == {"enabled": False} and "thinking" not in body
    assert body["temperature"] == 0.0 and body["response_format"] == {"type": "json_object"}
    assert resp.prob_method == "verbalised"
    assert resp.identity == {"served_model": "deepseek/deepseek-v4.1-flash", "revision": "none",
                             "runtime": "nous-portal", "runtime_version": "none"}
    assert resp.answers["target"].value == "A" and resp.answers["release"].p_yes == 0.9
    assert resp.answers["amount"].probabilities["2"] == pytest.approx(0.6)
    assert resp.cites == (BRIEF[0],)  # the unknown id "deadbeef" is dropped
    assert resp.cost_usd == pytest.approx((300 * 0.106 + 60 * 0.513) / 1e6)
    # thinking switch is configurable; fenced replies parse; cut-off replies fail
    assert resp.reasoning_tokens == 0 and resp.alarm == {}
    be2 = _nous(req, thinking={"thinking": {"type": "disabled"}})
    run(be2.decide(req))
    assert "reasoning" not in json.loads(be2.rec.posts[0].content)
    # hidden reasoning tokens with thinking off -> recorded, and an alarm payload (INTERFACES section 4) is attached
    loud = _nous(req, reasoning_tokens=37)
    r = run(loud.decide(req))
    assert r.reasoning_tokens == 37 and r.alarm["kind"] == "other" and r.alarm["value_ppm"] == 37 and len(r.alarm["note"]) <= 140
    fenced = _nous(req, content="```json\n" + json.dumps({"target": {"A": 1.0}, "release": 0.1, "amount": {"2": 1.0}}) + "\n```")
    assert run(fenced.decide(req)).answers["target"].value == "A"
    with pytest.raises(DecisionError, match="finish normally"):
        run(_nous(req, finish="length").decide(req))
    with pytest.raises(ValueError):
        NousFWBackend(api_key="")


def test_logprob_extraction_tolerates_stripped_empty_think_block():
    """vLLM's qwen3 reasoning parser strips the empty think block from content but not from the logprob tokens."""
    import json as _json

    from antelligence.kernel.decision import DecisionRequest, Question
    from antelligence.providers.decision_backends import DecisionError, extract_label_probs

    q = {"release": Question("noul", "release?")}
    req = DecisionRequest("m", "state", q)
    content = _json.dumps({"release": "yes"}, separators=(",", ":"))
    pre = ["<think>", "\n\n", "</think>", "\n\n"]
    body = ['{"', "release", '":"', "yes", '"}']
    assert "".join(body) == content
    toks = [{"token": t, "logprob": -0.1, "top_logprobs": []} for t in pre + body]
    toks[len(pre) + 3]["top_logprobs"] = [{"token": "yes", "logprob": -0.2}, {"token": "no", "logprob": -1.8}]
    probs, meta = extract_label_probs(req, content, toks)
    assert probs["release"]["yes"] > probs["release"]["no"]
    bad = [{"token": "junk", "logprob": -0.1, "top_logprobs": []}] + toks[len(pre):]
    try:
        extract_label_probs(req, content, bad)
    except DecisionError:
        pass
    else:
        raise AssertionError("unexpected prefix text must be refused")


# --- S13: transient 401 is retried on the worker HTTP path ---------------------------------------------------------


def test_worker_backend_retries_transient_401_but_not_403():
    state = {"n": 0}
    base = vllm_recorder(make_request(served()))

    def flaky(r):
        state["n"] += 1
        return httpx.Response(401, text="invalid key") if state["n"] == 1 else base.post(r)
    req = make_request(served())
    rec = Recorder(flaky, version={"version": "0.30.0"})
    b = VLLMLogprobBackend("http://pod:8100", *QWEN9, runtime_version="0.30.0", api_key="key-123", retries=3,
                           backoff_s=0, transport=rec.transport())
    assert run(b.decide(req)).served_model
    assert state["n"] == 2

    rec403 = Recorder(lambda r: httpx.Response(403, text="no"), version={"version": "0.30.0"})
    b = VLLMLogprobBackend("http://pod:8100", *QWEN9, runtime_version="0.30.0", api_key="key-123", retries=3,
                           backoff_s=0, transport=rec403.transport())
    with pytest.raises(DecisionError, match="403"):
        run(b.decide(req))
    assert state["n"] == 2
