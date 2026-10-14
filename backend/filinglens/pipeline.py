"""RAG API query path: retrieve -> cite or refuse -> generate. Yields SSE event dicts."""
from . import llm
from .config import Settings
from .retrieval.service import PRESETS, Retriever, config_from
from .trace import Trace


def _supported(cands: list[dict], settings: Settings) -> bool:
    return any((c["scores"]["bm25"] or 0) >= settings.refusal_min_bm25
               or (c["scores"]["dense"] or 0) >= settings.refusal_min_dense for c in cands)


def _round(scores: dict) -> dict:
    return {k: (None if v is None else round(v, 4)) for k, v in scores.items()}


def _generate(settings: Settings, question: str, citations: list[dict], trace: Trace):
    """Stream tokens; if the model says NOT_FOUND, emit a refusal instead of an answer."""
    usage: dict = {}
    head = ""        # hold back the first few characters to detect the NOT_FOUND sentinel
    decided = False
    with trace.stage("llm"):
        for piece in llm.stream_answer(settings, question, citations, usage):
            if decided:
                yield {"type": "token", "text": piece}
                continue
            head += piece
            if llm.NOT_FOUND.startswith(head.strip()) and len(head.strip()) < len(llm.NOT_FOUND):
                continue  # could still turn into NOT_FOUND
            decided = True
            if head.strip().startswith(llm.NOT_FOUND):
                yield {"type": "refusal", "reason": "Not found in the filings: the passages in scope "
                                                    "don't contain the answer."}
                break
            yield {"type": "token", "text": head}
        else:
            if head and not decided:  # stream ended while still ambiguous (very short answer)
                if head.strip() == llm.NOT_FOUND:
                    yield {"type": "refusal", "reason": "Not found in the filings."}
                else:
                    yield {"type": "token", "text": head}
    trace.prompt_tokens = usage.get("prompt_tokens", 0)
    trace.completion_tokens = usage.get("completion_tokens", len(head.split()))
    trace.cost_usd = round((trace.prompt_tokens * settings.llm_price_in
                            + trace.completion_tokens * settings.llm_price_out) / 1e6, 6)


def run_query(retriever: Retriever, settings: Settings, req: dict):
    trace = Trace()
    question = req["question"].strip()
    config = config_from(req.get("pipeline"), PRESETS[settings.pipeline])
    if req.get("top_k"):
        config = config_from({"top_k": int(req["top_k"])}, config)

    cands, stages = retriever.retrieve(question, req.get("filters"), config)
    trace.stages.extend(stages)

    citations = [{
        "id": n, "chunk_id": c["chunk_id"], "company": c["company"], "ticker": c["ticker"],
        "fiscal_year": c["fiscal_year"], "section": c["section"], "text": c["text"],
        "source_url": c["source_url"], "scores": _round(c["scores"]),
    } for n, c in enumerate(cands, start=1)]
    yield {"type": "citations", "citations": citations}

    if not cands or not _supported(cands, settings):
        yield {"type": "refusal", "reason": "Not found in the filings: the passages in scope "
                                            "don't contain enough information to answer."}
    else:
        yield from _generate(settings, question, citations, trace)
    yield {"type": "done", "trace": trace.to_dict()}
