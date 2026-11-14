"""Workstream A: cleaning, chunking, retrieval, grading/rewrite loop, metrics."""
import pytest

from common.providers import get_embedder
from rag.chunking import Chunk, chunk_corpus, fixed_size, sentence_boundary
from common.events import EventLog
from rag.data import _from_rows, clean_text, load_local_sample
from rag.evaluate import lexical_faithfulness, recall_at_k, reciprocal_rank, retrieval_grid, token_f1
from rag.pipeline import ABSTAIN, RAGPipeline
from rag.retrieval import BM25Retriever, build_retriever, to_passages
from tests.conftest import ScriptedChat


@pytest.fixture(scope="module")
def corpus():
    return load_local_sample()


def test_clean_text():
    assert clean_text("  Tom &amp; Jerry\n\n\tare\x00 here ") == "Tom & Jerry are here"


def test_local_sample_is_consistent(corpus):
    assert len(corpus.queries) >= 15 and len(corpus.passages) >= 40
    for q in corpus.queries:
        assert q.relevant_ids and q.answers
        assert all(pid in corpus.passages for pid in q.relevant_ids)


def test_fixed_size_chunking_overlaps_and_covers_everything():
    words = [f"w{i}" for i in range(25)]
    chunks = fixed_size(" ".join(words), size=10, overlap=3)
    assert all(len(c.split()) <= 10 for c in chunks)
    assert chunks[0].split()[-3:] == chunks[1].split()[:3]          # overlap
    assert {w for c in chunks for w in c.split()} == set(words)       # nothing lost
    assert fixed_size("short text", 10, 3) == ["short text"]
    with pytest.raises(ValueError):
        fixed_size("a b c", size=5, overlap=5)


def test_sentence_chunking_never_splits_a_sentence():
    text = "One two three four. Five six seven eight nine. Ten eleven. Twelve thirteen fourteen fifteen sixteen seventeen."
    chunks = sentence_boundary(text, max_words=9)
    assert chunks == ["One two three four. Five six seven eight nine.",
                      "Ten eleven. Twelve thirteen fourteen fifteen sixteen seventeen."]
    assert all(len(c.split()) <= 9 for c in chunks)
    assert all(c.endswith(".") for c in chunks)


def test_chunks_keep_their_parent_passage(corpus):
    chunks = chunk_corpus(corpus.passages, "fixed", 20, 5)
    assert len(chunks) > len(corpus.passages)
    assert all(c.passage_id in corpus.passages and c.chunk_id.startswith(c.passage_id) for c in chunks)
    with pytest.raises(ValueError):
        chunk_corpus(corpus.passages, "magic")


def test_bm25_finds_the_obvious_passage(corpus):
    chunks = chunk_corpus(corpus.passages, "sentence", 40)
    hits = BM25Retriever(chunks).search("who wrote pride and prejudice", 5)
    assert to_passages(hits, chunks, 5)[0] == "p13"
    assert BM25Retriever(chunks).search("zzzz qqqq", 5) == []


@pytest.mark.parametrize("kind", ["dense", "bm25", "hybrid"])
def test_all_retrievers_return_ranked_unique_passages(corpus, settings, kind):
    chunks = chunk_corpus(corpus.passages, "fixed", 40, 10)
    retriever = build_retriever(kind, chunks, get_embedder(settings))
    hits = retriever.search("what is the largest planet in the solar system", 10)
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)
    ranked = to_passages(hits, chunks, 5)
    assert len(ranked) == len(set(ranked)) and "p28" in ranked


def test_metrics():
    assert recall_at_k(["a", "b", "c"], ["c", "z"], 3) == 0.5
    assert recall_at_k(["a", "b", "c"], ["c"], 2) == 0.0
    assert reciprocal_rank(["a", "b", "c"], ["b"]) == 0.5
    assert token_f1("Jupiter is the largest planet [1]", "Jupiter is the largest planet") == 1.0
    assert token_f1("banana", "Jupiter") == 0.0
    assert lexical_faithfulness("Jupiter is the largest planet", ["Jupiter is the largest planet in the solar system"]) == 1.0
    assert lexical_faithfulness("Saturn has 900 moons", ["Jupiter is the largest planet"]) == 0.0


def test_retrieval_grid_reports_every_combination(corpus, settings):
    rows, failures = retrieval_grid(corpus, settings, get_embedder(settings))
    assert {(r["chunker"], r["retriever"]) for r in rows} == {(c, r) for c in ("fixed", "sentence") for r in ("dense", "bm25", "hybrid")}
    bm25 = next(r for r in rows if r["retriever"] == "bm25" and r["chunker"] == "sentence")
    assert bm25["recall@5"] >= 0.9                                   # regression floor on the fixed sample
    assert all(0 <= r["recall@1"] <= r["recall@3"] <= r["recall@5"] <= 1 for r in rows)
    assert any(f["query_id"] == "q19" for f in failures)             # the known vocabulary-mismatch case


class FakeRetriever:
    """Returns the off-topic chunk unless the query contains the magic word 'notebook'."""

    def __init__(self):
        self.chunks = [Chunk("x#0", "x", "Bananas are yellow fruit that grow in bunches."),
                       Chunk("y#0", "y", "To stop a laptop overheating, clean the vents and fan regularly.")]
        self.queries = []

    def search(self, query, k):
        self.queries.append(query)
        return [(1, 0.9)] if "notebook" in query else [(0, 0.1)]


def test_insufficient_context_triggers_rewrite_then_grounded_answer(settings):
    retriever = FakeRetriever()
    chat = ScriptedChat("notebook cooling vents", "Clean the vents and fan regularly. [1]")
    log = EventLog("test")
    result = RAGPipeline(retriever, chat, settings, log).answer("how do i stop my laptop overheating")
    assert retriever.queries == ["how do i stop my laptop overheating", "notebook cooling vents"]
    assert not result.abstained and result.contexts[0].chunk.passage_id == "y"
    assert "Bananas" not in chat.calls[1][1]["content"]                 # the irrelevant chunk never reaches the generator
    assert [e["kind"] for e in log.events] == ["retrieved", "query_rewritten", "retrieved", "answered"]
    assert result.retrieved_passage_ids == ["x"]                        # retrieval metrics use the FIRST ranking


def test_abstains_without_calling_the_generator_when_rewrite_does_not_help(settings):
    retriever = FakeRetriever()
    chat = ScriptedChat("laptop heat problem")                          # only the rewrite call
    result = RAGPipeline(retriever, chat, settings).answer("how do i stop my laptop overheating")
    assert result.abstained and result.answer == ABSTAIN and result.contexts == []
    assert len(retriever.queries) == settings.rag_max_rewrites + 1      # bounded retries
    assert len(chat.calls) == 1


def test_rewrite_that_repeats_the_query_stops_the_loop(settings):
    settings.rag_max_rewrites = 3
    retriever = FakeRetriever()
    question = "how do i stop my laptop overheating"
    result = RAGPipeline(retriever, ScriptedChat(question), settings).answer(question)
    assert result.abstained and retriever.queries == [question]


def test_llm_grader_uses_one_call_for_all_chunks_and_falls_back(settings):
    settings.rag_grader = "llm"
    chunks = [Chunk("a#0", "a", "Bananas are yellow."), Chunk("b#0", "b", "Apples are red or green.")]

    class Both:
        def __init__(self): self.chunks = chunks
        def search(self, query, k): return [(0, 0.9), (1, 0.8)]

    chat = ScriptedChat('{"relevant": [2]}', "Apples are red or green. [1]")
    result = RAGPipeline(Both(), chat, settings).answer("what colour are apples")
    assert len(chat.calls) == 2 and [c.chunk.passage_id for c in result.contexts] == ["b"]
    assert "[1] Bananas are yellow.\n[2] Apples" in chat.calls[0][1]["content"]

    log = EventLog("test")
    chat = ScriptedChat("I think the second one?", "Apples are red or green. [1]")     # unparseable grade
    result = RAGPipeline(Both(), chat, settings, log).answer("what colour are apples")
    assert [c.chunk.passage_id for c in result.contexts] == ["b"]                        # threshold fallback
    assert log.events[0]["kind"] == "grade_parse_failed"


def test_model_abstention_is_detected(settings):
    retriever = FakeRetriever()
    retriever.search = lambda query, k: [(1, 0.9)]
    result = RAGPipeline(retriever, ScriptedChat(ABSTAIN), settings).answer("how do i stop my laptop overheating")
    assert result.abstained and result.contexts


def test_ms_marco_rows_are_cleaned_deduplicated_and_labelled():
    rows = [{"query_id": 7, "query": " what is  rag ", "answers": ["No Answer Present.", "Retrieval augmented generation."],
             "passages": [{"text": "RAG combines retrieval &amp; generation in one system.", "is_selected": 1},
                          {"text": "RAG combines retrieval & generation in one   system.", "is_selected": 0},   # duplicate
                          {"text": "too short", "is_selected": 0},
                          {"text": "A rag is a piece of old cloth used for cleaning.", "is_selected": 0}]},
            {"query_id": 8, "query": "unlabelled", "answers": [], "passages": [{"text": "one two three four five six", "is_selected": 0}]}]
    corpus = _from_rows(rows, "ms_marco_v1.1")
    assert len(corpus.passages) == 3 and len(corpus.queries) == 1          # query without a label is not evaluable
    q = corpus.queries[0]
    assert q.text == "what is rag" and q.answers == ["Retrieval augmented generation."]
    assert corpus.passages[q.relevant_ids[0]] == "RAG combines retrieval & generation in one system."


def test_rag_api_default_corpus_upload_then_question_answered_from_the_upload(settings):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from rag.api import create_router

    app = FastAPI()
    chat = ScriptedChat("Zorblax is the capital of Mars [1].", "submarine sonar navigation")
    app.include_router(create_router(settings, chat, get_embedder(settings)))
    client = TestClient(app)
    assert client.get("/rag").status_code == 200
    default = client.get("/api/rag/corpus").json()
    assert default["documents"] == 57 and len(default["examples"]) == 3            # configured corpus, built on first use

    docs = [{"name": "mars.txt", "text": "The capital of Mars is Zorblax, a city built in 2090."},
            {"name": "venus.txt", "text": "Venus has no capital city because nobody lives there yet."}]
    indexed = client.post("/api/rag/documents", json={"documents": docs}).json()
    assert (indexed["source"], indexed["documents"], indexed["events"][0]["kind"]) == ("uploaded", 2, "indexed") and indexed["examples"] == []

    reply = client.post("/api/rag/ask", json={"question": "what is the capital of Mars"}).json()
    assert reply["answer"].startswith("Zorblax") and reply["contexts"][0]["document"] == "mars.txt"
    assert [e["kind"] for e in reply["events"]] == ["question", "retrieved", "answered"]
    assert "capital of Mars is Zorblax" in reply["events"][-1]["sent"]              # the prompt is in the log

    nothing = client.post("/api/rag/ask", json={"question": "how do submarines navigate"}).json()
    assert nothing["abstained"] and nothing["contexts"] == []                      # rewrite did not help: no generator call
    assert client.post("/api/rag/documents", json={"documents": [{"name": "x", "text": "   "}]}).status_code == 422
    assert client.post("/api/rag/documents", json={"documents": []}).json()["documents"] == 57   # back to the default
