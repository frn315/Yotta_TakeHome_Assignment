import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from builder.schema import Site, default_props          # noqa: E402
from common.config import Endpoint, Settings            # noqa: E402


@pytest.fixture()
def settings(tmp_path):
    """Offline settings: tests never read .env and never touch the network."""
    return Settings(llm=Endpoint("llama", "http://llm.test/v1", "test-model"),
                    embedding=Endpoint("hash", "", "hash-1024"),
                    llm_max_retries=0, rag_force_local=True, log_dir=tmp_path / "logs")


@pytest.fixture()
def site():
    sections = [{"id": t, "type": t, "props": default_props(t, "Fitness Studio")}
                for t in ("hero", "pricing", "testimonials", "contact")]
    sections[0]["style"] = {"background": "#1e3a8a", "text_color": "#ffffff"}
    return Site.model_validate({"title": "Fitness Studio", "sections": sections}).model_dump()


class ScriptedChat:
    """Chat double that returns queued replies and records what it was asked. The workflow code
    under test (parsing, validation, repair, patching) is the real thing; only the model is scripted."""

    provider, model = "scripted", "scripted"

    def __init__(self, *replies: str):
        self.replies, self.calls = list(replies), []

    def complete(self, messages, *, json_mode=False):
        self.calls.append(messages)
        if not self.replies:
            raise AssertionError("ScriptedChat ran out of replies")
        return self.replies.pop(0)
