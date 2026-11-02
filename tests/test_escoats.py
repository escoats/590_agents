"""Exercise escoats trajectory reporting without making model requests."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agents590.contracts import RunContext
from agents590.escoats import EscoatsAgent


def _response(
    output: list[Any], *, text: str, prompt_tokens: int, completion_tokens: int, cached_tokens: int
) -> SimpleNamespace:
    return SimpleNamespace(
        output=output,
        output_text=text,
        usage=SimpleNamespace(
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            input_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
        ),
    )


def test_run_writes_atif_trajectory_with_aggregate_usage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.txt"
    source.write_text("tool result")
    tool_call = SimpleNamespace(
        type="function_call",
        call_id="call-1",
        name="read_file",
        arguments=json.dumps({"filename": str(source)}),
    )
    responses = [
        _response([tool_call], text="", prompt_tokens=10, completion_tokens=2, cached_tokens=3),
        _response([], text="finished", prompt_tokens=20, completion_tokens=4, cached_tokens=5),
    ]

    class FakeResponses:
        async def create(self, **kwargs: object) -> SimpleNamespace:
            return responses.pop(0)

    settings = SimpleNamespace(
        model="test-deployment",
        create_async_client=lambda: SimpleNamespace(responses=FakeResponses()),
    )
    agent = EscoatsAgent()
    monkeypatch.setattr(agent, "provider_settings", lambda context: settings)
    trajectory_path = tmp_path / "logs" / "trajectory.json"
    monkeypatch.setenv("AGENTS590_ATIF_PATH", str(trajectory_path))

    output = asyncio.run(agent.run("read the source", RunContext(cwd=tmp_path)))

    assert output == "finished"
    trajectory = json.loads(trajectory_path.read_text())
    assert trajectory["schema_version"] == "ATIF-v1.8"
    assert trajectory["agent"]["name"] == "escoats"
    assert trajectory["agent"]["model_name"] == "test-deployment"
    assert [step["source"] for step in trajectory["steps"]] == ["user", "agent", "agent"]
    assert trajectory["steps"][1]["tool_calls"] == [
        {
            "tool_call_id": "call-1",
            "function_name": "read_file",
            "arguments": {"filename": str(source)},
        }
    ]
    assert trajectory["steps"][1]["observation"]["results"] == [
        {"source_call_id": "call-1", "content": "tool result"}
    ]
    assert trajectory["final_metrics"] == {
        "total_prompt_tokens": 30,
        "total_completion_tokens": 6,
        "total_cached_tokens": 8,
        "total_steps": 3,
    }
