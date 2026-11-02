"""Small, dependency-free ATIF recorder for the escoats agent."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from openai.types.responses import FunctionToolParam, Response


class AtifRecorder:
    """Collect Responses API turns and expose them to Harbor as ATIF JSON."""

    def __init__(
        self, path: Path, *, instruction: str, model: str, tools: list[FunctionToolParam]
    ) -> None:
        self.path = path
        self.model = model
        self.steps: list[dict[str, Any]] = [
            {
                "step_id": 1,
                "timestamp": self._timestamp(),
                "source": "user",
                "message": instruction,
            }
        ]
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cached_tokens = 0
        self.trajectory: dict[str, Any] = {
            "schema_version": "ATIF-v1.8",
            "session_id": str(uuid4()),
            "agent": {
                "name": "escoats",
                "version": "0.1.0",
                "model_name": model,
                "tool_definitions": tools,
            },
            "steps": self.steps,
        }

    def record_response(self, response: Response, tool_results: list[tuple[str, str]]) -> None:
        """Record one model inference and any tool observations it produced."""
        usage = response.usage
        prompt_tokens = usage.input_tokens if usage is not None else 0
        completion_tokens = usage.output_tokens if usage is not None else 0
        cached_tokens = usage.input_tokens_details.cached_tokens if usage is not None else 0
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.cached_tokens += cached_tokens

        tool_calls: list[dict[str, Any]] = []
        for item in response.output:
            if item.type == "function_call":
                tool_calls.append(
                    {
                        "tool_call_id": item.call_id,
                        "function_name": item.name,
                        "arguments": json.loads(item.arguments),
                    }
                )

        step: dict[str, Any] = {
            "step_id": len(self.steps) + 1,
            "timestamp": self._timestamp(),
            "source": "agent",
            "model_name": self.model,
            "message": response.output_text,
            "llm_call_count": 1,
            "metrics": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cached_tokens": cached_tokens,
            },
        }
        if tool_calls:
            step["tool_calls"] = tool_calls
            step["observation"] = {
                "results": [
                    {"source_call_id": call_id, "content": result}
                    for call_id, result in tool_results
                ]
            }
        self.steps.append(step)
        self.trajectory["final_metrics"] = {
            "total_prompt_tokens": self.prompt_tokens,
            "total_completion_tokens": self.completion_tokens,
            "total_cached_tokens": self.cached_tokens,
            "total_steps": len(self.steps),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.trajectory, indent=2) + "\n")

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(UTC).isoformat().replace("+00:00", "Z")
