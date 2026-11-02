"""The course starter agent, independent of Harbor."""

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from openai.types.responses import FunctionToolParam

from agents590.contracts import RunContext
from agents590.escoats.atif import AtifRecorder
from agents590.provider import UNCOpenAISettings


class EscoatsAgent:
    """Scaffold for a model/tool loop using the shared standalone interface."""

    tools: list[FunctionToolParam] = [
        {
            "type": "function",
            "name": "run_shell_command",
            "description": "Run a shell command and return its output.",
            "strict": False,
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Command to be run",
                    },
                },
                "required": ["command"],
            },
        },
    ]

    def provider_settings(self, context: RunContext) -> UNCOpenAISettings:
        """Prefer the per-run model to the environment's configured deployment."""
        return UNCOpenAISettings.from_env(model=context.model)

    async def run(self, instruction: str, context: RunContext) -> str:
        settings = self.provider_settings(context)
        client: AsyncOpenAI = settings.create_async_client()
        atif_path = os.environ.get("AGENTS590_ATIF_PATH")
        recorder = (
            AtifRecorder(
                Path(atif_path), instruction=instruction, model=settings.model, tools=self.tools
            )
            if atif_path
            else None
        )

        input_list: list[Any] = [
            {
                "role": "user",
                "content": instruction,
            }
        ]

        while True:
            response = await client.responses.create(
                model=settings.model,
                tools=self.tools,
                input=input_list,
                store=False,
                background=False,
            )
            input_list += response.output

            tool_calls = [item for item in response.output if item.type == "function_call"]

            if not tool_calls:
                if recorder is not None:
                    recorder.record_response(response, [])
                break

            tool_results: list[tuple[str, str]] = []
            for item in tool_calls:
                name = item.name
                args = json.loads(item.arguments)
                result = self.call_function(name, args, context)
                tool_results.append((item.call_id, result))
                input_list.append(
                    {"type": "function_call_output", "call_id": item.call_id, "output": result}
                )
            if recorder is not None:
                recorder.record_response(response, tool_results)

        return response.output_text

    def call_function(self, name: str, args: dict[str, Any], context: RunContext) -> str:
        """Call the specified function and return its result."""
        match name:
            case "read_file":
                filename: str = args["filename"]
                return self.read_file(filename)
            case "write_file":
                filename: str = args["filename"]
                contents: str = args["contents"]
                return self.write_file(filename, contents)
            case "run_shell_command":
                command: str = args["command"]
                return self.run_shell_command(command, context)
            case _:
                return "no tool called"

    def read_file(self, filename: str) -> str:
        """Read a file and return its contents."""
        with open(filename) as f:
            contents = f.read()
        return contents

    def write_file(self, filename: str, contents: str) -> str:
        """Write to a file and return its full contents afterwards."""
        with open(filename, "w") as f:
            f.write(contents)

        return json.dumps({"filename": filename, "contents": contents})

    def run_shell_command(self, command: str, context: RunContext) -> str:
        """Run a shell command and return its output."""
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=context.cwd,
            check=False,
        )

        return json.dumps(
            {"exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
        )
