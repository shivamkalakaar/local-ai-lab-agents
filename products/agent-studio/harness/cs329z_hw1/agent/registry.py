"""Problem 5 (tool_registry): hold tools, validate calls, run them.

validate() and execute() are separate so a bad call can be rejected before
anyone is asked to approve it (Problem 13). Neither ever raises.
"""

from __future__ import annotations

import json

from jsonschema import Draft202012Validator

from cs329z_hw1.types import ToolResult, ToolSpec


def _describe_errors(errors) -> str:
    lines = []
    for err in errors[:5]:
        where = ".".join(str(p) for p in err.absolute_path) or "(arguments)"
        lines.append(f"  - {where}: {err.message}")
    return "\n".join(lines)


class ToolRegistry:
    def __init__(self, tools: list[ToolSpec] | None = None):
        self._tools: dict[str, ToolSpec] = {}
        self._validators: dict[str, Draft202012Validator] = {}
        for spec in tools or []:
            self.register(spec)

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec  # dicts keep insertion order
        self._validators[spec.name] = Draft202012Validator(spec.parameters or {"type": "object"})

    def names(self) -> list[str]:
        return list(self._tools)

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def schemas(self) -> list[dict]:
        return [
            {"name": s.name, "description": s.description, "parameters": s.parameters}
            for s in self._tools.values()
        ]

    def validate(self, name: str, args: dict) -> ToolResult | None:
        try:
            if name not in self._tools:
                known = ", ".join(self._tools) or "(none)"
                return ToolResult("unknown_tool", f"There is no tool named {name!r}. Available tools: {known}.")
            if not isinstance(args, dict):
                return ToolResult("invalid_args", f"Arguments for {name!r} must be a JSON object, got {type(args).__name__}.")
            errors = sorted(self._validators[name].iter_errors(args), key=lambda e: list(e.absolute_path))
            if errors:
                schema = self._tools[name].parameters or {}
                allowed = ", ".join((schema.get("properties") or {}).keys()) or "(none)"
                required = ", ".join(schema.get("required") or []) or "(none)"
                return ToolResult(
                    "invalid_args",
                    f"Invalid arguments for {name!r}:\n{_describe_errors(errors)}\n"
                    f"Allowed arguments: {allowed}. Required: {required}.\n"
                    f"You sent: {json.dumps(args, ensure_ascii=False, default=str)[:500]}",
                )
            return None
        except Exception as exc:  # a broken schema must not crash the agent
            return ToolResult("invalid_args", f"Could not validate arguments for {name!r}: {exc}")

    def execute(self, name: str, args: dict) -> ToolResult:
        rejected = self.validate(name, args)
        if rejected is not None:
            return rejected
        try:
            value = self._tools[name].fn(**args)
        except Exception as exc:
            return ToolResult("tool_error", f"Tool {name!r} failed: {type(exc).__name__}: {exc}")
        try:
            return ToolResult("ok", str(value))
        except Exception as exc:
            return ToolResult("tool_error", f"Tool {name!r} returned a value that cannot be shown: {exc}")
