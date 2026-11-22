#!/usr/bin/env python3
"""Phase 4 prompt goldens for Inkling 2, rendered by the checkpoint's own template.

The cases cover what the text path does not: tool DECLARATIONS, assistant TOOL
CALLS (escaped, nested, repeated, truncated), and named tool RESULTS. Each is
rendered by `transformers`' own `render_jinja_template` against the checkpoint's
`chat_template.jinja`, so the Swift renderer is compared with the template
rather than with a transcription of it -- the same rule the text goldens follow.

Requires the pinned reference environment (jinja2 + transformers). Writes
`/tmp/inkling2_tool_goldens.json` by default; pass a path as argv[1].
"""
import json
import sys

from transformers.utils import chat_template_utils as ctu

SNAPSHOT = (
    "/Users/derekparris/.cache/huggingface/hub/"
    "models--thinkingmachines--Inkling-2-Mini/snapshots/ed40e7da625f594fc9c4990f5f66be24873c22d7"
)
TEMPLATE = open(f"{SNAPSHOT}/chat_template.jinja").read()

# The template's own effort map, so a case can name the effort it renders.
EFFORTS = {"none": 0.0, "minimal": 0.1, "low": 0.2, "medium": 0.7, "high": 0.9,
           "xhigh": 0.99, "max": 0.99}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from disk.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "no_args_tool",
            # No description and no parameters: the template must still emit
            # every key with its default.
            "parameters": {},
        },
    },
]

CASES = [
    {
        "name": "declarationOnly",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": TOOLS[:1],
        "effort": "high",
    },
    {
        "name": "declarationDefaults",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": TOOLS,
        "effort": "high",
    },
    {
        "name": "callWithNestedArguments",
        "messages": [
            {"role": "user", "content": "read it"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "read_file",
                            # Nested object, array, boolean, number, and a string
                            # whose escapes and unicode must survive the round trip.
                            "arguments": {
                                "path": "a \"quoted\"/file.txt",
                                "options": {"limit": 3, "strict": True, "tags": ["x", "y"]},
                                "note": "line\nbreak \u00e9\u4e2d",
                            },
                        },
                    }
                ],
            },
        ],
        "tools": TOOLS[:1],
        "effort": "low",
    },
    {
        "name": "twoCallsInOneTurn",
        "messages": [
            {"role": "user", "content": "read both"},
            {
                "role": "assistant",
                "content": "I'll read both.",
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "read_file", "arguments": {"path": "a.txt"}}},
                    {"id": "call_2", "function": {"name": "read_file", "arguments": {"path": "b.txt"}}},
                ],
            },
        ],
        "tools": TOOLS[:1],
        "effort": "high",
    },
    {
        "name": "callThenNamedResult",
        "messages": [
            {"role": "user", "content": "read it"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "read_file", "arguments": {"path": "a.txt"}}}
                ],
            },
            # The template writes the tool's NAME straight after the role
            # marker, taken from `name` or resolved from `tool_call_id`.
            {"role": "tool", "tool_call_id": "call_1", "content": "file contents"},
            {"role": "user", "content": "summarize"},
        ],
        "tools": TOOLS[:1],
        "effort": "high",
    },
    {
        "name": "callThenResultByName",
        "messages": [
            {"role": "user", "content": "read it"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "read_file", "arguments": {}}}
                ],
            },
            {"role": "tool", "name": "read_file", "content": "file contents"},
        ],
        "tools": None,
        "effort": "high",
    },
    {
        "name": "reasoningThenCall",
        "messages": [
            {"role": "user", "content": "read it"},
            {
                "role": "assistant",
                "reasoning_content": "I should read the file.",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "read_file", "arguments": {"path": "a.txt"}}}
                ],
            },
        ],
        "tools": TOOLS[:1],
        "effort": "medium",
    },
    {
        "name": "reasoningAndTextAndCall",
        "messages": [
            {"role": "user", "content": "read it"},
            {
                "role": "assistant",
                "reasoning_content": "Thinking.",
                "content": "Reading now.",
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "read_file", "arguments": {"path": "a.txt"}}}
                ],
            },
        ],
        "tools": None,
        "effort": "high",
    },
]


def main():
    output = sys.argv[1] if len(sys.argv) > 1 else "/tmp/inkling2_tool_goldens.json"
    root = {}
    for case in CASES:
        rendered, _ = ctu.render_jinja_template(
            [case["messages"]],
            tools=case["tools"],
            chat_template=TEMPLATE,
            reasoning_effort=EFFORTS[case["effort"]],
        )
        root[case["name"]] = {
            "rendered": rendered[0] if isinstance(rendered, list) else rendered,
            "effort": case["effort"],
            "tools": case["tools"],
        }
    with open(output, "w") as handle:
        json.dump(root, handle, indent=1, ensure_ascii=False)
    print(f"wrote {len(root)} case(s) to {output}")


if __name__ == "__main__":
    main()
