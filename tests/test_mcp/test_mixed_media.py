# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from mcp.types import CallToolResult, ImageContent, TextContent

from nooa.mcp import MCPTool, MCPToolError
from nooa.runtime.media_capture import capture_media


def image(data="AA=="):
    return ImageContent(type="image", data=data, mimeType="image/png")


def text(value):
    return TextContent(type="text", text=value)


class Client:
    def __init__(self, result):
        self.result = result

    @asynccontextmanager
    async def connect_to_server(self):
        async def call_tool(name, arguments):
            assert name == "probe" and arguments == {"keep": 1}
            return self.result

        yield SimpleNamespace(call_tool=call_tool)


@pytest.mark.asyncio
async def test_all_text_and_repeated_images_in_order():
    tool = MCPTool(
        Client(
            CallToolResult(content=[text("first"), image(), text("last"), image("AQ=="), image()])
        ),
        "fixture",
    )
    with capture_media() as captured:
        result = await tool._call_tool("probe", {"keep": 1})
    assert result == "first\nlast"
    assert [b["image_url"]["url"] for b in captured.blocks] == [
        "data:image/png;base64,AA==",
        "data:image/png;base64,AQ==",
        "data:image/png;base64,AA==",
    ]


@pytest.mark.asyncio
async def test_structured_content_keeps_original_value_and_image():
    payload = {"ok": True}
    tool = MCPTool(
        Client(CallToolResult(content=[text('{"ok":true}'), image()], structuredContent=payload)),
        "fixture",
    )
    with capture_media() as captured:
        result = await tool._call_tool("probe", {"keep": 1})
    assert result == payload and len(captured.blocks) == 1


@pytest.mark.asyncio
async def test_error_remains_error_without_success_media():
    tool = MCPTool(
        Client(CallToolResult(content=[text("failed"), image()], isError=True)), "fixture"
    )
    with capture_media() as captured:
        with pytest.raises(MCPToolError, match="failed"):
            await tool._call_tool("probe", {"keep": 1})
    assert captured.blocks == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "parts, expected",
    [([text("unchanged")], "unchanged"), ([text("a"), text("b")], "a\nb"), ([image()], "")],
)
async def test_text_and_image_only_results(parts, expected):
    tool = MCPTool(Client(CallToolResult(content=parts)), "fixture")
    with capture_media() as captured:
        result = await tool._call_tool("probe", {"keep": 1})
    assert result == expected
    assert len(captured.blocks) == sum(isinstance(p, ImageContent) for p in parts)


@pytest.mark.asyncio
async def test_outside_codeact_preserves_structured_return_contract():
    parts = [text("first"), image(), text("last")]
    response = CallToolResult(content=parts, structuredContent={"ok": True})
    tool = MCPTool(Client(response), "fixture")
    assert await tool._call_tool("probe", {"keep": 1}) == {"ok": True}


@pytest.mark.asyncio
async def test_outside_codeact_preserves_text_return_contract():
    tool = MCPTool(
        Client(CallToolResult(content=[text("first"), image(), text("last")])), "fixture"
    )
    assert await tool._call_tool("probe", {"keep": 1}) == "first"


@pytest.mark.asyncio
@pytest.mark.parametrize("mime_type", ["image/png", "image/jpeg"])
async def test_actual_sdk_image_wire_mime_and_unchanged_outside_capture(mime_type):
    part = ImageContent.model_validate(
        {"type": "image", "data": "AA==", "mimeType": mime_type}
    )
    before = part.model_dump(mode="json", by_alias=True)
    tool = MCPTool(Client(CallToolResult(content=[part])), "fixture")
    assert await tool._call_tool("probe", {"keep": 1}) == [part]
    with capture_media() as captured:
        assert await tool._call_tool("probe", {"keep": 1}) == ""
    assert [b["image_url"]["url"] for b in captured.blocks] == [
        f"data:{mime_type};base64,AA=="
    ]
    assert part.model_dump(mode="json", by_alias=True) == before
