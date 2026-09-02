"""普通文本、HTML、JSON 和结构化富文本的确定性 Markdown 渲染。"""
from __future__ import annotations

import html
import json
import re
from html.parser import HTMLParser
from typing import Any, Dict, Iterable, List, Sequence

from .models import ProcessingWarning

# 通用标签检测，确保 script/style/section/a 及未知 HTML 也经过统一清理器。
_HTML_RE = re.compile(r"<\s*/?\s*[A-Za-z][^>]*>")
_SPACE_RE = re.compile(r"[ \t\f\v]+")
_BLANK_RE = re.compile(r"\n{3,}")
_INDENTED_LIST_RE = re.compile(r"^(?P<indent>[ \t]+)(?P<body>(?:[-+*]|\d+\.)\s+.*)$")
_TEXT_KEYS = ("text", "content", "value")
_CHILD_KEYS = ("children", "nodes", "data", "blocks", "paragraph", "tables", "items")
_IGNORED_KEYS = {
    "src", "url", "href", "style", "styles", "id", "key", "uuid", "type", "nodeType",
    "node_type", "attributes", "attrs", "metadata", "meta", "width", "height", "class",
    "className", "target", "rel",
}
_SUPPORTED_CONTENT_TYPES = (str, dict, list, tuple, int, float, bool, type(None))


def _clean(text: str) -> str:
    lines = []
    for raw_line in text.replace("\r", "").split("\n"):
        match = _INDENTED_LIST_RE.match(raw_line)
        if match:
            # Markdown 依赖列表项的前导空格表达嵌套层级。
            indent = match.group("indent").replace("\t", "  ")
            body = _SPACE_RE.sub(" ", match.group("body")).strip()
            lines.append(indent + body)
        else:
            lines.append(_SPACE_RE.sub(" ", raw_line).strip())
    return _BLANK_RE.sub("\n\n", "\n".join(lines)).strip()


class _HTMLToMarkdown(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []
        self.list_stack: List[str] = []
        self.list_indices: List[int] = []
        self.in_ignored = 0
        self.table: List[List[str]] | None = None
        self.row: List[str] | None = None
        self.cell: List[str] | None = None

    def _append(self, text: str) -> None:
        """所有正文和内联标记都写入当前活动缓冲区。"""
        if self.cell is not None:
            self.cell.append(text)
        else:
            self.parts.append(text)

    def handle_starttag(self, tag: str, attrs: List[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attrs_dict = dict(attrs)
        if tag in {"script", "style", "noscript"}:
            self.in_ignored += 1
            return
        if self.in_ignored:
            return
        if tag == "br":
            self._append("\n")
        elif tag in {"p", "div", "section", "article"}:
            self._append("\n\n")
        elif tag.startswith("h") and len(tag) == 2 and tag[1].isdigit():
            self._append("\n\n" + "#" * int(tag[1]) + " ")
        elif tag in {"ul", "ol"}:
            self.list_stack.append(tag)
            self.list_indices.append(0)
            self._append("\n")
        elif tag == "li":
            if self.list_stack and self.list_stack[-1] == "ol":
                self.list_indices[-1] += 1
                marker = f"{self.list_indices[-1]}. "
            else:
                marker = "- "
            self._append("\n" + "  " * max(0, len(self.list_stack) - 1) + marker)
        elif tag == "img":
            description = next(
                (attrs_dict.get(k) for k in ("alt", "title", "caption", "description") if attrs_dict.get(k)),
                None,
            )
            if description:
                self._append(f"[图片：{description}]")
        elif tag == "table":
            self.table = []
        elif tag == "tr" and self.table is not None:
            self.row = []
        elif tag in {"td", "th"} and self.row is not None:
            self.cell = []
        elif tag in {"strong", "b"}:
            self._append("**")
        elif tag in {"em", "i"}:
            self._append("*")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript"}:
            self.in_ignored = max(0, self.in_ignored - 1)
            return
        if self.in_ignored:
            return
        if tag in {"ul", "ol"} and self.list_stack:
            self.list_stack.pop()
            self.list_indices.pop()
            self._append("\n")
        elif tag in {"strong", "b"}:
            self._append("**")
        elif tag in {"em", "i"}:
            self._append("*")
        elif tag in {"td", "th"} and self.cell is not None and self.row is not None:
            self.row.append(_clean("".join(self.cell)))
            self.cell = None
        elif tag == "tr" and self.row is not None and self.table is not None:
            if self.row:
                self.table.append(self.row)
            self.row = None
        elif tag == "table" and self.table is not None:
            self.parts.append("\n\n" + _markdown_table(self.table) + "\n\n")
            self.table = None

    def handle_data(self, data: str) -> None:
        if self.in_ignored:
            return
        if self.cell is not None:
            self.cell.append(data)
        else:
            self.parts.append(data)

    def result(self) -> str:
        return _clean("".join(self.parts))


def _markdown_table(rows: Sequence[Sequence[Any]]) -> str:
    normalized = [[_clean(str(cell)).replace("\n", "<br>").replace("|", "\\|") for cell in row] for row in rows]
    normalized = [row for row in normalized if row]
    if not normalized:
        return ""
    width = max(len(row) for row in normalized)
    normalized = [row + [""] * (width - len(row)) for row in normalized]
    header = normalized[0]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in normalized[1:])
    return "\n".join(lines)


def _plain_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _table_rows(value: Any, warnings: List[ProcessingWarning], path: str) -> List[List[str]]:
    if isinstance(value, dict):
        value = value.get("rows") or value.get("data") or value.get("children") or []
    if not isinstance(value, (list, tuple)):
        return []
    rows: List[List[str]] = []
    for row_index, row in enumerate(value):
        if isinstance(row, dict):
            row = row.get("cells") or row.get("columns") or row.get("children") or row.get("data") or []
        if not isinstance(row, (list, tuple)):
            row = [row]
        rows.append([
            _render(cell, warnings, f"{path}.rows[{row_index}]") for cell in row
        ])
    return rows


def _render_mapping(node: Dict[str, Any], warnings: List[ProcessingWarning], path: str) -> str:
    node_type = str(node.get("type") or node.get("nodeType") or node.get("node_type") or "").lower()
    if node_type in {"table", "tables"} or "rows" in node:
        return _markdown_table(_table_rows(node, warnings, path))
    if node_type in {"img", "image", "picture"}:
        desc = next((node.get(k) for k in ("alt", "title", "caption", "description") if node.get(k)), None)
        return f"[图片：{_clean(str(desc))}]" if desc else ""
    if node_type in {"ul", "ol"}:
        children = next((node.get(k) for k in _CHILD_KEYS if isinstance(node.get(k), (list, tuple))), [])
        lines = []
        for i, child in enumerate(children, 1):
            text = _render(child, warnings, f"{path}.{node_type}[{i - 1}]")
            if text:
                marker = f"{i}." if node_type == "ol" else "-"
                item_lines = text.splitlines()
                rendered_lines = [f"{marker} {item_lines[0]}"]
                rendered_lines.extend(
                    "" if not line else f"  {line}" for line in item_lines[1:]
                )
                lines.append("\n".join(rendered_lines))
        return "\n".join(lines)
    if node_type == "li":
        return _render_first_content(node, warnings, path)
    if node_type in {"paragraph", "p", "block", "blocks", "document", "root", "text"}:
        return _render_first_content(node, warnings, path)

    pieces: List[str] = []
    recognized = False
    for key in _TEXT_KEYS + _CHILD_KEYS:
        if key in node:
            recognized = True
            text = _render(node[key], warnings, f"{path}.{key}")
            if text and text not in pieces:
                pieces.append(text)
    if not recognized:
        for key, value in node.items():
            if key in _IGNORED_KEYS or key in {"alt", "title", "caption", "description"}:
                continue
            if isinstance(value, (dict, list, tuple)):
                text = _render(value, warnings, f"{path}.{key}")
                if text:
                    pieces.append(text)
    if node_type and node_type not in {
        "table", "tables", "img", "image", "picture", "ul", "ol", "li", "paragraph",
        "p", "block", "blocks", "document", "root", "text",
    }:
        warnings.append(ProcessingWarning(
            code="unknown_richtext_node",
            message=f"未知富文本节点 {node_type}，已递归提取文本",
            field=path,
            details={"node_type": node_type},
        ))
    return "\n\n".join(pieces)


def _render_first_content(node: Dict[str, Any], warnings: List[ProcessingWarning], path: str) -> str:
    pieces = []
    for key in _TEXT_KEYS + _CHILD_KEYS:
        if key in node:
            text = _render(node[key], warnings, f"{path}.{key}")
            if text and text not in pieces:
                pieces.append(text)
    return "\n\n".join(pieces)


def _render(value: Any, warnings: List[ProcessingWarning], path: str) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ""
        if stripped[:1] in {"{", "["}:
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, (dict, list)):
                return _render(parsed, warnings, path)
        if _HTML_RE.search(stripped):
            parser = _HTMLToMarkdown()
            parser.feed(stripped)
            parser.close()
            return parser.result()
        return _clean(html.unescape(stripped))
    if isinstance(value, dict):
        return _render_mapping(value, warnings, path)
    if isinstance(value, (list, tuple)):
        return "\n\n".join(
            text for index, item in enumerate(value)
            if (text := _render(item, warnings, f"{path}[{index}]"))
        )
    if isinstance(value, (int, float, bool)):
        return _plain_value(value)
    warnings.append(ProcessingWarning(
        code="unsupported_richtext_type",
        message=f"不支持的富文本类型 {type(value).__name__}，已转为文本",
        field=path,
    ))
    return _clean(str(value))


def render_richtext(
    value: Any,
    warnings: List[ProcessingWarning] | None = None,
    path: str = "content",
) -> str:
    target = warnings if warnings is not None else []
    return _clean(_render(value, target, path))


def is_supported_content_type(value: Any) -> bool:
    """返回富文本渲染器原生支持、无需兜底转字符串的正文类型。"""
    return isinstance(value, _SUPPORTED_CONTENT_TYPES)


def is_renderable_content(value: Any) -> bool:
    """正文类型受支持且实际渲染后存在可见内容。"""
    return is_supported_content_type(value) and bool(render_richtext(value))


def render_richtext_with_warnings(value: Any, path: str = "content") -> tuple[str, List[ProcessingWarning]]:
    warnings: List[ProcessingWarning] = []
    return render_richtext(value, warnings, path), warnings
