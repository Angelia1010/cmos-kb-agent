"""候选可渲染性和重排资格的单一判定源。"""
from __future__ import annotations

import re
from typing import Optional

from .models import KnowledgeCandidate, ProcessedKnowledge
from .richtext import is_renderable_content, render_richtext


def _without_own_title(text: str, name: str) -> str:
    lines = text.splitlines()
    if lines and re.sub(r"^#+\s*", "", lines[0]).strip() == str(name or "").strip():
        lines = lines[1:]
    return "\n".join(lines).strip()


def _markdown_has_body(content_md: str, name: str) -> bool:
    text = _without_own_title(str(content_md or "").strip(), name)
    return any(
        line.strip() and not line.lstrip().startswith("#")
        for line in text.splitlines()
    )


def has_renderable_candidate_content(candidate: KnowledgeCandidate) -> bool:
    """不把知识标题、分组名或字段名本身当作业务证据。"""
    if is_renderable_content(candidate.content):
        main_text = _without_own_title(render_richtext(candidate.content), candidate.name)
        if main_text:
            return True
    if any(is_renderable_content(atom.content) for atom in candidate.atoms):
        return True
    if isinstance(candidate, ProcessedKnowledge):
        return _markdown_has_body(candidate.content_md, candidate.name)
    return False


def rerank_ineligibility_reason(candidate: ProcessedKnowledge) -> Optional[str]:
    knowledge_id = candidate.knowledge_id
    if knowledge_id is None or not str(knowledge_id).strip():
        return "missing_knowledge_id"
    if not has_renderable_candidate_content(candidate):
        return "empty_rendered_content"
    return None


def is_rerank_eligible(candidate: ProcessedKnowledge) -> bool:
    return rerank_ineligibility_reason(candidate) is None
