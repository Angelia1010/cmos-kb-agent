"""标准候选集的只读质量分析。"""
from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Dict, Sequence

from .eligibility import has_renderable_candidate_content
from .models import KnowledgeCandidate, KnowledgeProcessingOptions
from .richtext import is_renderable_content, is_supported_content_type

_HTML_RE = re.compile(r"<\s*[a-zA-Z][^>]*>")
_TABLE_RE = re.compile(r"<\s*table\b|\|[^\n]+\|", re.I)


def _serialized(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError, RecursionError):
        return str(value)


def analyze_candidates(
    candidates: Sequence[KnowledgeCandidate],
    options: KnowledgeProcessingOptions | None = None,
) -> Dict[str, Any]:
    """不修改候选，只统计后续处理需要关注的特征。"""
    options = options or KnowledgeProcessingOptions()
    missing = Counter()
    invalid_types = Counter()
    atom_count = html_count = table_count = 0
    except_count = annotation_count = long_count = 0
    for candidate in candidates:
        if not candidate.knowledge_id:
            missing["knowledge_id"] += 1
        if not candidate.name:
            missing["name"] += 1
        if not has_renderable_candidate_content(candidate):
            missing["content"] += 1
        if not isinstance(candidate.atoms, list):
            invalid_types["atoms"] += 1
            atoms = []
        else:
            atoms = candidate.atoms
        texts = [_serialized(candidate.content)]
        atom_count += len(atoms)
        for atom in atoms:
            if not is_renderable_content(atom.content):
                missing["atom_content"] += 1
            if atom.except_rules not in (None, "", [], {}):
                except_count += 1
            if atom.annotation not in (None, "", [], {}):
                annotation_count += 1
            texts.append(_serialized(atom.content))
        joined = "\n".join(texts)
        if _HTML_RE.search(joined):
            html_count += 1
        if _TABLE_RE.search(joined) or any(
            isinstance(atom.content, dict)
            and str(atom.content.get("type", "")).lower() in {"table", "tables"}
            for atom in atoms
        ):
            table_count += 1
        if len(joined) > options.long_content_threshold:
            long_count += 1
        if not is_supported_content_type(candidate.content):
            invalid_types[type(candidate.content).__name__] += 1
    return {
        "candidate_count": len(candidates),
        "atom_count": atom_count,
        "missing_fields": dict(sorted(missing.items())),
        "html_candidate_count": html_count,
        "table_candidate_count": table_count,
        "except_rules_count": except_count,
        "annotation_count": annotation_count,
        "long_content_count": long_count,
        "invalid_types": dict(sorted(invalid_types.items())),
    }


# 更明确的别名，方便核心流水线和外部调用。
analyze_knowledge_candidates = analyze_candidates
