"""
src/reviewer.py
---------------
Stage 3 — LLM Code Review (Orchestration)
Sends code chunks to an LLM (Claude Sonnet or GPT-4o-mini) and parses
schema-valid JSON review comments with confidence scores.
"""

import json
import logging
import os
import time
from typing import Any

from src.parser import CodeNode

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema / data types
# ---------------------------------------------------------------------------

REVIEW_CATEGORIES = [
    "bug",
    "security",
    "performance",
    "style",
    "maintainability",
    "documentation",
    "error_handling",
    "type_safety",
]

SEVERITY_LEVELS = ["critical", "high", "medium", "low", "info"]

LOW_CONFIDENCE_THRESHOLD = 60  # Below this → show "Verify This" label


class ReviewComment:
    """
    A single structured review comment produced by the LLM.

    Attributes
    ----------
    file_path : str
    node_name : str
    category : str
    severity : str
    line_hint : str          e.g. "12" or "12-18" or "N/A"
    title : str              Short headline.
    body : str               Detailed explanation.
    suggestion : str         Concrete code / wording fix.
    confidence : int         0-100
    low_confidence : bool    True when confidence < LOW_CONFIDENCE_THRESHOLD
    """

    __slots__ = (
        "file_path", "node_name", "category", "severity",
        "line_hint", "title", "body", "suggestion", "confidence", "low_confidence",
    )

    def __init__(self, **kwargs: Any) -> None:
        for slot in self.__slots__:
            setattr(self, slot, kwargs.get(slot, ""))
        self.confidence = int(self.confidence)
        self.low_confidence = self.confidence < LOW_CONFIDENCE_THRESHOLD

    def to_dict(self) -> dict:
        return {s: getattr(self, s) for s in self.__slots__}


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """
You are an expert code reviewer with deep knowledge of software engineering best practices,
security vulnerabilities, performance anti-patterns, and clean code principles.

Your task: review the provided code snippet and return a JSON array of review comments.

RULES:
1. Return ONLY a JSON array — no prose, no markdown fences, no extra keys.
2. Each element must conform exactly to this schema:
{
  "category":   "<one of: bug|security|performance|style|maintainability|documentation|error_handling|type_safety>",
  "severity":   "<one of: critical|high|medium|low|info>",
  "line_hint":  "<line number(s) as a string, e.g. '12' or '12-18', or 'N/A'>",
  "title":      "<short one-line headline>",
  "body":       "<detailed explanation, 2-5 sentences>",
  "suggestion": "<concrete code change or wording fix>",
  "confidence": <integer 0-100>
}
3. confidence reflects HOW CERTAIN you are about this comment.
   - 90-100: You are sure. Clear bug / definite anti-pattern.
   - 70-89:  Likely issue; depends on context you can't see.
   - 50-69:  Possible issue; needs human verification.
   - 0-49:   Speculative; flag it as uncertain.
4. If the code looks correct and has no issues, return an empty array: []
5. Do NOT invent issues. Prefer precision over quantity.
6. Limit output to at most 8 comments per snippet.
""".strip()


def _build_user_message(node: CodeNode) -> str:
    meta_lines = [
        f"File: {node.file_path}",
        f"Scope: {node.kind} '{node.name}'  (lines {node.start_line}–{node.end_line})",
        f"Language: {node.language}",
    ]
    if node.extra.get("docstring"):
        meta_lines.append(f"Docstring: {node.extra['docstring'][:200]}")
    if node.extra.get("decorators"):
        meta_lines.append(f"Decorators: {', '.join(node.extra['decorators'])}")
    if node.extra.get("base_classes"):
        meta_lines.append(f"Base classes: {', '.join(node.extra['base_classes'])}")

    meta_block = "\n".join(meta_lines)
    return f"<metadata>\n{meta_block}\n</metadata>\n\n<code>\n{node.source}\n</code>"


# ---------------------------------------------------------------------------
# Client adapters
# ---------------------------------------------------------------------------

def _call_openai(client: Any, model: str, user_msg: str) -> str:
    """Call OpenAI-compatible API; return raw text."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
        max_tokens=2048,
        response_format={"type": "json_object"},  # only works on gpt-4o-mini / gpt-4o
    )
    return response.choices[0].message.content or "[]"


def _call_anthropic(client: Any, model: str, user_msg: str) -> str:
    """Call Anthropic API; return raw text."""
    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
        temperature=0.2,
    )
    return response.content[0].text if response.content else "[]"


# ---------------------------------------------------------------------------
# Main Agent
# ---------------------------------------------------------------------------

class CodeReviewAgent:
    """
    Orchestrates LLM calls for each CodeNode and returns ReviewComment objects.

    Parameters
    ----------
    provider : str
        "openai" or "anthropic"
    model : str
        Model string, e.g. "gpt-4o-mini" or "claude-sonnet-4-20250514"
    api_key : str | None
        If None, reads from environment variable (OPENAI_API_KEY / ANTHROPIC_API_KEY).
    max_retries : int
        Number of retries on transient errors.
    retry_delay : float
        Seconds between retries.
    """

    def __init__(
        self,
        provider: str = "anthropic",
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> None:
        self.provider = provider.lower()
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.ai_client = self._build_client(api_key)

    def _build_client(self, api_key: str | None) -> Any:
        if self.provider == "openai":
            try:
                import openai
            except ImportError as exc:
                raise ImportError("openai package not installed") from exc
            key = api_key or os.environ.get("OPENAI_API_KEY", "")
            if not key:
                raise ValueError("OPENAI_API_KEY not set")
            return openai.OpenAI(api_key=key)

        elif self.provider == "anthropic":
            try:
                import anthropic
            except ImportError as exc:
                raise ImportError("anthropic package not installed") from exc
            key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
            if not key:
                raise ValueError("ANTHROPIC_API_KEY not set")
            return anthropic.Anthropic(api_key=key)

        else:
            raise ValueError(f"Unknown provider '{self.provider}'. Use 'openai' or 'anthropic'.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def review_node(self, node: CodeNode) -> list[ReviewComment]:
        """
        Send *node* to the LLM and return parsed ReviewComment objects.

        Parameters
        ----------
        node : CodeNode
            The parsed code chunk to review.

        Returns
        -------
        list[ReviewComment]
            Parsed, validated review comments.  May be empty if no issues found.
        """
        user_msg = _build_user_message(node)

        for attempt in range(1, self.max_retries + 1):
            try:
                raw = self._raw_llm_call(user_msg)
                comments = self.verify_json(raw, node)
                return comments
            except json.JSONDecodeError as exc:
                logger.warning(
                    "JSON parse error on attempt %d/%d for %s.%s: %s",
                    attempt, self.max_retries, node.file_path, node.name, exc,
                )
                if attempt == self.max_retries:
                    logger.error("Giving up on %s.%s after %d retries", node.file_path, node.name, self.max_retries)
                    return []
                time.sleep(self.retry_delay)
            except Exception as exc:
                logger.warning("LLM call error attempt %d/%d: %s", attempt, self.max_retries, exc)
                if attempt == self.max_retries:
                    return []
                time.sleep(self.retry_delay * attempt)  # exponential-ish back-off

        return []

    def verify_json(self, raw: str, node: CodeNode) -> list[ReviewComment]:
        """
        Parse and validate the LLM's raw JSON output.

        Strips markdown fences if the model added them, coerces field types,
        and annotates each comment with the originating file / node name.

        Parameters
        ----------
        raw : str
            Raw string response from the LLM.
        node : CodeNode
            Source node (used to fill file_path / node_name).

        Returns
        -------
        list[ReviewComment]
        """
        # Strip markdown fences if the model ignored our instruction
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        # The model might return {"comments": [...]} instead of [...]
        data = json.loads(text)
        if isinstance(data, dict):
            data = data.get("comments") or data.get("reviews") or []

        if not isinstance(data, list):
            raise json.JSONDecodeError("Expected a JSON array", text, 0)

        comments: list[ReviewComment] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            # Coerce / fill defaults
            category = item.get("category", "maintainability")
            if category not in REVIEW_CATEGORIES:
                category = "maintainability"
            severity = item.get("severity", "low")
            if severity not in SEVERITY_LEVELS:
                severity = "low"
            try:
                confidence = max(0, min(100, int(item.get("confidence", 50))))
            except (TypeError, ValueError):
                confidence = 50

            comments.append(ReviewComment(
                file_path=node.file_path,
                node_name=node.name,
                category=category,
                severity=severity,
                line_hint=str(item.get("line_hint", "N/A")),
                title=str(item.get("title", "Review comment")),
                body=str(item.get("body", "")),
                suggestion=str(item.get("suggestion", "")),
                confidence=confidence,
            ))

        return comments

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _raw_llm_call(self, user_msg: str) -> str:
        if self.provider == "openai":
            return _call_openai(self.ai_client, self.model, user_msg)
        return _call_anthropic(self.ai_client, self.model, user_msg)