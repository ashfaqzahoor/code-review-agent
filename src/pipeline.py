"""
src/pipeline.py
---------------
Orchestrates Stages 1 → 2 → 3 end-to-end.
Coordinates Ingestor → ASTStaticAnalyzer → CodeReviewAgent
and surfaces a clean run() generator that yields progress events.
"""

import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

from src.ingestor import RepositoryIngestor
from src.parser import ASTStaticAnalyzer, CodeNode
from src.reviewer import CodeReviewAgent, ReviewComment

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """
    All configuration required to run a full review pipeline.
    """
    github_url: str
    provider: str = "anthropic"            # "openai" | "anthropic"
    model: str = "claude-sonnet-4-20250514"
    api_key: str = ""
    max_files: int = 30                    # Cap to avoid runaway API costs
    max_nodes_per_file: int = 20
    workspace: str = ""                    # Auto-created if empty


@dataclass
class PipelineResult:
    repo_name: str
    repo_root: str
    metadata: dict
    all_nodes: list[CodeNode] = field(default_factory=list)
    all_comments: list[ReviewComment] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    # Derived convenience properties
    @property
    def high_confidence(self) -> list[ReviewComment]:
        return [c for c in self.all_comments if not c.low_confidence]

    @property
    def low_confidence(self) -> list[ReviewComment]:
        return [c for c in self.all_comments if c.low_confidence]

    @property
    def by_severity(self) -> dict[str, list[ReviewComment]]:
        buckets: dict[str, list[ReviewComment]] = {
            "critical": [], "high": [], "medium": [], "low": [], "info": [],
        }
        for c in self.all_comments:
            buckets.setdefault(c.severity, []).append(c)
        return buckets

    @property
    def by_category(self) -> dict[str, list[ReviewComment]]:
        buckets: dict[str, list[ReviewComment]] = {}
        for c in self.all_comments:
            buckets.setdefault(c.category, []).append(c)
        return buckets


# ---------------------------------------------------------------------------
# Progress events (yielded by the generator)
# ---------------------------------------------------------------------------

@dataclass
class ProgressEvent:
    stage: str        # "clone" | "parse" | "review" | "done" | "error"
    message: str
    pct: float = 0.0  # 0.0 – 1.0
    detail: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

class ReviewPipeline:
    """
    End-to-end pipeline: clone → parse → review.

    Usage
    -----
    pipeline = ReviewPipeline(config)
    for event in pipeline.run():
        print(event.stage, event.message, f"{event.pct:.0%}")
    result = pipeline.result   # Available after generator is exhausted
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.result: PipelineResult | None = None
        self._workspace = config.workspace or tempfile.mkdtemp(prefix="review_")

    def run(self) -> Generator[ProgressEvent, None, None]:
        """Yield ProgressEvent objects as pipeline stages complete."""
        cfg = self.config

        # ── Stage 1: Clone ────────────────────────────────────────────
        yield ProgressEvent("clone", f"Cloning {cfg.github_url} …", pct=0.02)
        ingestor = RepositoryIngestor(target_workspace=self._workspace)
        try:
            repo_root = ingestor.clone_repo(cfg.github_url)
        except Exception as exc:
            yield ProgressEvent("error", f"Clone failed: {exc}", pct=0.0)
            return

        metadata = ingestor.get_repo_metadata(repo_root)
        yield ProgressEvent(
            "clone", "Repository cloned successfully.", pct=0.10,
            detail={"metadata": metadata},
        )

        # ── Stage 2: Parse ────────────────────────────────────────────
        yield ProgressEvent("parse", "Scanning and parsing source files …", pct=0.12)
        analyzer = ASTStaticAnalyzer()

        source_files = list(ingestor.iter_source_files(repo_root))
        if not source_files:
            yield ProgressEvent("error", "No parseable source files found in this repository.", pct=0.15)
            return

        # Limit file count
        if len(source_files) > cfg.max_files:
            source_files = source_files[: cfg.max_files]
            yield ProgressEvent(
                "parse",
                f"Repository is large — analysing first {cfg.max_files} files only.",
                pct=0.14,
            )

        all_nodes: list[CodeNode] = []
        for i, fpath in enumerate(source_files, start=1):
            pct = 0.12 + 0.18 * (i / len(source_files))
            nodes = analyzer.parse_file_nodes(fpath, repo_root)
            nodes = nodes[: cfg.max_nodes_per_file]
            all_nodes.extend(nodes)
            yield ProgressEvent(
                "parse",
                f"Parsed {fpath.relative_to(repo_root)} → {len(nodes)} nodes",
                pct=pct,
                detail={"file": str(fpath.relative_to(repo_root)), "nodes": len(nodes)},
            )

        yield ProgressEvent("parse", f"Parsing complete — {len(all_nodes)} code nodes ready.", pct=0.30)

        # ── Stage 3: LLM Review ───────────────────────────────────────
        yield ProgressEvent("review", f"Initialising {cfg.provider} / {cfg.model} …", pct=0.32)
        try:
            agent = CodeReviewAgent(
                provider=cfg.provider,
                model=cfg.model,
                api_key=cfg.api_key or None,
            )
        except Exception as exc:
            yield ProgressEvent("error", f"LLM client init failed: {exc}", pct=0.32)
            return

        all_comments: list[ReviewComment] = []
        errors: list[str] = []

        for idx, node in enumerate(all_nodes, start=1):
            pct = 0.32 + 0.65 * (idx / len(all_nodes))
            yield ProgressEvent(
                "review",
                f"[{idx}/{len(all_nodes)}] Reviewing {node.file_path} → {node.name}",
                pct=pct,
                detail={"current": idx, "total": len(all_nodes)},
            )
            try:
                comments = agent.review_node(node)
                all_comments.extend(comments)
            except Exception as exc:
                msg = f"Error reviewing {node.file_path}::{node.name}: {exc}"
                logger.error(msg)
                errors.append(msg)

        # ── Done ──────────────────────────────────────────────────────
        self.result = PipelineResult(
            repo_name=metadata["name"],
            repo_root=repo_root,
            metadata=metadata,
            all_nodes=all_nodes,
            all_comments=all_comments,
            errors=errors,
        )
        yield ProgressEvent(
            "done",
            f"Review complete — {len(all_comments)} comments across {len(all_nodes)} nodes.",
            pct=1.0,
            detail={
                "total_comments": len(all_comments),
                "total_nodes": len(all_nodes),
                "errors": len(errors),
            },
        )