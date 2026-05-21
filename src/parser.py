"""
src/parser.py
-------------
Stage 2 — AST Parsing & Chunking
Parses Python source files with the built-in `ast` module and produces
discrete, reviewable *code nodes* (functions, classes, top-level statements).
Non-Python files fall back to line-based chunking so the LLM still gets context.
"""

import ast
import textwrap
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

NodeKind = Literal["function", "class", "global_block", "raw_chunk"]

# Maximum lines per chunk sent to the LLM
MAX_CHUNK_LINES = 80
# Minimum lines — skip trivial one-liners (e.g., `pass`)
MIN_CHUNK_LINES = 3


@dataclass
class CodeNode:
    """
    A discrete unit of code extracted from a source file.

    Attributes
    ----------
    file_path : str
        Relative path of the source file (for display).
    kind : NodeKind
        Structural category of the chunk.
    name : str
        Human-readable identifier (function / class name, or a short label).
    source : str
        The raw source text of this chunk.
    start_line : int
        1-based line number where this chunk starts.
    end_line : int
        1-based line number where this chunk ends.
    language : str
        Programming language inferred from the file extension.
    extra : dict
        Optional metadata (docstring, decorators, base classes, …).
    """

    file_path: str
    kind: NodeKind
    name: str
    source: str
    start_line: int
    end_line: int
    language: str = "python"
    extra: dict = field(default_factory=dict)

    @property
    def line_count(self) -> int:
        return self.end_line - self.start_line + 1

    def display_label(self) -> str:
        icons = {"function": "⚙", "class": "◈", "global_block": "≋", "raw_chunk": "⊞"}
        return f"{icons.get(self.kind, '?')} {self.name}  [{self.start_line}–{self.end_line}]"


# ---------------------------------------------------------------------------
# Python AST helpers
# ---------------------------------------------------------------------------

def _get_source_segment(source_lines: list[str], node: ast.AST) -> str:
    """Extract the raw source text for an AST node using line numbers."""
    start = node.lineno - 1          # type: ignore[attr-defined]
    end = node.end_lineno            # type: ignore[attr-defined]
    return "".join(source_lines[start:end])


def _get_docstring(node: ast.AST) -> str:
    """Safely extract a docstring from a function/class node."""
    try:
        return ast.get_docstring(node) or ""  # type: ignore[arg-type]
    except Exception:
        return ""


def _get_decorators(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> list[str]:
    """Return decorator names as strings."""
    result = []
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name):
            result.append(dec.id)
        elif isinstance(dec, ast.Attribute):
            result.append(f"{ast.unparse(dec)}")
        else:
            result.append(ast.unparse(dec))
    return result


def _split_into_chunks(text: str, max_lines: int = MAX_CHUNK_LINES) -> list[str]:
    """Split a large text block into chunks of at most *max_lines* lines."""
    lines = text.splitlines(keepends=True)
    chunks = []
    for i in range(0, len(lines), max_lines):
        chunks.append("".join(lines[i: i + max_lines]))
    return chunks or [""]


# ---------------------------------------------------------------------------
# Main Analyzer
# ---------------------------------------------------------------------------

class ASTStaticAnalyzer:
    """
    Parses source files and extracts reviewable code nodes.

    Python files → full AST extraction (functions, classes, global statements).
    All other supported files → line-based chunking.

    Attributes
    ----------
    target_extensions : list[str]
        Extensions the analyzer will attempt to parse deeply.
    """

    def __init__(self, target_extensions: list[str] | None = None) -> None:
        self.target_extensions: list[str] = target_extensions or [".py"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse_file_nodes(self, file_path: Path, repo_root: str) -> list[CodeNode]:
        """
        Parse *file_path* and return a list of :class:`CodeNode` objects.

        Parameters
        ----------
        file_path : Path
            Absolute path to the source file.
        repo_root : str
            Absolute path to the repo root (used to compute relative paths).
        """
        relative = str(file_path.relative_to(repo_root))
        ext = file_path.suffix.lower()
        language = _ext_to_language(ext)

        try:
            source = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning("Cannot read %s: %s", file_path, exc)
            return []

        if ext == ".py":
            return self._parse_python(source, relative, language)
        else:
            return self._chunk_raw(source, relative, language)

    def extract_line_context(self, file_path: Path, start: int, end: int) -> str:
        """
        Return lines *start..end* (1-based, inclusive) from *file_path* as a string.
        Useful for building extra context around a flagged region.
        """
        try:
            lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
            return "\n".join(lines[start - 1: end])
        except OSError:
            return ""

    # ------------------------------------------------------------------
    # Python-specific parsing
    # ------------------------------------------------------------------

    def _parse_python(self, source: str, relative: str, language: str) -> list[CodeNode]:
        nodes: list[CodeNode] = []
        source_lines = source.splitlines(keepends=True)

        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            logger.warning("Syntax error in %s: %s — falling back to raw chunks", relative, exc)
            return self._chunk_raw(source, relative, language)

        # Track which line ranges are covered by top-level defs/classes
        covered_ranges: list[tuple[int, int]] = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                chunk_nodes = self._extract_function(node, source_lines, relative, language)
                nodes.extend(chunk_nodes)
                covered_ranges.append((node.lineno, node.end_lineno))  # type: ignore

            elif isinstance(node, ast.ClassDef):
                chunk_nodes = self._extract_class(node, source_lines, relative, language)
                nodes.extend(chunk_nodes)
                covered_ranges.append((node.lineno, node.end_lineno))  # type: ignore

        # Capture top-level "global" statements not inside any function/class
        global_lines = self._collect_global_lines(source_lines, covered_ranges)
        if global_lines:
            global_source = "".join(global_lines)
            if len(global_source.strip().splitlines()) >= MIN_CHUNK_LINES:
                # Chunk if too long
                for idx, chunk in enumerate(_split_into_chunks(global_source)):
                    if chunk.strip():
                        nodes.append(CodeNode(
                            file_path=relative,
                            kind="global_block",
                            name=f"module_globals_{idx + 1}" if idx else "module_globals",
                            source=chunk,
                            start_line=1,
                            end_line=len(source_lines),
                            language=language,
                        ))

        return nodes

    def _extract_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        source_lines: list[str],
        relative: str,
        language: str,
    ) -> list[CodeNode]:
        """Extract a top-level function (chunked if too long)."""
        raw = _get_source_segment(source_lines, node)
        raw = textwrap.dedent(raw)
        extra = {
            "docstring": _get_docstring(node),
            "decorators": _get_decorators(node),
            "is_async": isinstance(node, ast.AsyncFunctionDef),
            "args": [a.arg for a in node.args.args],
        }
        chunks = _split_into_chunks(raw)
        result = []
        for idx, chunk in enumerate(chunks):
            if len(chunk.strip().splitlines()) < MIN_CHUNK_LINES:
                continue
            suffix = f"_part{idx + 1}" if len(chunks) > 1 else ""
            result.append(CodeNode(
                file_path=relative,
                kind="function",
                name=f"{node.name}{suffix}",
                source=chunk,
                start_line=node.lineno,  # type: ignore
                end_line=node.end_lineno,  # type: ignore
                language=language,
                extra=extra,
            ))
        return result

    def _extract_class(
        self,
        node: ast.ClassDef,
        source_lines: list[str],
        relative: str,
        language: str,
    ) -> list[CodeNode]:
        """Extract each method of a class as its own node, plus a class-level summary."""
        result: list[CodeNode] = []
        bases = [ast.unparse(b) for b in node.bases]

        # Class-level header (without method bodies) — gives the LLM structural context
        class_header_lines = []
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Extract each method separately
                method_raw = _get_source_segment(source_lines, child)
                method_raw = textwrap.dedent(method_raw)
                extra = {
                    "class_name": node.name,
                    "docstring": _get_docstring(child),
                    "decorators": _get_decorators(child),
                    "base_classes": bases,
                }
                for idx, chunk in enumerate(_split_into_chunks(method_raw)):
                    if len(chunk.strip().splitlines()) < MIN_CHUNK_LINES:
                        continue
                    suffix = f"_part{idx + 1}" if len(_split_into_chunks(method_raw)) > 1 else ""
                    result.append(CodeNode(
                        file_path=relative,
                        kind="function",
                        name=f"{node.name}.{child.name}{suffix}",
                        source=chunk,
                        start_line=child.lineno,  # type: ignore
                        end_line=child.end_lineno,  # type: ignore
                        language=language,
                        extra=extra,
                    ))
            else:
                class_header_lines.append(_get_source_segment(source_lines, child))

        # Emit a class-level node summarising its structure (attributes, bases)
        class_summary_src = _get_source_segment(source_lines, node)
        # Truncate to first MAX_CHUNK_LINES lines if the class is huge
        summary_lines = class_summary_src.splitlines()[:MAX_CHUNK_LINES]
        class_summary_src = "\n".join(summary_lines)
        if len(class_summary_src.strip().splitlines()) >= MIN_CHUNK_LINES:
            result.insert(0, CodeNode(
                file_path=relative,
                kind="class",
                name=node.name,
                source=class_summary_src,
                start_line=node.lineno,  # type: ignore
                end_line=node.end_lineno,  # type: ignore
                language=language,
                extra={"base_classes": bases, "docstring": _get_docstring(node)},
            ))
        return result

    # ------------------------------------------------------------------
    # Raw / fallback chunking
    # ------------------------------------------------------------------

    def _chunk_raw(self, source: str, relative: str, language: str) -> list[CodeNode]:
        """Chunk non-Python files by line count."""
        chunks = _split_into_chunks(source)
        result = []
        for idx, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            lines_so_far = idx * MAX_CHUNK_LINES
            result.append(CodeNode(
                file_path=relative,
                kind="raw_chunk",
                name=f"chunk_{idx + 1}",
                source=chunk,
                start_line=lines_so_far + 1,
                end_line=lines_so_far + len(chunk.splitlines()),
                language=language,
            ))
        return result

    @staticmethod
    def _collect_global_lines(
        source_lines: list[str],
        covered: list[tuple[int, int]],
    ) -> list[str]:
        """Return lines NOT covered by any function or class definition."""
        covered_set: set[int] = set()
        for start, end in covered:
            covered_set.update(range(start, end + 1))
        return [
            line for i, line in enumerate(source_lines, start=1)
            if i not in covered_set
        ]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _ext_to_language(ext: str) -> str:
    return {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".go": "go",
        ".java": "java",
        ".cpp": "cpp",
        ".c": "c",
        ".rs": "rust",
    }.get(ext, "text")