"""
src/ingestor.py
---------------
Stage 1 — Repository Ingestion
Clones a GitHub repository into a local workspace, validates the URL,
and exposes helpers to enumerate source files for parsing.
"""

import os
import re
import shutil
import tempfile
import logging
from pathlib import Path
from typing import Iterator

import git  # GitPython

logger = logging.getLogger(__name__)

# File extensions we know how to parse (Python only for AST; others go raw)
SUPPORTED_EXTENSIONS: set[str] = {".py", ".js", ".ts", ".go", ".java", ".cpp", ".c", ".rs"}
SKIP_DIRS: set[str] = {
    ".git", "__pycache__", "node_modules", ".venv", "venv", "env",
    "dist", "build", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "migrations", "vendor",
}
MAX_FILE_SIZE_BYTES = 300_000  # 300 KB — skip very large generated files


def _validate_github_url(url: str) -> str:
    """Normalise and validate a GitHub URL. Returns the sanitised URL."""
    url = url.strip().rstrip("/")
    # Accept both HTTPS and SSH patterns
    patterns = [
        r"^https://github\.com/[\w.\-]+/[\w.\-]+(\.git)?$",
        r"^git@github\.com:[\w.\-]+/[\w.\-]+(\.git)?$",
    ]
    if not any(re.match(p, url) for p in patterns):
        raise ValueError(
            f"'{url}' does not look like a valid GitHub repository URL.\n"
            "Expected format: https://github.com/<owner>/<repo>"
        )
    # Ensure HTTPS clone URL
    if url.startswith("git@"):
        url = "https://github.com/" + url[len("git@github.com:"):].replace(":", "/")
    if not url.endswith(".git"):
        url += ".git"
    return url


class RepositoryIngestor:
    """
    Clones a remote GitHub repository into a temporary workspace on disk.

    Attributes
    ----------
    target_workspace : str
        Absolute path to the directory where the repo will be cloned.
    """

    def __init__(self, target_workspace: str | None = None) -> None:
        if target_workspace is None:
            target_workspace = tempfile.mkdtemp(prefix="code_review_")
        self.target_workspace: str = str(target_workspace)
        self._repo: git.Repo | None = None
        self._repo_name: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clone_repo(self, github_url: str, depth: int = 1) -> str:
        """
        Clone *github_url* into ``self.target_workspace``.

        Parameters
        ----------
        github_url : str
            Full GitHub HTTPS or SSH URL.
        depth : int
            Shallow clone depth (1 = latest snapshot only, speeds up cloning).

        Returns
        -------
        str
            Absolute path to the cloned repository root.
        """
        clean_url = _validate_github_url(github_url)
        self._repo_name = clean_url.rstrip(".git").split("/")[-1]
        clone_dir = os.path.join(self.target_workspace, self._repo_name)

        if os.path.exists(clone_dir):
            logger.info("Workspace already exists — wiping before re-clone.")
            shutil.rmtree(clone_dir)

        logger.info("Cloning %s → %s (depth=%d)", clean_url, clone_dir, depth)
        try:
            self._repo = git.Repo.clone_from(
                clean_url,
                clone_dir,
                depth=depth,
                multi_options=["--single-branch"],
            )
        except git.exc.GitCommandError as exc:
            raise RuntimeError(
                f"Git clone failed for '{clean_url}'.\n"
                f"Reason: {exc.stderr.strip() if exc.stderr else str(exc)}"
            ) from exc

        return clone_dir

    def wipe_workspace(self) -> None:
        """Delete the entire workspace directory tree."""
        if os.path.exists(self.target_workspace):
            shutil.rmtree(self.target_workspace)
            logger.info("Wiped workspace: %s", self.target_workspace)

    def iter_source_files(self, repo_root: str) -> Iterator[Path]:
        """
        Yield Path objects for every parseable source file in *repo_root*.

        Skips hidden directories, dependency trees, and oversized files.
        """
        root = Path(repo_root)
        for path in root.rglob("*"):
            # Skip non-files
            if not path.is_file():
                continue
            # Skip blacklisted directories anywhere in the path
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            # Skip unsupported extensions
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            # Skip gigantic files (generated code, minified bundles, etc.)
            try:
                if path.stat().st_size > MAX_FILE_SIZE_BYTES:
                    logger.warning("Skipping large file: %s", path)
                    continue
            except OSError:
                continue
            yield path

    def get_repo_metadata(self, repo_root: str) -> dict:
        """Return lightweight metadata about the cloned repo."""
        root = Path(repo_root)
        all_files = list(self.iter_source_files(repo_root))
        ext_counts: dict[str, int] = {}
        for f in all_files:
            ext_counts[f.suffix] = ext_counts.get(f.suffix, 0) + 1

        return {
            "name": self._repo_name,
            "root": str(root),
            "total_source_files": len(all_files),
            "extension_breakdown": ext_counts,
        }