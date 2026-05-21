"""
src/github_poster.py
--------------------
Optional Bonus — Post review comments as inline PR review comments via GitHub API.
Requires a Personal Access Token with repo:write scope.
"""

import logging
import os
import re
from typing import Any

import requests

from src.reviewer import ReviewComment

logger = logging.getLogger(__name__)

GITHUB_API_BASE = "https://api.github.com"


def _extract_owner_repo(github_url: str) -> tuple[str, str]:
    """Parse 'owner' and 'repo' from a GitHub URL."""
    match = re.search(r"github\.com[/:]([^/]+)/([^/.]+)", github_url)
    if not match:
        raise ValueError(f"Cannot extract owner/repo from URL: {github_url}")
    return match.group(1), match.group(2)


class GitHubPRPoster:
    """
    Posts ReviewComment objects as inline review comments on a GitHub Pull Request.

    Parameters
    ----------
    github_url : str
        Full repository URL, e.g. https://github.com/owner/repo
    pr_number : int
        Target pull-request number.
    token : str | None
        GitHub Personal Access Token.  Falls back to GITHUB_TOKEN env var.
    """

    def __init__(
        self,
        github_url: str,
        pr_number: int,
        token: str | None = None,
    ) -> None:
        self.owner, self.repo = _extract_owner_repo(github_url)
        self.pr_number = pr_number
        self.token = token or os.environ.get("GITHUB_TOKEN", "")
        if not self.token:
            raise ValueError("GITHUB_TOKEN not set — cannot post PR comments.")
        self._session = self._make_session()

    def _make_session(self) -> requests.Session:
        s = requests.Session()
        s.headers.update({
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })
        return s

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_pr_head_sha(self) -> str:
        """Fetch the HEAD commit SHA of the pull request."""
        url = f"{GITHUB_API_BASE}/repos/{self.owner}/{self.repo}/pulls/{self.pr_number}"
        resp = self._session.get(url)
        resp.raise_for_status()
        return resp.json()["head"]["sha"]

    def post_review(self, comments: list[ReviewComment], body: str = "") -> dict[str, Any]:
        """
        Submit a pending review with all *comments* attached.

        Parameters
        ----------
        comments : list[ReviewComment]
            Review comments to post.
        body : str
            Top-level review body text (optional).

        Returns
        -------
        dict
            GitHub API response for the created review.
        """
        sha = self.get_pr_head_sha()
        gh_comments = []
        for c in comments:
            # Convert our line_hint (e.g. "12" or "12-18") to an integer
            try:
                line = int(c.line_hint.split("-")[0])
            except (ValueError, AttributeError):
                line = 1

            severity_emoji = {
                "critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🔵", "info": "⚪",
            }.get(c.severity, "⚪")

            confidence_note = (
                f"\n\n> ⚠️ **Verify this** — confidence: {c.confidence}%"
                if c.low_confidence
                else f"\n\n> Confidence: {c.confidence}%"
            )

            comment_body = (
                f"{severity_emoji} **[{c.severity.upper()}] {c.title}**\n\n"
                f"{c.body}\n\n"
                f"**Suggestion:** {c.suggestion}"
                f"{confidence_note}"
            )

            gh_comments.append({
                "path": c.file_path,
                "line": line,
                "side": "RIGHT",
                "body": comment_body,
            })

        if not gh_comments:
            logger.info("No comments to post.")
            return {}

        if not body:
            high = sum(1 for c in comments if c.severity in ("critical", "high"))
            body = (
                f"## 🤖 AI Code Review\n\n"
                f"Found **{len(comments)}** issue(s) — {high} high/critical.\n"
                f"_Low-confidence comments are marked with ⚠️ Verify this._"
            )

        url = f"{GITHUB_API_BASE}/repos/{self.owner}/{self.repo}/pulls/{self.pr_number}/reviews"
        payload = {
            "commit_id": sha,
            "body": body,
            "event": "COMMENT",
            "comments": gh_comments,
        }
        resp = self._session.post(url, json=payload)
        resp.raise_for_status()
        logger.info("Posted %d PR comments to PR #%d", len(gh_comments), self.pr_number)
        return resp.json()

    def post_summary_issue_comment(self, comments: list[ReviewComment]) -> dict[str, Any]:
        """Post a summary comment on the PR (not inline)."""
        by_severity: dict[str, int] = {}
        for c in comments:
            by_severity[c.severity] = by_severity.get(c.severity, 0) + 1

        lines = ["## 🤖 AI Code Review Summary\n"]
        for sev in ("critical", "high", "medium", "low", "info"):
            count = by_severity.get(sev, 0)
            if count:
                emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🔵", "info": "⚪"}[sev]
                lines.append(f"- {emoji} **{sev.capitalize()}**: {count}")

        low_conf = sum(1 for c in comments if c.low_confidence)
        if low_conf:
            lines.append(f"\n> ⚠️ {low_conf} comment(s) have low confidence — please verify manually.")

        body = "\n".join(lines)
        url = f"{GITHUB_API_BASE}/repos/{self.owner}/{self.repo}/issues/{self.pr_number}/comments"
        resp = self._session.post(url, json={"body": body})
        resp.raise_for_status()
        return resp.json()