"""
__main__.py
-----------
CLI entry point: python -m ai_code_review <github_url>
Allows running the pipeline headlessly (no Streamlit), useful for CI/CD.

Usage:
    python -m ai_code_review https://github.com/owner/repo [--provider anthropic] [--output report.md]
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="codesight",
        description="CodeSight — Autonomous AI Code Review Agent (CLI mode)",
    )
    parser.add_argument("github_url", help="Public GitHub repository URL")
    parser.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"])
    parser.add_argument("--model", default="", help="LLM model string (auto-selected if empty)")
    parser.add_argument("--api-key", default="", help="LLM API key (falls back to env var)")
    parser.add_argument("--max-files", type=int, default=20)
    parser.add_argument("--max-nodes", type=int, default=15)
    parser.add_argument("--output", default="review_report.md", help="Output Markdown file path")
    parser.add_argument("--json-output", default="", help="Optional JSON output file path")
    args = parser.parse_args()

    # Lazy imports (keep startup fast)
    from src.pipeline import PipelineConfig, ReviewPipeline
    from src.markdown_reporter import generate_markdown_report

    model = args.model or (
        "claude-sonnet-4-20250514" if args.provider == "anthropic" else "gpt-4o-mini"
    )
    api_key = args.api_key or os.environ.get(
        "ANTHROPIC_API_KEY" if args.provider == "anthropic" else "OPENAI_API_KEY", ""
    )
    if not api_key:
        print(f"ERROR: No API key found. Set {'ANTHROPIC_API_KEY' if args.provider == 'anthropic' else 'OPENAI_API_KEY'} env var.")
        sys.exit(1)

    cfg = PipelineConfig(
        github_url=args.github_url,
        provider=args.provider,
        model=model,
        api_key=api_key,
        max_files=args.max_files,
        max_nodes_per_file=args.max_nodes,
    )

    pipeline = ReviewPipeline(cfg)
    print(f"\n🔬 CodeSight — reviewing {args.github_url}\n")

    for event in pipeline.run():
        bar = "█" * int(event.pct * 30) + "░" * (30 - int(event.pct * 30))
        print(f"[{bar}] {event.stage.upper():8s} {event.message}")
        if event.stage == "error":
            print(f"\n❌ Pipeline error: {event.message}")
            sys.exit(1)

    result = pipeline.result
    if result is None:
        print("Pipeline produced no result.")
        sys.exit(1)

    print(f"\n✅ Done — {len(result.all_comments)} comments across {len(result.all_nodes)} nodes.\n")

    # Write Markdown report
    md = generate_markdown_report(
        repo_name=result.repo_name,
        comments=result.all_comments,
        metadata=result.metadata,
        errors=result.errors,
    )
    Path(args.output).write_text(md, encoding="utf-8")
    print(f"📄 Markdown report → {args.output}")

    # Write JSON (optional)
    if args.json_output:
        data = [c.to_dict() for c in result.all_comments]
        Path(args.json_output).write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"📦 JSON output    → {args.json_output}")

    # Print quick summary to stdout
    print("\n── Summary ─────────────────────────────────")
    for sev in ("critical", "high", "medium", "low", "info"):
        count = sum(1 for c in result.all_comments if c.severity == sev)
        if count:
            icons = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🔵", "info": "⚪"}
            print(f"  {icons[sev]} {sev.capitalize():15s} {count}")
    low_c = sum(1 for c in result.all_comments if c.low_confidence)
    print(f"\n  ⚠️  Needs verification: {low_c}")
    print("────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()