# 🔬 CodeSight — AI Code Review Agent

> **Autonomous AI-powered code review:** clone → parse → analyze → report.  
> Built for the CipherSchools Advanced AI Assignment.

[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-deployment-url.streamlit.app)

---

## 📖 Project Overview

CodeSight is a fully autonomous code review agent that:

1. **Clones** any public GitHub repository using GitPython (shallow clone for speed)
2. **Parses** Python source files using Python's built-in `ast` module, extracting functions, classes, and global-scope blocks as discrete reviewable units
3. **Reviews** each unit via a structured LLM prompt (Claude Sonnet or GPT-4o-mini) that returns schema-valid JSON comments
4. **Rates** each comment with a 0–100 confidence score — low-confidence items are flagged with a "Verify This" label (epistemic humility in production AI)
5. **Presents** results in a Streamlit dashboard with severity filters, category filters, Plotly charts, and CSV/JSON export
6. **Posts** (optional bonus) inline review comments directly to a GitHub Pull Request via the GitHub REST API

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        app.py  (Streamlit UI)                   │
│  sidebar config → run button → progress stream → result render  │
└───────────────────────────┬─────────────────────────────────────┘
                            │ PipelineConfig
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   src/pipeline.py  (Orchestrator)               │
│                                                                 │
│  Stage 1: RepositoryIngestor                                    │
│    GitPython.clone_from() → shallow clone → validate URL        │
│           │                                                     │
│  Stage 2: ASTStaticAnalyzer                                     │
│    ast.parse() → FunctionDef / ClassDef / global statements     │
│    → CodeNode objects (chunked at 80 lines)                     │
│           │                                                     │
│  Stage 3: CodeReviewAgent                                       │
│    Structured prompt → LLM JSON → verify_json() → ReviewComment │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
               ┌────────────────────────┐
               │  PipelineResult        │
               │  .all_comments         │
               │  .high_confidence      │
               │  .low_confidence       │
               │  .by_severity          │
               └────────────────────────┘
                            │
               Optional:  src/github_poster.py
               POST /repos/:owner/:repo/pulls/:pr/reviews
```

### File Structure

```
ai-code-review-agent/
├── app.py                   # Streamlit dashboard (Stage 4)
├── src/
│   ├── __init__.py
│   ├── ingestor.py          # Stage 1 — GitPython repo cloning
│   ├── parser.py            # Stage 2 — AST parsing & chunking
│   ├── reviewer.py          # Stage 3 — LLM review + JSON schema
│   ├── pipeline.py          # Orchestrator (Stages 1→2→3)
│   └── github_poster.py     # Bonus — GitHub PR API integration
├── .streamlit/
│   └── config.toml          # Streamlit dark theme config
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚙ Setup

### Prerequisites
- Python 3.11+
- Git installed locally

### Local Installation

```bash
# 1. Clone this repo
git clone https://github.com/your-username/ai-code-review-agent
cd ai-code-review-agent

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API keys
cp .env.example .env
# Edit .env and fill in your keys

# 5. Run
streamlit run app.py
```

### Streamlit Cloud Deployment

1. Push this repo to GitHub (public)
2. Go to [share.streamlit.io](https://share.streamlit.io) → New App
3. Select your repo, branch `main`, file `app.py`
4. Under **Settings → Secrets**, add:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   OPENAI_API_KEY    = "sk-..."   # optional
   GITHUB_TOKEN      = "ghp_..."  # optional, for PR posting
   ```
5. Deploy!

---

## 🔑 Configuration

| Setting | Where | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | `.env` / Streamlit Secrets | Required if using Claude |
| `OPENAI_API_KEY` | `.env` / Streamlit Secrets | Required if using GPT-4o-mini |
| `GITHUB_TOKEN` | `.env` / Streamlit Secrets | Optional — PR comment posting |
| Max files | Sidebar slider | Caps files reviewed (default 20) |
| Max nodes/file | Sidebar slider | Caps AST nodes per file (default 15) |

---

## 🎯 Key Design Decisions

### Chunking Strategy
Large functions are split at 80-line boundaries before being sent to the LLM. This prevents context overflow and keeps prompts focused. The `CodeNode` carries both `start_line` and `end_line` so the LLM can return meaningful `line_hint` values.

### Prompt Engineering
The system prompt enforces strict JSON output (array only, no prose). It defines `confidence` semantics explicitly (90–100 = certain, 0–49 = speculative) so the model self-calibrates rather than always returning 100. The `response_format: json_object` parameter is used with OpenAI; Anthropic relies on prompt instruction.

### Confidence Scoring & Epistemic Humility
Every comment carries a `confidence` integer (0–100). Comments below 60 are displayed in a separate "Verify This" bucket with a visual amber label. This is a production-grade pattern: the agent signals its own uncertainty rather than presenting all output as equally authoritative.

### Retry & Fallback
The `CodeReviewAgent` retries up to 3 times with exponential back-off on transient errors. If JSON parsing fails after all retries, the node is skipped (logged as an error) and the pipeline continues.

---

## ⚠️ Known Limitations

1. **Python-only deep AST**: Non-Python files (JS, Go, etc.) fall back to line-based chunking. Full tree-sitter integration is planned.
2. **Public repos only** (unless a GitHub token with `repo` scope is provided).
3. **No caching**: Each run re-clones and re-reviews. Adding a SQLite cache keyed on file hash would dramatically reduce API costs on repeated runs.
4. **Rate limits**: Very large repos (>50 files, >500 functions) may hit LLM rate limits. The `max_files` slider mitigates this.
5. **LLM hallucination**: Despite structured prompts, the model occasionally invents issues. The confidence score is the primary mitigation — low-confidence comments should always be human-reviewed.

---

## 🚀 What I'd Build Next

- **tree-sitter integration** for JavaScript, TypeScript, Go, Rust — true AST for all languages
- **SQLite result cache** keyed on `(file_path, file_hash)` — skip unchanged files on re-runs
- **Diff-mode** — only review changed lines in a PR, not the whole file
- **Severity trend charts** across commits (timeline view)
- **One-click GitHub PR creation** with the review as a structured check run
- **Vector search over past reviews** — surface similar issues found in other repos

---

## 📚 Data Sources & Citations

- Test repositories used during development:
  - [pallets/flask](https://github.com/pallets/flask) — Python web framework
  - [psf/requests](https://github.com/psf/requests) — HTTP library
  - [tiangolo/fastapi](https://github.com/tiangolo/fastapi) — FastAPI framework

---

## 📄 License

MIT — see `LICENSE`.