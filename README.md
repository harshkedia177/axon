<p align="center">
  <img src="https://raw.githubusercontent.com/harshkedia177/axon/main/axon-logo.png" alt="Axon logo" width="420" />
</p>

# Axon

[![PyPI version](https://img.shields.io/pypi/v/axoniq)](https://pypi.org/project/axoniq/)
[![PyPI downloads](https://static.pepy.tech/badge/axoniq/month)](https://pepy.tech/projects/axoniq)
[![Total installs](https://static.pepy.tech/badge/axoniq)](https://pepy.tech/projects/axoniq)
[![GitHub stars](https://img.shields.io/github/stars/harshkedia177/axon?style=social)](https://github.com/harshkedia177/axon)
[![License](https://img.shields.io/pypi/l/axoniq)](https://github.com/harshkedia177/axon/blob/main/LICENSE)

**The knowledge graph for your codebase — explore it visually, or let your AI agent query it.**

Indexes any codebase into a structural knowledge graph — every dependency, call chain, cluster, and execution flow. Explore it through an **interactive web dashboard** with force-directed graph visualization, or expose it through **MCP tools** so AI agents get full structural understanding in every tool call.

```
$ axon analyze .

Walking files...               142 files found
Parsing code...                142/142
Tracing calls...               847 calls resolved
Analyzing types...             234 type relationships
Detecting communities...       8 clusters found
Detecting execution flows...   34 processes found
Finding dead code...           12 unreachable symbols
Analyzing git history...       18 coupled file pairs
Generating embeddings...       623 vectors stored

Done in 4.2s — 623 symbols, 1,847 edges, 8 clusters, 34 flows
```

Then explore your codebase visually:

```bash
axon ui                      # Opens interactive dashboard at localhost:8420
```

**Three views, one command:**

- **Explorer** — Interactive force-directed graph (Sigma.js + WebGL). Click any node to see its code, callers, callees, impact radius, and community. Community hull overlays show architectural clusters at a glance.
- **Analysis** — Health score, coupling heatmap, dead code report, inheritance tree, branch diff — your codebase health in one dashboard.
- **Cypher Console** — Write and run Cypher queries against the graph with syntax highlighting, presets, and history.

Plus: command palette (`Cmd+K`), keyboard shortcuts, flow trace animations, graph minimap, and SSE-powered live reload when watch mode is active.

---

## The Problem

Your AI agent edits `UserService.validate()`. It doesn't know that 47 functions depend on that return type, 3 execution flows pass through it, and `payment_handler.py` changes alongside it 80% of the time.

**Breaking changes ship.**

This happens because AI agents work with flat text. They grep for callers, miss indirect ones, and have no understanding of how code is *connected*. Context windows are finite. LSPs don't expose call graphs. Grepping gives you strings, not structure.

The agent needs a **knowledge graph** — not more text.

---

## How Axon Solves It

Most code intelligence tools give the agent raw files and hope it reads enough. Axon takes a different approach: **precompute structure at index time** so every tool call returns complete, actionable context.

A 12-phase pipeline runs once over your repo. After that:

- `axon_impact("validate")` returns all 47 affected symbols, grouped by depth (will break / may break / review), with confidence scores — in a single call
- `axon_query("auth handler")` returns hybrid-ranked results grouped by execution flow, not a flat list of name matches
- `axon_context("UserService")` returns callers, callees, type references, community membership, and dead code status — the full picture

**Three benefits:**

1. **Reliability** — the context is already in the tool response. No multi-step exploration that can miss code.
2. **Token efficiency** — one tool call instead of a 10-query search chain. Agents spend tokens on reasoning, not navigation.
3. **Model democratization** — even smaller models get full architectural clarity because the tools do the heavy lifting.

**Zero cloud dependencies.** Everything runs locally — parsing, graph storage, embeddings, search. No API keys, no data leaving your machine.

---

## TL;DR

```bash
pip install axoniq            # 1. Install
cd your-project && axon analyze .  # 2. Index (one command, ~5s for most repos)
axon ui                       # 3. Explore visually at localhost:8420
```

**For AI agents** — add to `.mcp.json` in your project root:

```json
{
  "mcpServers": {
    "axon": {
      "command": "axon",
      "args": ["serve", "--watch"]
    }
  }
}
```

**For developers** — explore the graph yourself:

```bash
axon ui                      # Interactive dashboard (standalone or attaches to running host)
axon ui --watch              # Live reload on file changes
axon host --watch            # Shared host: UI + multi-session MCP
```

---

## What You Get

### Explore your codebase visually

**Web UI**

A full interactive dashboard — no terminal or extensions required. One command:

```bash
axon ui                           # Launch at localhost:8420
axon ui --watch                   # Live reload on file changes
axon ui --port 9000               # Custom port
axon ui --dev                     # Dev mode (Vite HMR on :5173)
```

| View | What It Shows |
|------|--------------|
| **Explorer** | Interactive force-directed graph (Sigma.js + WebGL), file tree sidebar, symbol detail panel with code preview, callers/callees, impact analysis, and process memberships. Community hull overlays reveal architectural clusters. |
| **Analysis** | Health score, coupling heatmap, dead code report, inheritance tree visualization, branch diff, and aggregate stats — your codebase health at a glance. |
| **Cypher Console** | Query editor with syntax highlighting, preset query library, results table, and query history. |

**Extras:** Command palette (`Cmd+K`), keyboard shortcuts, graph minimap, flow trace and impact ripple animations, SSE-powered live reload when watch mode is enabled.

The UI is backed by a FastAPI server with a full REST API — see [API Endpoints](#api-endpoints) below.

### Find anything — by name, concept, or typo

**Hybrid Search (BM25 + Vector + Fuzzy)**

Three search strategies fused with [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf):

- **BM25 full-text search** — fast exact name and keyword matching via KuzuDB FTS
- **Semantic vector search** — conceptual queries via 384-dim embeddings (BAAI/bge-small-en-v1.5)
- **Fuzzy name search** — Levenshtein fallback for typos and partial matches

Results are ranked with test file down-ranking (0.5x) and source function/class boosting (1.2x), then **grouped by execution flow** so the agent sees architectural context in a single call.

### Know what breaks before you change it

**Impact Analysis with Depth Grouping**

When you're about to change a symbol, Axon traces upstream through the call graph, type references, and git coupling history. Results are grouped by depth for actionability:

- **Depth 1** — Direct callers (will break)
- **Depth 2** — Indirect callers (may break)
- **Depth 3+** — Transitive (review)

Every edge carries a confidence score (1.0 = exact match, 0.8 = receiver method, 0.5 = fuzzy) so you can prioritize what to review.

### Find what to delete

**Dead Code Detection**

Not just "zero callers" — a multi-pass analysis that understands your framework:

1. **Initial scan** — flags symbols with no incoming calls
2. **Exemptions** — entry points, exports, constructors, test code, dunder methods, `__init__.py` symbols, decorated functions, `@property` methods
3. **Override pass** — un-flags methods overriding non-dead base class methods
4. **Protocol conformance** — un-flags methods on Protocol-conforming classes
5. **Protocol stubs** — un-flags all methods on Protocol classes (interface contracts)

### Understand how code runs, not just where it sits

**Execution Flow Tracing**

Detects entry points using framework-aware patterns:
- **Python**: `@app.route`, `@router.get`, `@click.command`, `test_*` functions, `__main__` blocks
- **JavaScript/TypeScript**: Express handlers, exported functions, `handler`/`middleware` patterns

Then traces BFS execution flows from each entry point through the call graph, classifying flows as intra-community or cross-community.

### See your architecture without reading docs

**Community Detection**

Uses the [Leiden algorithm](https://www.nature.com/articles/s41598-019-41695-z) (igraph + leidenalg) to automatically discover functional clusters. Each community gets a cohesion score and auto-generated label. Agents can ask "what cluster does this symbol belong to?" and get the answer without reading a single design doc.

### Find hidden dependencies git knows about

**Change Coupling (Git History)**

Analyzes 6 months of git history to find dependencies that static analysis misses:

```
coupling(A, B) = co_changes(A, B) / max(changes(A), changes(B))
```

Files with coupling strength >= 0.3 and 3+ co-changes get linked. These show up in impact analysis — so when you change `user.py`, the agent also knows to check `user_test.py` and `auth_middleware.py`.

### Always up to date

**Watch Mode**

Live re-indexing powered by a Rust-based file watcher (watchfiles):

```bash
$ axon watch
Watching /Users/you/project for changes...

[10:32:15] src/auth/validate.py modified -> re-indexed (0.3s)
[10:33:02] 2 files modified -> re-indexed (0.5s)
```

File-local phases (parse, imports, calls, types) run immediately on change. Global phases (communities, processes, dead code) batch every 30 seconds.

### Structural diff, not text diff

**Branch Comparison**

Compare branches at the symbol level using git worktrees (no stashing required):

```bash
$ axon diff main..feature

Symbols added (4):
  + process_payment (Function) -- src/payments/stripe.py
  + PaymentIntent (Class) -- src/payments/models.py

Symbols modified (2):
  ~ checkout_handler (Function) -- src/routes/checkout.py

Symbols removed (1):
  - old_charge (Function) -- src/payments/legacy.py
```

### Clean call graphs

**Noise Filtering**

Built-in blocklist (138 entries) automatically filters language builtins (`print`, `len`, `isinstance`), JS/TS globals (`console`, `setTimeout`, `fetch`), React hooks (`useState`, `useEffect`), and common stdlib methods from the call graph. Your graph shows *your* code's relationships, not noise from `list.append()`.

---

## The Pipeline

Axon builds deep structural understanding through 12 sequential analysis phases:

| Phase | What It Does |
|-------|-------------|
| **File Walking** | Walks repo respecting `.gitignore`, filters by supported languages |
| **Structure** | Creates File/Folder hierarchy with CONTAINS relationships |
| **Parsing** | tree-sitter AST extraction — functions, classes, methods, interfaces, enums, type aliases |
| **Import Resolution** | Resolves import statements to actual files (relative, absolute, bare specifiers) |
| **Call Tracing** | Maps function calls with confidence scores. Noise filtering skips 138 language builtins |
| **Heritage** | Tracks class inheritance (EXTENDS) and interface implementation (IMPLEMENTS) |
| **Type Analysis** | Extracts type references from parameters, return types, and variable annotations |
| **Community Detection** | Leiden algorithm clusters related symbols into functional communities |
| **Process Detection** | Framework-aware entry point detection + BFS flow tracing |
| **Dead Code Detection** | Multi-pass analysis with override, protocol, and decorator awareness |
| **Change Coupling** | Git history analysis — finds files that always change together |
| **Embeddings** | 384-dim vectors for every symbol, enabling semantic search. Skip with `--no-embeddings` |

---

## MCP Integration

Axon exposes its full intelligence as an MCP server. Set it up once, and your AI agent has structural understanding of your codebase forever.

### Setup

**Claude Code** — add to `.mcp.json` in your project root (or run `claude mcp add axon -- axon serve --watch`):

```json
{
  "mcpServers": {
    "axon": {
      "command": "axon",
      "args": ["serve", "--watch"]
    }
  }
}
```

**Cursor** — add to your MCP settings:

```json
{
  "axon": {
    "command": "axon",
    "args": ["serve", "--watch"]
  }
}
```

Optional new feature:

```bash
axon host --watch
```

This starts a shared host for the UI and multiple MCP clients. `axon setup --claude` / `axon setup --cursor` still prints the standard config.

The `--watch` flag enables live re-indexing — the graph updates as you edit code.

### Tools

| Tool | What the agent gets |
|------|-------------|
| `axon_query` | Hybrid search (BM25 + vector + fuzzy) with results grouped by execution flow |
| `axon_context` | 360-degree view — callers, callees, type refs, confidence tags, dead code status |
| `axon_impact` | Blast radius grouped by depth — direct (will break), indirect (may break), transitive |
| `axon_dead_code` | All unreachable symbols grouped by file |
| `axon_detect_changes` | Map a `git diff` to affected symbols and execution flows |
| `axon_list_repos` | All indexed repositories with stats |
| `axon_cypher` | Read-only Cypher queries against the knowledge graph |

Every tool response includes a **next-step hint** guiding the agent through a natural investigation workflow:

```
query   -> "Next: Use context() on a specific symbol for the full picture."
context -> "Next: Use impact() if planning changes to this symbol."
impact  -> "Tip: Review each affected symbol before making changes."
```

### Resources

| URI | Description |
|-----|-------------|
| `axon://overview` | Node and relationship counts by type |
| `axon://dead-code` | Full dead code report |
| `axon://schema` | Graph schema reference for Cypher queries |

---

## API Endpoints

The web UI is backed by a FastAPI server. All endpoints are under `/api`:

| Endpoint | Description |
|----------|-------------|
| `GET /api/graph` | Full knowledge graph (paginated) |
| `GET /api/node/{id}` | Node detail with callers, callees, type refs |
| `GET /api/overview` | Aggregate node/edge counts |
| `GET /api/search` | Hybrid search (BM25 + vector + fuzzy) |
| `GET /api/impact/{id}` | Blast radius analysis by depth |
| `GET /api/dead-code` | Dead code report |
| `GET /api/communities` | Community listing with members |
| `GET /api/coupling` | Change coupling heatmap data |
| `GET /api/files/{path}` | Source file content with syntax context |
| `POST /api/cypher` | Execute read-only Cypher queries |
| `GET /api/diff` | Structural branch comparison |
| `GET /api/processes` | Execution flow listing |
| `GET /api/events` | SSE stream for live reload events |
| `POST /api/reindex` | Trigger a full re-index (watch mode only) |

Cypher queries are validated server-side — write keywords (`CREATE`, `DELETE`, `DROP`, etc.) are rejected after comment stripping.

---

## How It Compares

| Capability | grep / ripgrep | LSP | Context window stuffing | Axon |
|-----------|---------------|-----|------------------------|------|
| **Interactive graph UI** | **No** | **No** | **No** | **Yes (full web dashboard)** |
| Text search | Yes | No | Yes | Yes (hybrid BM25 + vector) |
| Find all callers | No | Partial | Hit-or-miss | Yes (full call graph with confidence) |
| Type relationships | No | Yes | No | Yes (param/return/variable roles) |
| Dead code detection | No | No | No | Yes (multi-pass, framework-aware) |
| Execution flow tracing | No | No | No | Yes (entry point -> flow) |
| Community detection | No | No | No | Yes (Leiden algorithm) |
| Change coupling (git) | No | No | No | Yes (6-month co-change analysis) |
| Impact analysis | No | No | No | Yes (depth-grouped with confidence) |
| AI agent integration | No | Partial | N/A | Yes (full MCP server) |
| Structural branch diff | No | No | No | Yes (node/edge level) |
| Watch mode | No | Yes | No | Yes (Rust-based, 500ms debounce) |
| Works offline | Yes | Yes | No | Yes |

---

## Supported Languages

| Language | Extensions | Parser |
|----------|-----------|--------|
| Python | `.py` | tree-sitter-python |
| TypeScript | `.ts`, `.tsx` | tree-sitter-typescript |
| JavaScript | `.js`, `.jsx`, `.mjs`, `.cjs` | tree-sitter-javascript |

---

## Installation

```bash
# With pip
pip install axoniq

# With uv (recommended)
uv add axoniq

# With Neo4j backend support
pip install axoniq[neo4j]
```

Requires **Python 3.11+**. The web UI (frontend + backend) is included — no Node.js or extra install needed.

### From Source

```bash
git clone https://github.com/harshkedia177/axon.git
cd axon
uv sync --all-extras
uv run axon --help
```

To rebuild the frontend after making changes (requires Node.js 18+):

```bash
cd src/axon/web/frontend
npm install && npm run build
```

---

## CLI Reference

```
axon analyze [PATH]          Index a repository (default: current directory)
    --full                   Force full rebuild (skip incremental)
    --no-embeddings          Skip vector embedding generation (faster indexing)

axon status                  Show index status for current repo
axon list                    List all indexed repositories (auto-populated on analyze)
axon clean                   Delete index for current repo
    --force / -f             Skip confirmation prompt

axon query QUERY             Hybrid search the knowledge graph
    --limit / -n N           Max results (default: 20)

axon context SYMBOL          360-degree view of a symbol
axon impact SYMBOL           Blast radius analysis
    --depth / -d N           BFS traversal depth (default: 3)

axon dead-code               List all detected dead code
axon cypher QUERY            Execute a raw Cypher query (read-only)

axon watch                   Watch mode — live re-indexing on file changes
axon diff BASE..HEAD         Structural branch comparison

axon host                    Shared host for UI + HTTP MCP (default: localhost:8420)
    --port / -p PORT         Port to serve on (default: 8420)
    --watch / --no-watch     Enable live file watching
    --dev                    Dev mode — proxy to Vite dev server for HMR
    --no-open                Don't auto-open browser

axon ui                      Launch the web UI (default: localhost:8420)
    --port / -p PORT         Port to serve on (default: 8420)
    --watch / -w             Enable live file watching with auto-reindex
    --dev                    Dev mode — proxy to Vite dev server for HMR
    --no-open                Don't auto-open browser
    --direct                 Force standalone mode even if a shared host exists

axon setup                   Print MCP configuration JSON
    --claude                 For Claude Code
    --cursor                 For Cursor

axon mcp                     Start the MCP server (stdio transport)
axon serve                   Start the MCP server
    --watch, -w              Enable live file watching with auto-reindex
axon --version               Print version
```

---

## Example Workflows

### "I need to refactor the User class — what breaks?"

```bash
# See everything connected to User
axon context User

# Check blast radius — grouped by depth
axon impact User --depth 3

# Find files that always change with user.py
axon cypher "MATCH (a:File)-[r:CodeRelation]->(b:File) WHERE a.name = 'user.py' AND r.rel_type = 'coupled_with' RETURN b.name, r.strength ORDER BY r.strength DESC"
```

### "Is there dead code we should clean up?"

```bash
axon dead-code
```

### "What are the main execution flows?"

```bash
axon cypher "MATCH (p:Process) RETURN p.name, p.properties ORDER BY p.name"
```

### "Which parts of the codebase are most tightly coupled?"

```bash
axon cypher "MATCH (a:File)-[r:CodeRelation]->(b:File) WHERE r.rel_type = 'coupled_with' RETURN a.name, b.name, r.strength ORDER BY r.strength DESC LIMIT 20"
```

---

## Knowledge Graph Model

### Nodes

| Label | Description |
|-------|-------------|
| `File` | Source file |
| `Folder` | Directory |
| `Function` | Top-level function |
| `Class` | Class definition |
| `Method` | Method within a class |
| `Interface` | Interface / Protocol definition |
| `TypeAlias` | Type alias |
| `Enum` | Enumeration |
| `Community` | Auto-detected functional cluster |
| `Process` | Detected execution flow |

### Relationships

| Type | Description | Key Properties |
|------|-------------|----------------|
| `CONTAINS` | Folder -> File/Symbol hierarchy | -- |
| `DEFINES` | File -> Symbol it defines | -- |
| `CALLS` | Symbol -> Symbol it calls | `confidence` (0.0-1.0) |
| `IMPORTS` | File -> File it imports from | `symbols` (names list) |
| `EXTENDS` | Class -> Class it extends | -- |
| `IMPLEMENTS` | Class -> Interface it implements | -- |
| `USES_TYPE` | Symbol -> Type it references | `role` (param/return/variable) |
| `EXPORTS` | File -> Symbol it exports | -- |
| `MEMBER_OF` | Symbol -> Community it belongs to | -- |
| `STEP_IN_PROCESS` | Symbol -> Process it participates in | `step_number` |
| `COUPLED_WITH` | File -> File that co-changes with it | `strength`, `co_changes` |

### Node ID Format

```
{label}:{relative_path}:{symbol_name}

Examples:
  function:src/auth/validate.py:validate_user
  class:src/models/user.py:User
  method:src/models/user.py:User.save
```

---

## Architecture

```
Source Code (.py, .ts, .js, .tsx, .jsx)
    |
    v
+----------------------------------------------+
|         Ingestion Pipeline (12 phases)        |
|                                               |
|  walk -> structure -> parse -> imports        |
|  -> calls -> heritage -> types                |
|  -> communities -> processes -> dead_code     |
|  -> coupling -> embeddings                    |
+----------------------+-----------------------+
                       |
                       v
              +-----------------+
              | KnowledgeGraph  |  (in-memory during build)
              +--------+--------+
                       |
          +------------+------------+
          v            v            v
     +---------+ +---------+ +---------+
     | KuzuDB  | |  FTS    | | Vector  |
     | (graph) | | (BM25)  | | (HNSW)  |
     +----+----+ +----+----+ +----+----+
          +------------+------------+
                       |
              StorageBackend Protocol
                       |
           +-----------+-----------+
           v           v           v
     +----------+ +----------+ +----------+
     |   MCP    | |  Web UI  | |   CLI    |
     |  Server  | | (FastAPI | | (Typer)  |
     | (stdio)  | |  + React)| |          |
     +----+-----+ +----+-----+ +----+-----+
          |             |            |
     Claude Code    Browser      Terminal
     / Cursor      (developer)  (developer)
```

### Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Parsing | tree-sitter | Language-agnostic AST extraction |
| Graph Storage | KuzuDB | Embedded graph database with Cypher, FTS, and vector support |
| Graph Algorithms | igraph + leidenalg | Leiden community detection |
| Embeddings | fastembed | ONNX-based 384-dim vectors (~100MB, no PyTorch) |
| MCP Protocol | mcp SDK (FastMCP) | AI agent communication via stdio |
| Web Backend | FastAPI + Uvicorn | REST API for the web UI, SSE for live updates |
| Web Frontend | React + TypeScript + Vite | Interactive dashboard with Tailwind CSS |
| Graph Visualization | Sigma.js + Graphology | WebGL graph rendering with ForceAtlas2 layout |
| CLI | Typer + Rich | Terminal interface with progress bars |
| File Watching | watchfiles | Rust-based file system watcher |
| Gitignore | pathspec | Full `.gitignore` pattern matching |

### Storage

Everything lives locally:

```
your-project/
+-- .axon/
    +-- kuzu/          # KuzuDB graph database (graph + FTS + vectors)
    +-- meta.json      # Index metadata and stats
```

Add `.axon/` to your `.gitignore`.

A global registry at `~/.axon/repos/` is automatically populated on `axon analyze`, enabling `axon list` to discover all indexed repositories across your machine.

The storage layer is abstracted behind a `StorageBackend` Protocol — KuzuDB is the default, with an optional Neo4j backend available via `pip install axoniq[neo4j]`.

---

## Development

```bash
git clone https://github.com/harshkedia177/axon.git
cd axon
uv sync --all-extras

# Run tests
uv run pytest

# Lint
uv run ruff check src/

# Run from source
uv run axon --help

# Frontend development (React + Vite with HMR)
cd src/axon/web/frontend
npm install
npm run dev                  # Vite dev server on :5173
# In another terminal:
uv run axon ui --dev         # Backend on :8420, proxies to Vite
```

---

## License

MIT

---

Built by [@harshkedia177](https://github.com/harshkedia177)
