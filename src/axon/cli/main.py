"""Axon CLI — Graph-powered code intelligence engine."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import anyio
import typer
import uvicorn
from mcp.client.streamable_http import streamablehttp_client
from mcp.server.stdio import stdio_server
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from axon import __version__
from axon.core.diff import diff_branches, format_diff
from axon.core.embeddings.embedder import (
    _DEFAULT_MODEL,
    configure_cuda,
    validate_cuda,
)
from axon.core.ingestion.pipeline import PipelineResult, run_pipeline
from axon.core.storage.base import EMBEDDING_DIMENSIONS
from axon.core.ingestion.watcher import ensure_current_embeddings, watch_repo
from axon.core.storage.kuzu_backend import KuzuBackend
from axon.mcp import tools as mcp_tools
from axon.mcp.server import main as mcp_main
from axon.mcp.server import set_lock, set_storage
from axon.runtime import AxonRuntime
from axon.web import app as web_app_module

console = Console()
logger = logging.getLogger(__name__)
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8420
DEFAULT_MANAGED_PORT = 8421
UPDATE_CHECK_INTERVAL_SECONDS = 60 * 60 * 24
UPDATE_CHECK_URL = "https://pypi.org/pypi/axoniq/json"
UPDATE_CHECK_SKIP_COMMANDS = {"mcp", "serve", "host"}


def _load_storage(repo_path: Path | None = None) -> "KuzuBackend":  # noqa: F821
    target = (repo_path or Path.cwd()).resolve()
    db_path = target / ".axon" / "kuzu"
    if not db_path.exists():
        console.print(
            f"[red]Error:[/red] No index found at {target}. Run 'axon analyze' first."
        )
        raise typer.Exit(code=1)

    storage = KuzuBackend()
    storage.initialize(db_path, read_only=True)
    return storage


def _has_index_metadata(axon_dir: Path) -> bool:
    return (axon_dir / "meta.json").exists()


def _has_index_database(db_path: Path) -> bool:
    return db_path.is_dir() and any(db_path.iterdir())


def _has_existing_index(axon_dir: Path, db_path: Path) -> bool:
    return _has_index_metadata(axon_dir) and _has_index_database(db_path)


def _update_cache_path() -> Path:
    return Path.home() / ".axon" / "update-check.json"


def _parse_version_parts(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for raw_part in version.split("."):
        digits = "".join(ch for ch in raw_part if ch.isdigit())
        parts.append(int(digits or 0))
    return tuple(parts)


def _is_newer_version(candidate: str, current: str) -> bool:
    return _parse_version_parts(candidate) > _parse_version_parts(current)


def _read_update_cache() -> dict | None:
    cache_path = _update_cache_path()
    if not cache_path.exists():
        return None
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _write_update_cache(payload: dict) -> None:
    cache_path = _update_cache_path()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _fetch_latest_version() -> str | None:
    try:
        with urllib.request.urlopen(UPDATE_CHECK_URL, timeout=1.5) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return str(payload["info"]["version"])
    except (KeyError, OSError, ValueError, urllib.error.URLError):
        return None


def _get_latest_version() -> str | None:
    now = int(time.time())
    cache = _read_update_cache()
    if cache is not None:
        checked_at = int(cache.get("checked_at", 0))
        latest = cache.get("latest_version")
        if latest and now - checked_at < UPDATE_CHECK_INTERVAL_SECONDS:
            return str(latest)

    latest = _fetch_latest_version()
    if latest is not None:
        _write_update_cache({"checked_at": now, "latest_version": latest})
    return latest


def _maybe_notify_update(invoked_subcommand: str | None) -> None:
    if invoked_subcommand in UPDATE_CHECK_SKIP_COMMANDS:
        return
    latest = _get_latest_version()
    if latest and _is_newer_version(latest, __version__):
        console.print(
            f"[yellow]Update available:[/yellow] Axon {latest} "
            f"(current {__version__}). Run `pip install -U axoniq`."
        )


def _register_in_global_registry(meta: dict, repo_path: Path) -> None:
    """Write meta.json into ``~/.axon/repos/{slug}/`` for multi-repo discovery.

    Slug is ``{repo_name}`` if that slot is unclaimed or already belongs to
    this repo.  Falls back to ``{repo_name}-{sha256(path)[:8]}`` on collision.
    """
    registry_root = Path.home() / ".axon" / "repos"
    repo_name = repo_path.name

    candidate = registry_root / repo_name
    slug = repo_name
    if candidate.exists():
        existing_meta_path = candidate / "meta.json"
        try:
            existing = json.loads(existing_meta_path.read_text())
            if existing.get("path") != str(repo_path):
                short_hash = hashlib.sha256(str(repo_path).encode()).hexdigest()[:8]
                slug = f"{repo_name}-{short_hash}"
        except (json.JSONDecodeError, OSError):
            shutil.rmtree(candidate, ignore_errors=True)  # Clean broken slot before claiming

    # Remove any stale entry for the same repo_path under a different slug.
    if registry_root.exists():
        for old_dir in registry_root.iterdir():
            if not old_dir.is_dir() or old_dir.name == slug:
                continue
            old_meta = old_dir / "meta.json"
            try:
                old_data = json.loads(old_meta.read_text())
                if old_data.get("path") == str(repo_path):
                    shutil.rmtree(old_dir, ignore_errors=True)
            except (json.JSONDecodeError, OSError):
                continue

    slot = registry_root / slug
    slot.mkdir(parents=True, exist_ok=True)

    registry_meta = dict(meta)
    registry_meta["slug"] = slug
    (slot / "meta.json").write_text(
        json.dumps(registry_meta, indent=2) + "\n", encoding="utf-8"
    )


def _build_meta(result: "PipelineResult", repo_path: Path) -> dict:  # noqa: F821
    return {
        "version": __version__,
        "name": repo_path.name,
        "path": str(repo_path),
        "embedding_model": _DEFAULT_MODEL,
        "embedding_dimensions": EMBEDDING_DIMENSIONS,
        "stats": {
            "files": result.files,
            "symbols": result.symbols,
            "relationships": result.relationships,
            "clusters": result.clusters,
            "flows": result.processes,
            "dead_code": result.dead_code,
            "coupled_pairs": result.coupled_pairs,
            "embeddings": result.embeddings,
        },
        "last_indexed_at": datetime.now(tz=timezone.utc).isoformat(),
    }


def _host_meta_path(repo_path: Path) -> Path:
    return repo_path / ".axon" / "host.json"


def _host_lease_dir(repo_path: Path) -> Path:
    return repo_path / ".axon" / "host-leases"


def _display_host(host: str) -> str:
    return "127.0.0.1" if host in {"0.0.0.0", "::"} else host


def _build_host_urls(host: str, port: int) -> tuple[str, str]:
    base = f"http://{_display_host(host)}:{port}"
    return base, f"{base}/mcp"


def _read_host_meta(repo_path: Path) -> dict | None:
    meta_path = _host_meta_path(repo_path)
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _write_host_meta(
    repo_path: Path,
    host_url: str,
    mcp_url: str,
    port: int,
    *,
    ui_enabled: bool,
) -> None:
    meta_path = _host_meta_path(repo_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pid": os.getpid(),
        "repo_path": str(repo_path),
        "host_url": host_url,
        "mcp_url": mcp_url,
        "port": port,
        "ui_enabled": ui_enabled,
        "leases_dir": str(_host_lease_dir(repo_path)),
    }
    meta_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _clear_host_meta(repo_path: Path) -> None:
    meta_path = _host_meta_path(repo_path)
    if meta_path.exists():
        meta_path.unlink(missing_ok=True)


def _create_host_lease(repo_path: Path, lease_type: str) -> Path:
    lease_dir = _host_lease_dir(repo_path)
    lease_dir.mkdir(parents=True, exist_ok=True)
    lease_path = lease_dir / f"{os.getpid()}-{uuid.uuid4().hex}.json"
    payload = {
        "pid": os.getpid(),
        "type": lease_type,
        "created_at": time.time(),
    }
    lease_path.write_text(json.dumps(payload), encoding="utf-8")
    return lease_path


def _remove_host_lease(lease_path: Path | None) -> None:
    if lease_path is not None:
        lease_path.unlink(missing_ok=True)


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _count_live_host_leases(repo_path: Path) -> int:
    lease_dir = _host_lease_dir(repo_path)
    if not lease_dir.exists():
        return 0
    live = 0
    for lease_path in lease_dir.glob("*.json"):
        try:
            payload = json.loads(lease_path.read_text(encoding="utf-8"))
            pid = int(payload.get("pid", 0))
        except (ValueError, json.JSONDecodeError, OSError):
            lease_path.unlink(missing_ok=True)
            continue
        if _pid_is_alive(pid):
            live += 1
        else:
            lease_path.unlink(missing_ok=True)
    return live


def _is_host_alive(meta: dict, repo_path: Path) -> bool:
    host_url = meta.get("host_url")
    if not host_url:
        return False
    try:
        with urllib.request.urlopen(f"{host_url}/api/host", timeout=1.0) as response:
            if response.status != 200:
                return False
            payload = json.loads(response.read().decode("utf-8"))
            return payload.get("repoPath") == str(repo_path)
    except (OSError, ValueError, urllib.error.URLError):
        return False


def _get_live_host_info(repo_path: Path) -> dict | None:
    meta = _read_host_meta(repo_path)
    if meta is None:
        return None
    if _is_host_alive(meta, repo_path):
        return meta
    return None


def _start_host_background(
    repo_path: Path,
    *,
    port: int = DEFAULT_PORT,
    bind: str = DEFAULT_HOST,
    watch: bool = True,
    managed: bool = False,
) -> None:
    """Start a detached shared host process in the background."""
    command = [
        sys.executable,
        "-m",
        "axon.cli.main",
        "host",
        "--port",
        str(port),
        "--bind",
        bind,
        "--no-open",
    ]
    if watch:
        command.append("--watch")
    else:
        command.append("--no-watch")
    if managed:
        command.append("--managed")
    with open(os.devnull, "wb") as devnull:
        subprocess.Popen(  # noqa: S603
            command,
            cwd=repo_path,
            stdout=devnull,
            stderr=devnull,
            start_new_session=True,
        )


def _ensure_host_running(
    repo_path: Path,
    *,
    port: int = DEFAULT_PORT,
    bind: str = DEFAULT_HOST,
    watch: bool = True,
    timeout_seconds: float = 10.0,
    managed: bool = False,
) -> dict:
    """Return live host metadata, starting the shared host if necessary."""
    live_host = _get_live_host_info(repo_path)
    if live_host is not None:
        return live_host

    _start_host_background(repo_path, port=port, bind=bind, watch=watch, managed=managed)
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        live_host = _get_live_host_info(repo_path)
        if live_host is not None:
            return live_host
        time.sleep(0.2)
    raise RuntimeError("Timed out waiting for Axon host to start.")


app = typer.Typer(
    name="axon",
    help="Axon — Graph-powered code intelligence engine.",
    no_args_is_help=True,
)

def _version_callback(value: bool) -> None:
    if value:
        console.print(f"Axon v{__version__}")
        raise typer.Exit()

@app.callback()
def main(
    ctx: typer.Context,
    version: Optional[bool] = typer.Option(  # noqa: N803
        None,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """Axon — Graph-powered code intelligence engine."""
    _maybe_notify_update(ctx.invoked_subcommand)


def _initialize_writable_storage(
    repo_path: Path, *, auto_index: bool = True,
) -> tuple["KuzuBackend", Path, Path]:  # noqa: F821
    """Open the repo database in read-write mode.

    If *auto_index* is False and no index exists, raises typer.Exit instead of
    running the pipeline — callers like ``ui`` should tell the user to run
    ``axon analyze .`` themselves.
    """
    axon_dir = repo_path / ".axon"
    db_path = axon_dir / "kuzu"

    if not auto_index and not _has_existing_index(axon_dir, db_path):
        console.print(
            "[red]Error:[/red] No index found. "
            "Run [cyan]axon analyze .[/cyan] first to index this codebase."
        )
        raise typer.Exit(code=1)

    axon_dir.mkdir(parents=True, exist_ok=True)

    storage = KuzuBackend()
    storage.initialize(db_path)

    if not _has_index_metadata(axon_dir):
        console.print("[bold]Running initial index...[/bold]")
        _, result = run_pipeline(repo_path, storage)
        meta = _build_meta(result, repo_path)
        meta_path = axon_dir / "meta.json"
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        try:
            _register_in_global_registry(meta, repo_path)
        except Exception:
            logger.debug("Failed to register repo in global registry", exc_info=True)
    else:
        ensure_current_embeddings(storage, repo_path)

    return storage, axon_dir, db_path


async def _proxy_stdio_to_http_mcp(mcp_url: str) -> None:
    """Bridge a local stdio MCP session to the shared HTTP MCP host."""
    async with stdio_server() as (local_read, local_write):
        async with streamablehttp_client(mcp_url) as (remote_read, remote_write, _):
            async def _forward(reader, writer) -> None:
                async with writer:
                    async for message in reader:
                        await writer.send(message)

            async with anyio.create_task_group() as tg:
                tg.start_soon(_forward, local_read, remote_write)
                tg.start_soon(_forward, remote_read, local_write)


def _run_shared_host(
    *,
    port: int,
    bind: str,
    no_open: bool,
    watch: bool,
    dev: bool,
    managed: bool,
    open_browser: bool,
    announce_ui: bool,
    announce_mcp: bool,
    expose_ui: bool,
    already_running_message: str,
    auto_index: bool = True,
) -> None:
    """Run the shared Axon host with configurable UX messaging."""
    repo_path = Path.cwd().resolve()
    live_host = _get_live_host_info(repo_path)
    if live_host is not None:
        console.print(already_running_message.format(url=live_host["host_url"]))
        if open_browser and not no_open:
            webbrowser.open(live_host["host_url"])
        return

    storage, _, db_path = _initialize_writable_storage(repo_path, auto_index=auto_index)
    host_url, mcp_url = _build_host_urls(bind, port)
    lock = asyncio.Lock()
    runtime = AxonRuntime(
        storage=storage,
        repo_path=repo_path,
        watch=watch,
        lock=lock,
        host_url=host_url,
        mcp_url=mcp_url,
        owns_storage=True,
    )
    set_storage(storage)
    set_lock(lock)

    web_app = web_app_module.create_app(
        db_path=db_path,
        repo_path=repo_path,
        watch=watch,
        dev=dev,
        runtime=runtime,
        mount_mcp=True,
        host_url=host_url,
        mcp_url=mcp_url,
        mount_frontend=expose_ui,
    )

    if open_browser and not no_open:
        threading.Timer(1.0, lambda: webbrowser.open(host_url)).start()

    if announce_ui:
        console.print(f"[bold green]Axon UI[/bold green] running at {host_url}")
    if announce_mcp:
        console.print(f"[dim]HTTP MCP endpoint:[/dim] {mcp_url}")
    if watch:
        console.print("[dim]File watching enabled[/dim]")
    if dev:
        console.print("[dim]Dev mode — proxying to Vite on :5173[/dim]")

    _write_host_meta(repo_path, host_url, mcp_url, port, ui_enabled=expose_ui)

    async def _run() -> None:
        config = uvicorn.Config(
            web_app,
            host=bind,
            port=port,
            log_level="warning",
        )
        server = uvicorn.Server(config)
        stop = asyncio.Event()

        async def _serve() -> None:
            await server.serve()
            stop.set()

        async def _managed_shutdown() -> None:
            if not managed:
                return
            idle_started_at: float | None = None
            while not stop.is_set():
                live_leases = _count_live_host_leases(repo_path)
                if live_leases == 0:
                    if idle_started_at is None:
                        idle_started_at = time.time()
                    elif time.time() - idle_started_at >= 2.0:
                        server.should_exit = True
                        stop.set()
                        return
                else:
                    idle_started_at = None
                await asyncio.sleep(0.5)

        tasks = [_serve()]
        if watch:
            tasks.append(watch_repo(repo_path, storage, stop_event=stop, lock=lock))
        if managed:
            tasks.append(_managed_shutdown())
        await asyncio.gather(*tasks)

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass
    finally:
        _clear_host_meta(repo_path)
        storage.close()

def _run_background_embeddings(
    graph: "KnowledgeGraph",
    db_path: Path,
    meta_path: Path,
    repo_path: Path,
) -> None:
    """Generate embeddings in a background thread with its own storage connection."""
    from axon.core.ingestion.pipeline import _run_embedding_phase, PipelineResult

    bg_storage = KuzuBackend()
    bg_storage.initialize(db_path)
    try:
        bg_result = PipelineResult()
        _run_embedding_phase(graph, bg_storage, bg_result, lambda _phase, _pct: None)

        # Update meta.json with embedding count.
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["stats"]["embeddings"] = bg_result.embeddings
            meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        except Exception:
            logger.debug("Failed to update meta.json with embedding count", exc_info=True)

        if bg_result.embeddings > 0:
            console.print(
                f"[dim]Background embeddings complete: "
                f"{bg_result.embeddings} vectors generated.[/dim]"
            )
    except Exception:
        logger.warning("Background embedding failed — semantic search unavailable", exc_info=True)
    finally:
        bg_storage.close()


def _configure_and_validate_cuda(cuda_flag: bool) -> None:
    """Configure CUDA from --cuda flag and validate before pipeline runs."""
    if cuda_flag:
        configure_cuda(True)
    try:
        validate_cuda()
    except RuntimeError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)


@app.command()
def analyze(
    path: Path = typer.Argument(Path("."), help="Path to the repository to index."),
    no_embeddings: bool = typer.Option(
        False, "--no-embeddings", help="Skip vector embedding generation."
    ),
    foreground_embeddings: bool = typer.Option(
        False,
        "--foreground-embeddings",
        help="Generate embeddings synchronously instead of in the background.",
    ),
    cuda: bool = typer.Option(
        False, "--cuda", help="Use CUDA GPU acceleration for embedding generation."
    ),
) -> None:
    """Index a repository into a knowledge graph."""
    repo_path = path.resolve()
    if not repo_path.is_dir():
        console.print(f"[red]Error:[/red] {repo_path} is not a directory.")
        raise typer.Exit(code=1)

    console.print(f"[bold]Indexing[/bold] {repo_path}")

    axon_dir = repo_path / ".axon"
    axon_dir.mkdir(parents=True, exist_ok=True)
    db_path = axon_dir / "kuzu"

    storage = KuzuBackend()
    storage.initialize(db_path)

    # Run pipeline: skip embeddings here if we'll do them in the background.
    run_embeddings_inline = foreground_embeddings and not no_embeddings
    _configure_and_validate_cuda(cuda)

    result: PipelineResult | None = None
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Starting...", total=None)

        def on_progress(phase: str, pct: float) -> None:
            progress.update(task, description=f"{phase} ({pct:.0%})")

        graph, result = run_pipeline(
            repo_path=repo_path,
            storage=storage,
            progress_callback=on_progress,
            embeddings=run_embeddings_inline,
        )

    meta = _build_meta(result, repo_path)
    meta_path = axon_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    try:
        _register_in_global_registry(meta, repo_path)
    except Exception:
        logger.debug("Failed to register repo in global registry", exc_info=True)

    storage.close()

    # Launch background embedding thread if needed.
    if not no_embeddings and not run_embeddings_inline:
        embed_thread = threading.Thread(
            target=_run_background_embeddings,
            args=(graph, db_path, meta_path, repo_path),
            daemon=True,
        )
        embed_thread.start()

    console.print()
    console.print("[bold green]Indexing complete.[/bold green]")
    console.print(f"  Files:          {result.files}")
    console.print(f"  Symbols:        {result.symbols}")
    console.print(f"  Relationships:  {result.relationships}")
    if result.clusters > 0:
        console.print(f"  Clusters:       {result.clusters}")
    if result.processes > 0:
        console.print(f"  Flows:          {result.processes}")
    if result.dead_code > 0:
        console.print(f"  Dead code:      {result.dead_code}")
    if result.coupled_pairs > 0:
        console.print(f"  Coupled pairs:  {result.coupled_pairs}")
    if run_embeddings_inline and result.embeddings > 0:
        console.print(f"  Embeddings:     {result.embeddings}")
    elif not no_embeddings and not run_embeddings_inline:
        console.print("  Embeddings:     [dim]generating in background...[/dim]")
    console.print(f"  Duration:       {result.duration_seconds:.2f}s")

    # Wait for background embeddings to finish before exiting.
    if not no_embeddings and not run_embeddings_inline:
        embed_thread.join()

@app.command()
def status() -> None:
    """Show index status for current repository."""
    repo_path = Path.cwd().resolve()
    meta_path = repo_path / ".axon" / "meta.json"

    if not meta_path.exists():
        console.print(
            "[red]Error:[/red] No index found. "
            "Run [cyan]axon analyze .[/cyan] first to index this codebase."
        )
        raise typer.Exit(code=1)

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    stats = meta.get("stats", {})

    console.print(f"[bold]Index status for[/bold] {repo_path}")
    console.print(f"  Version:        {meta.get('version', '?')}")
    console.print(f"  Last indexed:   {meta.get('last_indexed_at', '?')}")
    console.print(f"  Files:          {stats.get('files', '?')}")
    console.print(f"  Symbols:        {stats.get('symbols', '?')}")
    console.print(f"  Relationships:  {stats.get('relationships', '?')}")

    if stats.get("clusters", 0) > 0:
        console.print(f"  Clusters:       {stats['clusters']}")
    if stats.get("flows", 0) > 0:
        console.print(f"  Flows:          {stats['flows']}")
    if stats.get("dead_code", 0) > 0:
        console.print(f"  Dead code:      {stats['dead_code']}")
    if stats.get("coupled_pairs", 0) > 0:
        console.print(f"  Coupled pairs:  {stats['coupled_pairs']}")

@app.command(name="list")
def list_repos() -> None:
    """List all indexed repositories."""
    result = mcp_tools.handle_list_repos()
    console.print(result)

@app.command()
def clean(
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation prompt."),
) -> None:
    """Delete index for current repository."""
    repo_path = Path.cwd().resolve()
    axon_dir = repo_path / ".axon"

    if not axon_dir.exists():
        console.print(
            f"[red]Error:[/red] No index found at {repo_path}. Nothing to clean."
        )
        raise typer.Exit(code=1)

    if not force:
        confirm = typer.confirm(f"Delete index at {axon_dir}?")
        if not confirm:
            console.print("Aborted.")
            raise typer.Exit()

    shutil.rmtree(axon_dir)
    console.print(f"[green]Deleted[/green] {axon_dir}")

@app.command()
def query(
    q: str = typer.Argument(..., help="Search query for the knowledge graph."),
    limit: int = typer.Option(20, "--limit", "-n", help="Maximum number of results."),
) -> None:
    """Search the knowledge graph."""
    storage = _load_storage()
    result = mcp_tools.handle_query(storage, q, limit=limit)
    console.print(result)
    storage.close()

@app.command()
def context(
    name: str = typer.Argument(..., help="Symbol name to inspect."),
) -> None:
    """Show 360-degree view of a symbol."""
    storage = _load_storage()
    result = mcp_tools.handle_context(storage, name)
    console.print(result)
    storage.close()

@app.command()
def impact(
    target: str = typer.Argument(..., help="Symbol to analyze blast radius for."),
    depth: int = typer.Option(3, "--depth", "-d", min=1, max=10, help="Traversal depth (1-10)."),
) -> None:
    """Show blast radius of changing a symbol."""
    storage = _load_storage()
    result = mcp_tools.handle_impact(storage, target, depth=depth)
    console.print(result)
    storage.close()

@app.command(name="dead-code")
def dead_code() -> None:
    """List all detected dead code."""
    storage = _load_storage()
    result = mcp_tools.handle_dead_code(storage)
    console.print(result)
    storage.close()

@app.command()
def cypher(
    query: str = typer.Argument(..., help="Raw Cypher query to execute."),
) -> None:
    """Execute raw Cypher against the knowledge graph."""
    storage = _load_storage()
    result = mcp_tools.handle_cypher(storage, query)
    console.print(result)
    storage.close()

@app.command()
def setup(
    claude: bool = typer.Option(False, "--claude", help="Configure MCP for Claude Code."),
    cursor: bool = typer.Option(False, "--cursor", help="Configure MCP for Cursor."),
) -> None:
    """Configure MCP for Claude Code / Cursor."""
    stdio_config = {
        "command": "axon",
        "args": ["serve", "--watch"],
    }

    if claude or (not claude and not cursor):
        console.print("[bold]Claude Code[/bold]")
        console.print("Add to your [cyan].mcp.json[/cyan] or [cyan]~/.claude.json[/cyan]:\n")
        console.print(json.dumps({"mcpServers": {"axon": stdio_config}}, indent=2))
        console.print("\nOr run directly:")
        console.print("[cyan]claude mcp add axon -- axon serve --watch[/cyan]")

    if cursor or (not claude and not cursor):
        console.print("[bold]Cursor[/bold]")
        console.print("Add to your MCP config:\n")
        console.print(json.dumps({"axon": stdio_config}, indent=2))

    console.print("\n[dim]Then index your codebase with:[/dim] [cyan]axon analyze .[/cyan]")

@app.command()
def watch(
    cuda: bool = typer.Option(
        False, "--cuda", help="Use CUDA GPU acceleration for embedding generation."
    ),
) -> None:
    """Watch mode — re-index on file changes."""
    _configure_and_validate_cuda(cuda)
    repo_path = Path.cwd().resolve()
    axon_dir = repo_path / ".axon"
    axon_dir.mkdir(parents=True, exist_ok=True)
    db_path = axon_dir / "kuzu"

    storage = KuzuBackend()
    storage.initialize(db_path)

    if not (axon_dir / "meta.json").exists():
        console.print("[bold]Running initial index...[/bold]")
        _, result = run_pipeline(repo_path, storage)
        meta = _build_meta(result, repo_path)
        meta_path = axon_dir / "meta.json"
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        try:
            _register_in_global_registry(meta, repo_path)
        except Exception:
            logger.debug("Failed to register repo in global registry", exc_info=True)
    else:
        ensure_current_embeddings(storage, repo_path)

    console.print(f"[bold]Watching[/bold] {repo_path} for changes (Ctrl+C to stop)")

    try:
        asyncio.run(watch_repo(repo_path, storage))
    except KeyboardInterrupt:
        console.print("\n[bold]Watch stopped.[/bold]")
    finally:
        storage.close()

@app.command()
def diff(
    branch_range: str = typer.Argument(
        ..., help="Branch range for comparison (e.g. main..feature)."
    ),
) -> None:
    """Structural branch comparison."""
    repo_path = Path.cwd().resolve()
    try:
        result = diff_branches(repo_path, branch_range)
    except (ValueError, RuntimeError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(format_diff(result))

@app.command()
def mcp() -> None:
    """Start MCP server (stdio transport)."""
    asyncio.run(mcp_main())


@app.command()
def host(
    port: int = typer.Option(
        DEFAULT_PORT, "--port", "-p", help="Port to serve UI and HTTP MCP on."
    ),
    bind: str = typer.Option(
        DEFAULT_HOST, "--bind", help="Host interface to bind the shared host to."
    ),
    no_open: bool = typer.Option(False, "--no-open", help="Don't auto-open browser."),
    watch: bool = typer.Option(
        True, "--watch/--no-watch", help="Enable file watching with auto-reindex."
    ),
    dev: bool = typer.Option(False, "--dev", help="Proxy to Vite dev server for HMR."),
    managed: bool = typer.Option(False, "--managed", hidden=True),
    cuda: bool = typer.Option(
        False, "--cuda", help="Use CUDA GPU acceleration for embedding generation."
    ),
) -> None:
    """Run the shared Axon host for UI and multi-session HTTP MCP clients."""
    _configure_and_validate_cuda(cuda)
    _run_shared_host(
        port=port,
        bind=bind,
        no_open=no_open,
        watch=watch,
        dev=dev,
        managed=managed,
        open_browser=True,
        announce_ui=True,
        announce_mcp=True,
        expose_ui=not managed,
        already_running_message="[yellow]Axon host already running[/yellow] at {url}",
    )

@app.command()
def serve(
    watch: bool = typer.Option(
        False, "--watch", "-w", help="Enable file watching with auto-reindex."
    ),
    cuda: bool = typer.Option(
        False, "--cuda", help="Use CUDA GPU acceleration for embedding generation."
    ),
) -> None:
    """Start MCP server, optionally with live file watching."""
    _configure_and_validate_cuda(cuda)
    if not watch:
        asyncio.run(mcp_main())
        return

    repo_path = Path.cwd().resolve()
    lease_path: Path | None = None
    try:
        live_host = _ensure_host_running(
            repo_path,
            port=DEFAULT_MANAGED_PORT,
            watch=True,
            managed=True,
        )
        lease_path = _create_host_lease(repo_path, "mcp")
    except RuntimeError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    try:
        asyncio.run(_proxy_stdio_to_http_mcp(live_host["mcp_url"]))
    finally:
        _remove_host_lease(lease_path)


@app.command()
def ui(
    port: int = typer.Option(8420, "--port", "-p", help="Port to serve on."),
    no_open: bool = typer.Option(False, "--no-open", help="Don't auto-open browser."),
    watch_files: bool = typer.Option(False, "--watch", "-w", help="Enable live file watching."),
    dev: bool = typer.Option(False, "--dev", help="Proxy to Vite dev server for HMR."),
    direct: bool = typer.Option(
        False,
        "--direct",
        help="Force standalone UI mode even if a shared Axon host is already running.",
    ),
) -> None:
    """Launch the Axon web UI."""
    repo_path = Path.cwd().resolve()
    if not direct:
        live_host = _get_live_host_info(repo_path)
        if live_host is not None:
            if live_host.get("ui_enabled", True):
                console.print(
                    f"[bold green]Axon UI[/bold green] available at {live_host['host_url']}"
                )
                if not no_open:
                    webbrowser.open(live_host["host_url"])
                return

            proxy_app = web_app_module.create_ui_proxy_app(live_host["host_url"], dev=dev)
            console.print(
                f"[bold green]Axon UI[/bold green] running at http://{DEFAULT_HOST}:{port}"
            )
            if not no_open:
                webbrowser.open(f"http://{DEFAULT_HOST}:{port}")
            uvicorn.run(proxy_app, host=DEFAULT_HOST, port=port, log_level="warning")
            return

        _run_shared_host(
            port=port,
            bind=DEFAULT_HOST,
            no_open=no_open,
            watch=watch_files,
            dev=dev,
            managed=False,
            open_browser=True,
            announce_ui=True,
            announce_mcp=False,
            expose_ui=True,
            already_running_message="[bold green]Axon UI[/bold green] available at {url}",
            auto_index=False,
        )
        return

    storage, _, db_path = _initialize_writable_storage(repo_path, auto_index=False)
    runtime = AxonRuntime(
        storage=storage,
        repo_path=repo_path,
        watch=watch_files,
        owns_storage=True,
    )

    web_app = web_app_module.create_app(
        db_path=db_path,
        repo_path=repo_path,
        watch=watch_files,
        dev=dev,
        runtime=runtime,
    )

    if not no_open:
        url = f"http://localhost:{port}"
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    console.print(f"[bold green]Axon UI[/bold green] running at http://localhost:{port}")
    if watch_files:
        console.print("[dim]File watching enabled — graph updates on save[/dim]")
    if dev:
        console.print("[dim]Dev mode — proxying to Vite on :5173[/dim]")

    if watch_files:
        async def _run() -> None:
            config = uvicorn.Config(
                web_app, host="127.0.0.1", port=port, log_level="warning"
            )
            server = uvicorn.Server(config)
            stop = asyncio.Event()

            async def _serve() -> None:
                await server.serve()
                stop.set()

            await asyncio.gather(
                _serve(),
                watch_repo(repo_path, web_app.state.storage, stop_event=stop),
            )

        try:
            asyncio.run(_run())
        except KeyboardInterrupt:
            console.print("\n[bold]UI stopped.[/bold]")
    else:
        uvicorn.run(web_app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    app()
