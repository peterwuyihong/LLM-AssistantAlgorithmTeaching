"""
FastAPI backend for the LLM-Assisted Algorithm Teaching web interface.

Provides:
- GET  /api/problems          — list all solved problems
- GET  /api/problems/{id}     — get a specific problem's details + code
- POST /api/solve             — submit a new problem to solve
- POST /api/solve-apps/{id}   — solve an APPS dataset problem
- GET  /api/apps-list         — list available APPS problems
- WS   /ws/solve/{task_id}    — WebSocket for real-time solve progress
"""

import asyncio
import json
import os
import shutil
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

from src.graph.builder import invoke_solve
from src.config import APPS_TRAIN_DIR, APPS_TEST_DIR, MAX_RETRIES, NUM_GENERATED_TESTS
from src.config import get_llm_config, set_llm_config
from src.converters.apps_converter import convert_apps_problem
from src.converters.cf_scraper import fetch_problemset, import_cf_problem, set_cache_dir
from src.llm.translate import translate_problem_statement

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "user_problems")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- In-memory task tracking for WebSocket progress ---
tasks: dict[str, dict] = {}
# Event loop reference for thread-safe event creation
_loop: Optional[asyncio.AbstractEventLoop] = None


def _new_task_entry(problem_id: str, folder: str) -> dict:
    """Create a new task entry with an asyncio.Event for WebSocket push."""
    return {
        "problem_id": problem_id,
        "folder": folder,
        "status": "running",
        "stage": "init",
        "log": [],
        "result": None,
        "update_event": asyncio.Event() if _loop and _loop.is_running() else None,
    }


@asynccontextmanager
async def lifespan(app):
    global _loop
    _loop = asyncio.get_event_loop()
    # Set CF scraper cache directory
    set_cache_dir(os.path.join(PROJECT_ROOT, ".cf_cache"))
    yield


app = FastAPI(title="LAAT - Algorithm Teaching", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ───────────────────────── Models ─────────────────────────

class SolveRequest(BaseModel):
    question: str
    input_format: str = ""
    output_format: str = ""
    data_range: str = ""
    test_inputs: list[str] = []
    test_outputs: list[str] = []
    ignore_format: bool = False
    max_retries: int = MAX_RETRIES
    num_tests: int = NUM_GENERATED_TESTS


class SolveAppsRequest(BaseModel):
    problem_id: str
    split: str = "train"
    ignore_format: bool = False
    max_retries: int = MAX_RETRIES
    force: bool = False


class LLMConfigRequest(BaseModel):
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


class CFImportRequest(BaseModel):
    contest_id: int
    index: str
    force: bool = False


class CFTranslateRequest(BaseModel):
    text: str


# ───────────────────────── Helpers ─────────────────────────

def _get_problem_folder(problem_id: str) -> Optional[str]:
    """Find a problem folder by ID across user_problems/, APPS_converted/, and cf_problems/."""
    # Check user_problems first
    user_dir = os.path.join(DATA_DIR, problem_id)
    if os.path.isdir(user_dir):
        return user_dir
    # Check cf_problems
    cf_dir = os.path.join(PROJECT_ROOT, "cf_problems", problem_id)
    if os.path.isdir(cf_dir):
        return cf_dir
    # Check APPS_converted
    for split in ("train", "test"):
        apps_dir = os.path.join(PROJECT_ROOT, "APPS_converted", split, problem_id)
        if os.path.isdir(apps_dir):
            return apps_dir
    return None


def _load_result(folder: str) -> Optional[dict]:
    """Load result.json from a problem folder."""
    path = os.path.join(folder, "result.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_code(folder: str, filename: str) -> Optional[str]:
    """Load a code file from a problem folder."""
    path = os.path.join(folder, filename)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _load_question(folder: str) -> Optional[dict]:
    """Load question.json from a problem folder."""
    path = os.path.join(folder, "question.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _list_solved_problems() -> list[dict]:
    """List all problems that have result.json."""
    results = []
    # User problems
    if os.path.isdir(DATA_DIR):
        for pid in sorted(os.listdir(DATA_DIR)):
            folder = os.path.join(DATA_DIR, pid)
            if not os.path.isdir(folder):
                continue
            r = _load_result(folder)
            if r:
                q = _load_question(folder) or {}
                results.append({
                    "id": pid,
                    "source": "user",
                    "question_title": q.get("question", "")[:60],
                    "result": r.get("result", "unknown"),
                    "timestamp": r.get("timestamp", ""),
                })
    # APPS converted
    for split in ("train", "test"):
        converted_dir = os.path.join(PROJECT_ROOT, "APPS_converted", split)
        if not os.path.isdir(converted_dir):
            continue
        for pid in sorted(os.listdir(converted_dir)):
            folder = os.path.join(converted_dir, pid)
            if not os.path.isdir(folder):
                continue
            r = _load_result(folder)
            if r:
                results.append({
                    "id": pid,
                    "source": f"apps-{split}",
                    "question_title": f"APPS {split}/{pid}",
                    "result": r.get("result", "unknown"),
                    "timestamp": r.get("timestamp", ""),
                })
    # Codeforces problems
    cf_dir = os.path.join(PROJECT_ROOT, "cf_problems")
    if os.path.isdir(cf_dir):
        for pid in sorted(os.listdir(cf_dir)):
            folder = os.path.join(cf_dir, pid)
            if not os.path.isdir(folder):
                continue
            r = _load_result(folder)
            if r:
                q = _load_question(folder) or {}
                name = q.get("name", "") or q.get("question", "")[:40]
                results.append({
                    "id": pid,
                    "source": "codeforces",
                    "question_title": f"CF {pid}: {name}",
                    "result": r.get("result", "unknown"),
                    "timestamp": r.get("timestamp", ""),
                })
    return results


# ───────────────────────── API Routes ─────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web UI."""
    html_path = os.path.join(PROJECT_ROOT, "web", "index.html")
    if os.path.isfile(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Web UI not found. Run with web/index.html in place.</h1>")


@app.get("/api/problems")
async def list_problems():
    """List all solved problems."""
    return _list_solved_problems()


@app.get("/api/problems/{problem_id}")
async def get_problem(problem_id: str):
    """Get full details for a problem: question, code, result, log."""
    folder = _get_problem_folder(problem_id)
    if not folder:
        raise HTTPException(status_code=404, detail=f"Problem {problem_id} not found")

    question = _load_question(folder) or {}
    result = _load_result(folder) or {}
    bf_code = _load_code(folder, "brute_force.py")
    dm_code = _load_code(folder, "data.py")
    sol_code = _load_code(folder, "solution.py")
    log_text = _load_code(folder, "execution_log.txt") or ""

    # Load test cases
    sample_cases = []
    sample_dir = os.path.join(folder, "sample")
    if os.path.isdir(sample_dir):
        in_files = sorted([f for f in os.listdir(sample_dir) if f.endswith(".in")])
        for inf in in_files:
            base = inf[:-3]
            ans_file = base + ".ans"
            inp = _load_code(sample_dir, inf) or ""
            ans = _load_code(sample_dir, ans_file) or ""
            if inp:
                sample_cases.append({"input": inp, "output": ans})

    return {
        "id": problem_id,
        "folder": folder,
        "question": question,
        "result": result,
        "brute_force_code": bf_code,
        "data_maker_code": dm_code,
        "solution_code": sol_code,
        "execution_log": log_text,
        "sample_cases": sample_cases,
    }


@app.post("/api/solve")
async def solve_problem(req: SolveRequest):
    """Submit a custom problem to solve.

    Creates a problem folder, writes question.json + sample test cases,
    then runs the LangGraph solver.
    """
    # Generate unique problem ID
    problem_id = f"user_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    folder = os.path.join(DATA_DIR, problem_id)
    os.makedirs(folder, exist_ok=True)
    os.makedirs(os.path.join(folder, "sample"), exist_ok=True)

    # Write question.json
    question_data = {
        "question": req.question,
        "input_format": req.input_format,
        "output_format": req.output_format,
        "data_range": req.data_range,
    }
    with open(os.path.join(folder, "question.json"), "w", encoding="utf-8") as f:
        json.dump(question_data, f, indent=2, ensure_ascii=False)

    # Write sample test cases
    for i, (inp, out) in enumerate(zip(req.test_inputs, req.test_outputs), 1):
        with open(os.path.join(folder, "sample", f"{i}.in"), "w", encoding="utf-8") as f:
            f.write(inp)
        with open(os.path.join(folder, "sample", f"{i}.ans"), "w", encoding="utf-8") as f:
            f.write(out)

    # Run solver in background
    task_id = str(uuid.uuid4())
    tasks[task_id] = _new_task_entry(problem_id, folder)

    asyncio.create_task(_run_solve(task_id, folder, req.ignore_format, req.max_retries, req.num_tests))

    return {"task_id": task_id, "problem_id": problem_id, "status": "running"}


@app.post("/api/solve-apps")
async def solve_apps_problem(req: SolveAppsRequest):
    """Solve an APPS dataset problem."""
    apps_dir = APPS_TRAIN_DIR if req.split == "train" else APPS_TEST_DIR
    apps_problem_dir = os.path.join(apps_dir, req.problem_id)
    if not os.path.isdir(apps_problem_dir):
        raise HTTPException(status_code=404, detail=f"APPS problem {req.split}/{req.problem_id} not found")

    converted_dir = os.path.join(PROJECT_ROOT, "APPS_converted", req.split, req.problem_id)

    # Skip if already solved and not forced
    if not req.force and os.path.isfile(os.path.join(converted_dir, "result.json")):
        result = _load_result(converted_dir) or {}
        return {
            "task_id": None,
            "problem_id": req.problem_id,
            "status": "skipped",
            "result": result,
        }

    # Convert
    try:
        convert_apps_problem(apps_problem_dir, converted_dir)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Conversion failed: {e}")

    # Run solver in background
    task_id = str(uuid.uuid4())
    tasks[task_id] = _new_task_entry(req.problem_id, converted_dir)

    asyncio.create_task(_run_solve(task_id, converted_dir, req.ignore_format, req.max_retries, NUM_GENERATED_TESTS))

    return {"task_id": task_id, "problem_id": req.problem_id, "status": "running"}


@app.get("/api/apps-list")
async def list_apps_problems(split: str = "train", limit: int = 100, offset: int = 0):
    """List available APPS problems with their solve status."""
    apps_dir = APPS_TRAIN_DIR if split == "train" else APPS_TEST_DIR
    if not os.path.isdir(apps_dir):
        return []

    problem_dirs = sorted([d for d in os.listdir(apps_dir) if os.path.isdir(os.path.join(apps_dir, d))])
    selected = problem_dirs[offset:offset + limit]

    results = []
    for pid in selected:
        converted_dir = os.path.join(PROJECT_ROOT, "APPS_converted", split, pid)
        r = _load_result(converted_dir) if os.path.isdir(converted_dir) else None
        results.append({
            "id": pid,
            "split": split,
            "solved": r is not None,
            "result": r.get("result") if r else None,
        })
    return results


@app.get("/api/config")
async def get_config():
    """Get the current LLM configuration (api_key masked for security)."""
    cfg = get_llm_config()
    # Mask the API key — show only last 4 chars
    key = cfg.get("api_key", "")
    masked_key = "*" * max(0, len(key) - 4) + key[-4:] if len(key) > 4 else "****"
    return {
        "model": cfg["model"],
        "base_url": cfg["base_url"],
        "api_key_masked": masked_key,
        "max_tokens": cfg["max_tokens"],
        "temperature": cfg["temperature"],
    }


@app.put("/api/config")
async def update_config(req: LLMConfigRequest):
    """Update LLM configuration fields. Only non-None fields are changed."""
    kwargs = {}
    if req.model is not None:
        kwargs["model"] = req.model
    if req.base_url is not None:
        kwargs["base_url"] = req.base_url
    if req.api_key is not None:
        kwargs["api_key"] = req.api_key
    if req.max_tokens is not None:
        kwargs["max_tokens"] = req.max_tokens
    if req.temperature is not None:
        kwargs["temperature"] = req.temperature

    updated = set_llm_config(**kwargs)
    return {
        "model": updated["model"],
        "base_url": updated["base_url"],
        "api_key_masked": "*" * max(0, len(updated["api_key"]) - 4) + updated["api_key"][-4:],
        "max_tokens": updated["max_tokens"],
        "temperature": updated["temperature"],
    }


# ───────────────────────── Codeforces Endpoints ─────────────────────────

@app.get("/api/cf/problems")
async def cf_problem_list(tags: str = "", rating_min: int = 800, rating_max: int = 3500,
                          count: int = 50, offset: int = 0):
    """List Codeforces problems from the API."""
    try:
        problems = fetch_problemset(
            tags=tags,
            rating_min=rating_min,
            rating_max=rating_max,
            count=count,
            offset=offset,
        )
        return problems
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/api/cf/import")
async def cf_import_problem(req: CFImportRequest):
    """Import a Codeforces problem: scrape statement + test cases, save to folder."""
    cf_dir = os.path.join(PROJECT_ROOT, "cf_problems", f"{req.contest_id}{req.index}")

    # Skip if already imported and not forced
    if not req.force and os.path.isfile(os.path.join(cf_dir, "question.json")):
        with open(os.path.join(cf_dir, "question.json"), "r", encoding="utf-8") as f:
            q = json.load(f)
        return {"status": "exists", "problem_id": f"{req.contest_id}{req.index}", "question": q}

    try:
        question_data = import_cf_problem(req.contest_id, req.index, cf_dir)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Import failed: {e}")

    return {"status": "imported", "problem_id": f"{req.contest_id}{req.index}", "question": question_data}


@app.post("/api/cf/translate")
async def cf_translate_problem(req: CFTranslateRequest):
    """Translate a problem statement from English to Chinese using LLM."""
    try:
        translated = await asyncio.get_event_loop().run_in_executor(
            None, lambda: translate_problem_statement(req.text)
        )
        return {"translated": translated}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Translation failed: {e}")


@app.post("/api/cf/solve")
async def cf_solve_problem(req: CFImportRequest):
    """Import a CF problem and immediately solve it."""
    cf_dir = os.path.join(PROJECT_ROOT, "cf_problems", f"{req.contest_id}{req.index}")

    # Import first (skip if already exists and not forced)
    if req.force or not os.path.isfile(os.path.join(cf_dir, "question.json")):
        try:
            import_cf_problem(req.contest_id, req.index, cf_dir)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Import failed: {e}")

    # Check if already solved
    if not req.force and os.path.isfile(os.path.join(cf_dir, "result.json")):
        result = _load_result(cf_dir) or {}
        return {"task_id": None, "problem_id": f"{req.contest_id}{req.index}", "status": "skipped", "result": result}

    # Run solver
    task_id = str(uuid.uuid4())
    tasks[task_id] = _new_task_entry(f"{req.contest_id}{req.index}", cf_dir)
    asyncio.create_task(_run_solve(task_id, cf_dir, False, MAX_RETRIES, NUM_GENERATED_TESTS))

    return {"task_id": task_id, "problem_id": f"{req.contest_id}{req.index}", "status": "running"}


@app.get("/api/task/{task_id}")
async def get_task_status(task_id: str):
    """Get the status of a background solve task."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    t = tasks[task_id]
    return {
        "task_id": task_id,
        "problem_id": t.get("problem_id"),
        "status": t.get("status", "unknown"),
        "stage": t.get("stage", ""),
        "log": t.get("log", []),
        "result": t.get("result"),
    }


@app.websocket("/ws/solve/{task_id}")
async def ws_solve_progress(websocket: WebSocket, task_id: str):
    """WebSocket endpoint for real-time solve progress updates.

    Uses asyncio.Event to push updates immediately when new log entries
    arrive, rather than polling on a fixed interval.
    """
    await websocket.accept()
    if task_id not in tasks:
        await websocket.send_json({"error": "Task not found"})
        await websocket.close()
        return

    last_log_len = 0
    try:
        while True:
            task = tasks.get(task_id, {})
            if not task:
                break

            # Send update if there are new log entries or status changed
            current_log = task.get("log", [])
            if len(current_log) != last_log_len or task.get("status") in ("completed", "failed"):
                await websocket.send_json({
                    "task_id": task_id,
                    "status": task.get("status", "unknown"),
                    "stage": task.get("stage", ""),
                    "log": current_log,
                    "result": task.get("result"),
                })
                last_log_len = len(current_log)

            if task.get("status") in ("completed", "failed"):
                break

            # Wait for the next update event (with 5s timeout as keepalive)
            evt = task.get("update_event")
            if evt:
                try:
                    await asyncio.wait_for(evt.wait(), timeout=5.0)
                    evt.clear()
                except asyncio.TimeoutError:
                    pass  # Send keepalive on next loop iteration
            else:
                await asyncio.sleep(2)  # Fallback polling if event not available
    except WebSocketDisconnect:
        pass


# ───────────────────────── Background Tasks ─────────────────────────

async def _run_solve(task_id: str, folder: str, ignore_format: bool, max_retries: int, num_tests: int):
    """Run the LangGraph solver in a background thread, updating task state.

    Uses invoke_solve's log_callback to stream log entries in real-time
    to the in-memory task dict, which the WebSocket handler pushes to clients.
    """
    task = tasks.get(task_id, {})
    try:
        loop = asyncio.get_event_loop()

        def log_callback(new_entries: list):
            """Called from executor thread — append logs and signal update."""
            task["log"] = task.get("log", []) + new_entries
            # Signal the WebSocket handler that new data is available
            evt = task.get("update_event")
            if evt and _loop and _loop.is_running():
                _loop.call_soon_threadsafe(evt.set)

        result = await loop.run_in_executor(
            None,
            lambda: invoke_solve(
                folder=folder,
                ignore_format=ignore_format,
                max_retries=max_retries,
                num_generated_tests=num_tests,
                save=True,
                log_callback=log_callback,
            )
        )
        task["status"] = "completed"
        task["stage"] = "done"
        task["result"] = {
            "result": result.get("result", "unknown"),
            "result_detail": result.get("result_detail", ""),
            "bf_attempt": result.get("bf_attempt", 0),
            "dm_attempt": result.get("dm_attempt", 0),
            "sol_attempt": result.get("sol_attempt", 0),
        }
        # Final log sync from result
        task["log"] = result.get("log", [])
    except Exception as e:
        task["status"] = "failed"
        task["stage"] = "error"
        task["result"] = {"error": str(e)}
        task["log"] = task.get("log", []) + [f"[ERROR] {e}"]
    finally:
        # Signal final update
        evt = task.get("update_event")
        if evt and _loop and _loop.is_running():
            _loop.call_soon_threadsafe(evt.set)


# ───────────────────────── Entry Point ─────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
