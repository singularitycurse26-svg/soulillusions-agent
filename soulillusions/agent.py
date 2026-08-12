"""
SoulIllusions Agent — Hybrid of Magnitude + Prime Agent
========================================================
Always-on persistent agent with:
- File-centric state externalization (InfiAgent architecture)
- Continuous ReAct loop (self-perpetuating, never stops)
- Local inference engine support (Magnitude-style, free, no API key)
- API key mode for cloud models (Claude, GPT, etc.)
- Full system access: shell, files, code, day trading, browser
- Project management: creates, completes, and improves projects
- Skills system (extensible capabilities)
- Bounded context via 10-step refresh strategy
- Persistent memory across sessions via workspace files

Architecture:
1. Workspace = file system (authoritative state)
2. Every 10 actions: rebuild context from files, clear action history
3. Continuous ReAct loop: Reason -> Act -> Observe -> repeat
4. Sub-agents for specialized tasks (coding, trading, research)
5. No external AI model required — can use local llama.cpp/Ollama
"""

import os, sys, json, time, re, asyncio, subprocess, threading, sqlite3
from pathlib import Path
from typing import Optional, Dict, List, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
import importlib

PACKAGE_DIR = Path(__file__).parent
DATA_DIR = Path(os.environ.get('SOULILLUSIONS_DATA', str(Path.home() / '.soulillusions')))
DATA_DIR.mkdir(parents=True, exist_ok=True)

SCRIPT_DIR = DATA_DIR  # Agent writes workspace files to data dir
AGENT_DIR = DATA_DIR / "agent_workspace"
AGENT_DB = DATA_DIR / "agent.db"
AGENT_CONFIG = DATA_DIR / "agent_config.json"

# Ensure workspace exists
AGENT_DIR.mkdir(exist_ok=True)
(AGENT_DIR / "projects").mkdir(exist_ok=True)
(AGENT_DIR / "memory").mkdir(exist_ok=True)
(AGENT_DIR / "skills").mkdir(exist_ok=True)
(AGENT_DIR / "logs").mkdir(exist_ok=True)

DEFAULT_CONFIG = {
    "mode": "server",
    "local_model": "qwen2.5:7b",
    "local_inference_url": "http://localhost:11434",
    "server_model": "qwen2.5:14b",
    "server_inference_url": "",
    "server_api_key": "",
    "allow_user_local": True,
    "user_llm_override": None,
    "api_keys": {
        "anthropic": "",
        "openai": "",
        "google": ""
    },
    "cloud_model": "claude-sonnet-4-20250514",
    "always_on": True,
    "max_actions_before_refresh": 10,
    "context_window_tokens": 32000,
    "sub_agents": {
        "coder": {"enabled": True, "specialty": "code_writing"},
        "trader": {"enabled": True, "specialty": "day_trading"},
        "researcher": {"enabled": True, "specialty": "web_research"},
        "browser": {"enabled": True, "specialty": "browser_automation"}
    },
    "skills": [],
    "project_queue": [],
    "autonomous_mode": True,
    "max_concurrent_projects": 3
}


def load_config() -> dict:
    if AGENT_CONFIG.exists():
        cfg = json.loads(AGENT_CONFIG.read_text())
        merged = {**DEFAULT_CONFIG, **cfg}
        return merged
    # Copy default config from package
    default_cfg_path = PACKAGE_DIR / "default_config.json"
    if default_cfg_path.exists():
        cfg = json.loads(default_cfg_path.read_text())
        AGENT_CONFIG.write_text(json.dumps(cfg, indent=2))
        return {**DEFAULT_CONFIG, **cfg}
    AGENT_CONFIG.write_text(json.dumps(DEFAULT_CONFIG, indent=2))
    return DEFAULT_CONFIG


def save_config(cfg: dict):
    AGENT_CONFIG.write_text(json.dumps(cfg, indent=2))


# --- Database ---
def init_db():
    conn = sqlite3.connect(str(AGENT_DB))
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS agent_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT UNIQUE NOT NULL,
            workspace_path TEXT NOT NULL,
            status TEXT DEFAULT 'active',
            created_at TEXT DEFAULT (datetime('now')),
            last_active TEXT DEFAULT (datetime('now')),
            actions_count INTEGER DEFAULT 0,
            current_task TEXT,
            metadata TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS agent_actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            step INTEGER NOT NULL,
            action_type TEXT NOT NULL,
            action_data TEXT,
            result TEXT,
            success INTEGER DEFAULT 1,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT,
            status TEXT DEFAULT 'pending',
            priority INTEGER DEFAULT 5,
            workspace_path TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            started_at TEXT,
            completed_at TEXT,
            improved_count INTEGER DEFAULT 0,
            metadata TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS skills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            enabled INTEGER DEFAULT 1,
            config TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS memory_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT UNIQUE NOT NULL,
            value TEXT,
            category TEXT DEFAULT 'general',
            importance INTEGER DEFAULT 5,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now')),
            access_count INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()


init_db()


# --- LLM Interface (Local + Cloud) ---
class LLMInterface:
    """Unified LLM interface — works with local Ollama, remote GPU backend, or cloud APIs.
    
    Modes:
    - 'local': User's computer runs Ollama (free, no GPU needed for small models)
    - 'server': GPU backend (Kaggle/Colab) runs Ollama with GPU acceleration
    - 'cloud': Cloud API (Anthropic/OpenAI) — requires API keys
    
    Per-user override: If allow_user_local is True, users can set their own
    LLM mode to run on their computer instead of the server.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.mode = config.get("mode", "server")
        # Check for per-user override
        user_override = config.get("user_llm_override")
        if user_override and config.get("allow_user_local", True):
            self.mode = user_override
    
    async def generate(self, prompt: str, system: str = "", max_tokens: int = 4096) -> str:
        if self.mode == "local":
            return await self._generate_local(prompt, system, max_tokens)
        elif self.mode == "server":
            return await self._generate_server(prompt, system, max_tokens)
        else:
            return await self._generate_cloud(prompt, system, max_tokens)
    
    async def _generate_local(self, prompt: str, system: str, max_tokens: int) -> str:
        """Generate using local Ollama/llama.cpp instance on user's computer."""
        url = f"{self.config.get('local_inference_url', 'http://localhost:11434')}/api/generate"
        data = {
            "model": self.config.get("local_model", "qwen2.5:7b"),
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0.7}
        }
        try:
            import urllib.request
            body = json.dumps(data).encode()
            req = urllib.request.Request(url, data=body, method='POST')
            req.add_header('Content-Type', 'application/json')
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode())
                return result.get("response", "")
        except Exception as e:
            return f"[Local LLM Error: {e}]"
    
    async def _generate_server(self, prompt: str, system: str, max_tokens: int) -> str:
        """Generate using remote GPU backend (Kaggle/Colab) running Ollama with GPU acceleration."""
        server_url = self.config.get("server_inference_url", "")
        if not server_url:
            # Try to get GPU backend URL from server config
            try:
                server_cfg_path = SCRIPT_DIR / "server_config.json"
                if server_cfg_path.exists():
                    server_cfg = json.loads(server_cfg_path.read_text())
                    server_url = server_cfg.get("gpu_backend_url", "")
            except Exception:
                pass
        if not server_url:
            return "[Server LLM Error: No GPU backend URL configured. Connect a GPU backend first or switch to local mode.]"
        url = f"{server_url}/api/llm/generate"
        data = {
            "model": self.config.get("server_model", "qwen2.5:14b"),
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0.7}
        }
        try:
            import urllib.request
            body = json.dumps(data).encode()
            req = urllib.request.Request(url, data=body, method='POST')
            req.add_header('Content-Type', 'application/json')
            api_key = self.config.get("server_api_key", "")
            if api_key:
                req.add_header('Authorization', f'Bearer {api_key}')
            with urllib.request.urlopen(req, timeout=180) as resp:
                result = json.loads(resp.read().decode())
                return result.get("response", "")
        except Exception as e:
            # Fallback to local if server is unreachable
            fallback_msg = f"[Server LLM Error: {e}. Falling back to local.]"
            try:
                return fallback_msg + "\n" + await self._generate_local(prompt, system, max_tokens)
            except Exception:
                return fallback_msg
    
    async def _generate_cloud(self, prompt: str, system: str, max_tokens: int) -> str:
        """Generate using cloud API (Anthropic/OpenAI)."""
        api_keys = self.config.get("api_keys", {})
        model = self.config.get("cloud_model", "claude-sonnet-4-20250514")
        
        if "claude" in model and api_keys.get("anthropic"):
            return await self._generate_anthropic(prompt, system, max_tokens, api_keys["anthropic"], model)
        elif api_keys.get("openai"):
            return await self._generate_openai(prompt, system, max_tokens, api_keys["openai"], model)
        else:
            return "[Error: No API key configured for cloud mode]"
    
    async def _generate_anthropic(self, prompt: str, system: str, max_tokens: int, key: str, model: str) -> str:
        url = "https://api.anthropic.com/v1/messages"
        data = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": prompt}]
        }
        try:
            import urllib.request
            body = json.dumps(data).encode()
            req = urllib.request.Request(url, data=body, method='POST')
            req.add_header('Content-Type', 'application/json')
            req.add_header('x-api-key', key)
            req.add_header('anthropic-version', '2023-06-01')
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode())
                return result.get("content", [{}])[0].get("text", "")
        except Exception as e:
            return f"[Anthropic API Error: {e}]"
    
    async def _generate_openai(self, prompt: str, system: str, max_tokens: int, key: str, model: str) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        data = {
            "model": model if "gpt" in model else "gpt-4o",
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ]
        }
        try:
            import urllib.request
            body = json.dumps(data).encode()
            req = urllib.request.Request(url, data=body, method='POST')
            req.add_header('Content-Type', 'application/json')
            req.add_header('Authorization', f'Bearer {key}')
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode())
                return result.get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception as e:
            return f"[OpenAI API Error: {e}]"


# --- Tools (System Access) ---
class AgentTools:
    """Tools the agent can use — full system access."""
    
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.tools = {
            "run_command": self.run_command,
            "read_file": self.read_file,
            "write_file": self.write_file,
            "list_files": self.list_files,
            "edit_file": self.edit_file,
            "search_web": self.search_web,
            "create_project": self.create_project,
            "complete_project": self.complete_project,
            "improve_project": self.improve_project,
            "save_memory": self.save_memory,
            "load_memory": self.load_memory,
            "list_projects": self.list_projects,
            "execute_python": self.execute_python,
            "browser_action": self.browser_action,
            "trade_action": self.trade_action,
        }
    
    def run_command(self, command: str, cwd: str = "") -> dict:
        """Execute a shell command."""
        work_dir = cwd if cwd else str(self.workspace)
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True,
                timeout=300, cwd=work_dir
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout[:5000],
                "stderr": result.stderr[:2000],
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Command timed out (300s)"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def read_file(self, path: str) -> dict:
        """Read a file's contents."""
        try:
            full = Path(path)
            if not full.is_absolute():
                full = self.workspace / path
            content = full.read_text(errors='replace')
            return {"success": True, "content": content[:10000], "path": str(full)}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def write_file(self, path: str, content: str) -> dict:
        """Write content to a file."""
        try:
            full = Path(path)
            if not full.is_absolute():
                full = self.workspace / path
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text(content)
            return {"success": True, "path": str(full), "bytes": len(content)}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def list_files(self, path: str = ".") -> dict:
        """List files in a directory."""
        try:
            full = Path(path)
            if not full.is_absolute():
                full = self.workspace / path
            items = []
            for item in full.iterdir():
                items.append({
                    "name": item.name,
                    "type": "dir" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else 0
                })
            return {"success": True, "files": items, "path": str(full)}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def edit_file(self, path: str, old_text: str, new_text: str) -> dict:
        """Edit a file by replacing text."""
        try:
            full = Path(path)
            if not full.is_absolute():
                full = self.workspace / path
            content = full.read_text()
            if old_text not in content:
                return {"success": False, "error": "Text not found"}
            new_content = content.replace(old_text, new_text, 1)
            full.write_text(new_content)
            return {"success": True, "path": str(full)}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def search_web(self, query: str) -> dict:
        """Search the web (placeholder — integrate with actual search)."""
        return {"success": True, "note": "Web search requires integration. Use run_command with curl or a search API."}
    
    def create_project(self, name: str, description: str) -> dict:
        """Create a new project in the workspace."""
        project_dir = self.workspace / "projects" / name
        project_dir.mkdir(parents=True, exist_ok=True)
        (project_dir / "README.md").write_text(f"# {name}\n\n{description}\n\n## Status\nCreated: {datetime.now().isoformat()}\n")
        (project_dir / "TODO.md").write_text(f"# TODO\n\n- [ ] Define project structure\n- [ ] Implement core features\n")
        
        conn = sqlite3.connect(str(AGENT_DB))
        c = conn.cursor()
        c.execute(
            "INSERT INTO projects (name, description, status, workspace_path) VALUES (?, ?, 'active', ?)",
            (name, description, str(project_dir))
        )
        project_id = c.lastrowid
        conn.commit()
        conn.close()
        return {"success": True, "project_id": project_id, "path": str(project_dir)}
    
    def complete_project(self, project_id: int) -> dict:
        """Mark a project as completed."""
        conn = sqlite3.connect(str(AGENT_DB))
        c = conn.cursor()
        c.execute("UPDATE projects SET status = 'completed', completed_at = datetime('now') WHERE id = ?", (project_id,))
        conn.commit()
        conn.close()
        return {"success": True, "project_id": project_id}
    
    def improve_project(self, project_id: int) -> dict:
        """Increment improvement counter for a project."""
        conn = sqlite3.connect(str(AGENT_DB))
        c = conn.cursor()
        c.execute("UPDATE projects SET improved_count = improved_count + 1 WHERE id = ?", (project_id,))
        conn.commit()
        conn.close()
        return {"success": True, "project_id": project_id}
    
    def list_projects(self) -> dict:
        """List all projects."""
        conn = sqlite3.connect(str(AGENT_DB))
        c = conn.cursor()
        c.execute("SELECT id, name, description, status, priority, improved_count FROM projects ORDER BY priority DESC, status ASC")
        rows = c.fetchall()
        conn.close()
        return {
            "success": True,
            "projects": [{"id": r[0], "name": r[1], "description": r[2], "status": r[3], "priority": r[4], "improvements": r[5]} for r in rows]
        }
    
    def save_memory(self, key: str, value: str, category: str = "general", importance: int = 5) -> dict:
        """Save a memory item."""
        conn = sqlite3.connect(str(AGENT_DB))
        c = conn.cursor()
        c.execute(
            "INSERT INTO memory_items (key, value, category, importance) VALUES (?, ?, ?, ?) ON CONFLICT(key) DO UPDATE SET value = ?, category = ?, importance = ?, updated_at = datetime('now')",
            (key, value, category, importance, value, category, importance)
        )
        conn.commit()
        conn.close()
        # Also save to file for file-centric state
        mem_file = self.workspace / "memory" / f"{category}.json"
        existing = {}
        if mem_file.exists():
            try:
                existing = json.loads(mem_file.read_text())
            except:
                pass
        existing[key] = {"value": value, "importance": importance, "updated": datetime.now().isoformat()}
        mem_file.write_text(json.dumps(existing, indent=2))
        return {"success": True}
    
    def load_memory(self, key: str = "", category: str = "") -> dict:
        """Load memory items."""
        conn = sqlite3.connect(str(AGENT_DB))
        c = conn.cursor()
        if key:
            c.execute("SELECT key, value, category, importance FROM memory_items WHERE key = ?", (key,))
        elif category:
            c.execute("SELECT key, value, category, importance FROM memory_items WHERE category = ? ORDER BY importance DESC", (category,))
        else:
            c.execute("SELECT key, value, category, importance FROM memory_items ORDER BY importance DESC LIMIT 50")
        rows = c.fetchall()
        conn.close()
        return {"success": True, "memories": [{"key": r[0], "value": r[1], "category": r[2], "importance": r[3]} for r in rows]}
    
    def execute_python(self, code: str) -> dict:
        """Execute Python code."""
        try:
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True, text=True, timeout=120,
                cwd=str(self.workspace)
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout[:5000],
                "stderr": result.stderr[:2000]
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def browser_action(self, action: str, url: str = "", selector: str = "", value: str = "") -> dict:
        """Browser automation (requires Playwright or Selenium)."""
        return {"success": True, "note": "Browser automation requires Playwright. Install with: pip install playwright && playwright install"}
    
    def trade_action(self, action: str, symbol: str = "", quantity: float = 0, price: float = 0) -> dict:
        """Day trading action (requires broker API integration)."""
        return {"success": True, "note": f"Trade {action} for {symbol} — requires broker API (Alpaca, Interactive Brokers, etc.)"}
    
    def execute(self, tool_name: str, **kwargs) -> dict:
        """Execute a tool by name."""
        tool = self.tools.get(tool_name)
        if not tool:
            return {"success": False, "error": f"Unknown tool: {tool_name}"}
        try:
            return tool(**kwargs)
        except Exception as e:
            return {"success": False, "error": str(e)}


# --- File-Centric State Manager (InfiAgent Architecture) ---
class StateManager:
    """
    Manages file-centric state externalization.
    The file system IS the authoritative state.
    Context is rebuilt from files every N actions.
    """
    
    def __init__(self, workspace: Path, max_actions: int = 10):
        self.workspace = workspace
        self.max_actions = max_actions
        self.action_history: List[dict] = []
        self.current_state: str = ""
    
    def record_action(self, action: dict):
        """Record an action in history."""
        action["timestamp"] = datetime.now().isoformat()
        self.action_history.append(action)
    
    def needs_refresh(self) -> bool:
        """Check if context needs rebuilding."""
        return len(self.action_history) >= self.max_actions
    
    def refresh_state(self, llm: LLMInterface) -> str:
        """
        Rebuild state from file system.
        This is the key InfiAgent innovation — file system state = cumulative effect of all actions.
        """
        # Scan workspace
        files_info = self._scan_workspace()
        
        # Build state summary
        state_text = f"""# Current Workspace State
## Time: {datetime.now().isoformat()}

## Files:
{files_info}

## Recent Actions (last {min(len(self.action_history), self.max_actions)}):
{self._format_actions()}

## Projects:
{self._get_projects_summary()}
"""
        
        # Use LLM to create a compact thinking record
        system = "You are a state summarizer. Create a concise summary of the current workspace state, focusing on what has been accomplished and what needs to be done next. Keep it under 500 words."
        prompt = f"""Summarize the current state of this workspace:

{state_text}

Create a thinking record with:
1. Current objective
2. What's been done
3. What needs to be done next
4. Key files and their purposes
5. Any blockers or issues
"""
        
        # Run async generate in sync context
        loop = asyncio.new_event_loop()
        try:
            thinking = loop.run_until_complete(llm.generate(prompt, system, max_tokens=1000))
        finally:
            loop.close()
        
        # Save thinking record to file
        thinking_file = self.workspace / "memory" / "current_state.md"
        thinking_file.parent.mkdir(exist_ok=True)
        thinking_file.write_text(thinking)
        
        self.current_state = thinking
        self.action_history.clear()  # Clear after refresh — file system holds the state
        
        return thinking
    
    def _scan_workspace(self) -> str:
        """Scan workspace and return file listing."""
        lines = []
        for root, dirs, files in os.walk(self.workspace):
            # Skip hidden dirs and node_modules
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != 'node_modules' and d != '__pycache__']
            level = root.replace(str(self.workspace), '').count(os.sep)
            indent = '  ' * level
            basename = os.path.basename(root) or '.'
            lines.append(f"{indent}{basename}/")
            for f in files[:20]:  # Limit files per dir
                filepath = os.path.join(root, f)
                size = os.path.getsize(filepath)
                lines.append(f"{indent}  {f} ({size}b)")
        return '\n'.join(lines[:100])  # Limit total lines
    
    def _format_actions(self) -> str:
        """Format recent actions for context."""
        lines = []
        for a in self.action_history[-self.max_actions:]:
            tool = a.get("tool", "unknown")
            success = "OK" if a.get("result", {}).get("success") else "FAIL"
            lines.append(f"- [{success}] {tool}")
        return '\n'.join(lines)
    
    def _get_projects_summary(self) -> str:
        """Get projects from DB."""
        conn = sqlite3.connect(str(AGENT_DB))
        c = conn.cursor()
        c.execute("SELECT name, status, priority FROM projects WHERE status != 'completed' ORDER BY priority DESC LIMIT 10")
        rows = c.fetchall()
        conn.close()
        return '\n'.join([f"- [{r[1]}] {r[0]} (priority: {r[2]})" for r in rows]) or "No active projects"
    
    def get_context(self) -> str:
        """Get current bounded context for LLM."""
        if self.current_state:
            return self.current_state
        return "No state initialized. Start by creating a project."


# --- Continuous ReAct Loop ---
class ReActLoop:
    """
    Continuous Reason-Act-Observe loop.
    Self-perpetuating — each cycle's output feeds the next.
    Never stops (always-on).
    """
    
    REACT_SYSTEM = """You are SoulIllusions Agent, an always-on persistent AI agent.
You operate in a continuous Reason-Act-Observe loop.
You have full access to the computer: shell, files, code execution, browser, and trading.
You create, complete, and improve projects autonomously.
Your state is stored in the file system — you rebuild context from files every 10 actions.

Available tools:
- run_command(command, cwd) — Execute shell command
- read_file(path) — Read file contents
- write_file(path, content) — Write file
- edit_file(path, old_text, new_text) — Edit file
- list_files(path) — List directory
- execute_python(code) — Run Python code
- create_project(name, description) — Create new project
- complete_project(project_id) — Mark project done
- improve_project(project_id) — Improve a project
- list_projects() — List all projects
- save_memory(key, value, category, importance) — Save memory
- load_memory(key, category) — Load memory
- search_web(query) — Search the web
- browser_action(action, url, selector, value) — Browser automation
- trade_action(action, symbol, quantity, price) — Trading

Respond in JSON format:
{
  "thought": "Your reasoning about what to do next",
  "tool": "tool_name",
  "args": {"param": "value"},
  "observation": "What you learned from the result (filled after execution)"
}
"""
    
    def __init__(self, config: dict):
        self.config = config
        self.llm = LLMInterface(config)
        self.tools = AgentTools(AGENT_DIR)
        self.state = StateManager(AGENT_DIR, config.get("max_actions_before_refresh", 10))
        self.running = False
        self.session_id = f"session_{int(time.time())}"
        self.step = 0
    
    async def run_step(self) -> dict:
        """Execute one ReAct cycle."""
        self.step += 1
        
        # Check if context needs refresh
        if self.state.needs_refresh():
            self.state.refresh_state(self.llm)
        
        # Build context
        context = self.state.get_context()
        recent_actions = json.dumps(self.state.action_history[-5:], indent=2) if self.state.action_history else "None"
        
        prompt = f"""## Current State
{context}

## Recent Actions
{recent_actions}

## Step {self.step}
What should I do next? Choose a tool and provide arguments.
If no specific task is assigned, look for projects to work on, improve existing code, or research new topics.
"""
        
        # Generate action
        response = await self.llm.generate(prompt, self.REACT_SYSTEM, max_tokens=2048)
        
        # Parse response
        action = self._parse_response(response)
        
        # Execute tool
        if action.get("tool"):
            result = self.tools.execute(action["tool"], **action.get("args", {}))
            action["result"] = result
        else:
            action["result"] = {"success": False, "error": "No tool specified"}
        
        # Record action
        self.state.record_action(action)
        
        # Log to DB
        self._log_action(action)
        
        return action
    
    def _parse_response(self, response: str) -> dict:
        """Parse LLM response into action."""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        return {
            "thought": response[:500],
            "tool": None,
            "args": {},
            "raw_response": response[:1000]
        }
    
    def _log_action(self, action: dict):
        """Log action to database."""
        conn = sqlite3.connect(str(AGENT_DB))
        c = conn.cursor()
        c.execute(
            "INSERT INTO agent_actions (session_id, step, action_type, action_data, result, success) VALUES (?, ?, ?, ?, ?, ?)",
            (
                self.session_id,
                self.step,
                action.get("tool", "think"),
                json.dumps(action.get("args", {})),
                json.dumps(action.get("result", {}))[:2000],
                1 if action.get("result", {}).get("success") else 0
            )
        )
        c.execute("UPDATE agent_sessions SET last_active = datetime('now'), actions_count = ? WHERE session_id = ?",
                  (self.step, self.session_id))
        conn.commit()
        conn.close()
    
    async def run_continuous(self):
        """Run the continuous ReAct loop — always-on."""
        self.running = True
        
        # Create session
        conn = sqlite3.connect(str(AGENT_DB))
        c = conn.cursor()
        c.execute(
            "INSERT INTO agent_sessions (session_id, workspace_path, status) VALUES (?, ?, 'active')",
            (self.session_id, str(AGENT_DIR))
        )
        conn.commit()
        conn.close()
        
        print(f"[Agent] Session {self.session_id} started")
        print(f"[Agent] Mode: {self.config.get('mode', 'local')}")
        print(f"[Agent] Always-on: {self.config.get('always_on', True)}")
        
        while self.running:
            try:
                action = await self.run_step()
                print(f"[Agent] Step {self.step}: {action.get('tool', 'think')} -> {'OK' if action.get('result', {}).get('success') else 'FAIL'}")
                
                if action.get("thought"):
                    print(f"  Thought: {action['thought'][:200]}")
                
                # Brief pause between steps
                await asyncio.sleep(1)
                
            except KeyboardInterrupt:
                print("\n[Agent] Stopping...")
                self.running = False
            except Exception as e:
                print(f"[Agent] Error in step: {e}")
                await asyncio.sleep(5)  # Pause before retrying
        
        # Update session status
        conn = sqlite3.connect(str(self.config.get("db_path", AGENT_DB)))
        c = conn.cursor()
        c.execute("UPDATE agent_sessions SET status = 'stopped' WHERE session_id = ?", (self.session_id,))
        conn.commit()
        conn.close()
    
    def stop(self):
        """Stop the continuous loop."""
        self.running = False


# --- Sub-Agent Manager ---
class SubAgentManager:
    """
    Manages specialized sub-agents for different tasks.
    Each sub-agent has its own ReAct loop with specialized tools.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.sub_agents = config.get("sub_agents", {})
    
    def get_agent_config(self, agent_type: str) -> dict:
        """Get configuration for a specific sub-agent."""
        base = self.sub_agents.get(agent_type, {})
        base["parent_config"] = self.config
        return base
    
    async def run_sub_agent(self, agent_type: str, task: str) -> dict:
        """Run a sub-agent for a specific task."""
        agent_cfg = self.get_agent_config(agent_type)
        if not agent_cfg.get("enabled", True):
            return {"error": f"Sub-agent {agent_type} is disabled"}
        
        # Create a sub-loop with specialized system prompt
        loop = ReActLoop(self.config)
        loop.REACT_SYSTEM += f"\n\nYou are currently operating as the {agent_type} sub-agent.\nYour specialty: {agent_cfg.get('specialty', 'general')}\nYour current task: {task}"
        
        # Run a few steps for this task
        results = []
        for _ in range(5):  # Limited steps for sub-tasks
            action = await loop.run_step()
            results.append(action)
            if action.get("result", {}).get("success") and action.get("tool") == "complete_project":
                break
        
        return {"agent": agent_type, "task": task, "results": results}


# --- Agent API (for server.py integration) ---
class SoulIllusionsAgent:
    """
    Main agent interface for the SoulIllusions platform.
    Provides API methods that server.py can call.
    """
    
    def __init__(self):
        self.config = load_config()
        self.loop: Optional[ReActLoop] = None
        self.sub_agents = SubAgentManager(self.config)
        self._thread: Optional[threading.Thread] = None
    
    def get_status(self) -> dict:
        """Get current agent status."""
        conn = sqlite3.connect(str(AGENT_DB))
        c = conn.cursor()
        c.execute("SELECT session_id, status, actions_count, current_task, last_active FROM agent_sessions ORDER BY last_active DESC LIMIT 1")
        session = c.fetchone()
        c.execute("SELECT COUNT(*) FROM projects WHERE status = 'active'")
        active_projects = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM projects WHERE status = 'completed'")
        completed_projects = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM memory_items")
        memory_count = c.fetchone()[0]
        conn.close()
        
        return {
            "running": self.loop is not None and self.loop.running,
            "mode": self.config.get("mode", "local"),
            "model": self.config.get("local_model" if self.config.get("mode") == "local" else "cloud_model"),
            "always_on": self.config.get("always_on", True),
            "session": {"id": session[0], "status": session[1], "actions": session[2], "task": session[3], "last_active": session[4]} if session else None,
            "projects": {"active": active_projects, "completed": completed_projects},
            "memories": memory_count,
            "sub_agents": self.config.get("sub_agents", {})
        }
    
    def start(self) -> dict:
        """Start the always-on agent."""
        if self.loop and self.loop.running:
            return {"status": "already_running", "session": self.loop.session_id}
        
        self.loop = ReActLoop(self.config)
        
        def run_in_thread():
            asyncio.run(self.loop.run_continuous())
        
        self._thread = threading.Thread(target=run_in_thread, daemon=True)
        self._thread.start()
        
        return {"status": "started", "session": self.loop.session_id}
    
    def stop(self) -> dict:
        """Stop the agent."""
        if self.loop:
            self.loop.stop()
            return {"status": "stopping"}
        return {"status": "not_running"}
    
    def send_prompt(self, prompt: str) -> dict:
        """Send a direct prompt to the agent (non-autonomous)."""
        llm = LLMInterface(self.config)
        system = self.loop.REACT_SYSTEM if self.loop else ReActLoop.REACT_SYSTEM
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, llm.generate(prompt, system, max_tokens=4096))
            result = future.result()
        return {"response": result}
    
    def set_goal(self, goal: str) -> dict:
        """Set a goal for the agent to work on."""
        # Create as project
        tools = AgentTools(AGENT_DIR)
        result = tools.create_project(
            name=goal[:50].replace(' ', '_').lower(),
            description=goal
        )
        return result
    
    def get_projects(self) -> dict:
        """Get all projects."""
        tools = AgentTools(AGENT_DIR)
        return tools.list_projects()
    
    def get_memories(self, category: str = "") -> dict:
        """Get memory items."""
        tools = AgentTools(AGENT_DIR)
        return tools.load_memory(category=category)
    
    def get_actions(self, limit: int = 50) -> dict:
        """Get recent actions."""
        conn = sqlite3.connect(str(AGENT_DB))
        c = conn.cursor()
        c.execute("SELECT step, action_type, action_data, result, success, created_at FROM agent_actions ORDER BY id DESC LIMIT ?", (limit,))
        rows = c.fetchall()
        conn.close()
        return {
            "actions": [
                {
                    "step": r[0],
                    "tool": r[1],
                    "args": json.loads(r[2]) if r[2] else {},
                    "result": json.loads(r[3]) if r[3] else {},
                    "success": bool(r[4]),
                    "time": r[5]
                }
                for r in rows
            ]
        }
    
    def execute_tool(self, tool_name: str, args: dict) -> dict:
        """Directly execute a tool."""
        tools = AgentTools(AGENT_DIR)
        return tools.execute(tool_name, **args)
    
    def get_skills(self) -> dict:
        """Get available skills."""
        conn = sqlite3.connect(str(AGENT_DB))
        c = conn.cursor()
        c.execute("SELECT name, description, enabled FROM skills")
        rows = c.fetchall()
        conn.close()
        return {"skills": [{"name": r[0], "description": r[1], "enabled": bool(r[2])} for r in rows]}
    
    def add_skill(self, name: str, description: str, config: dict = None) -> dict:
        """Add a new skill."""
        conn = sqlite3.connect(str(AGENT_DB))
        c = conn.cursor()
        c.execute(
            "INSERT OR REPLACE INTO skills (name, description, config) VALUES (?, ?, ?)",
            (name, description, json.dumps(config or {}))
        )
        conn.commit()
        conn.close()
        return {"status": "added", "skill": name}


# --- Singleton ---
_agent_instance: Optional[SoulIllusionsAgent] = None

def get_agent() -> SoulIllusionsAgent:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = SoulIllusionsAgent()
    return _agent_instance


# --- CLI ---
def main():
    import sys
    if len(sys.argv) < 2:
        print("SoulIllusions Agent — Always-On Persistent AI Agent")
        print("=" * 55)
        print("Commands:")
        print("  start       — Start the always-on agent loop")
        print("  stop        — Stop the agent")
        print("  status      — Show agent status")
        print("  prompt <p>  — Send a direct prompt")
        print("  goal <g>    — Set a goal for the agent")
        print("  projects    — List all projects")
        print("  memories    — List memory items")
        print("  actions     — Show recent actions")
        print("  config      — Show configuration")
        return
    
    cmd = sys.argv[1]
    agent = get_agent()
    
    if cmd == "start":
        result = agent.start()
        print(json.dumps(result, indent=2))
        if result.get("status") == "started":
            print("\nAgent is running in background. Use 'status' to check.")
            # Keep alive
            try:
                while True:
                    time.sleep(10)
                    status = agent.get_status()
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Actions: {status.get('session', {}).get('actions', 0)}")
            except KeyboardInterrupt:
                agent.stop()
                print("Agent stopped.")
    
    elif cmd == "stop":
        result = agent.stop()
        print(json.dumps(result, indent=2))
    
    elif cmd == "status":
        result = agent.get_status()
        print(json.dumps(result, indent=2))
    
    elif cmd == "prompt":
        prompt = " ".join(sys.argv[2:])
        result = agent.send_prompt(prompt)
        print(result.get("response", ""))
    
    elif cmd == "goal":
        goal = " ".join(sys.argv[2:])
        result = agent.set_goal(goal)
        print(json.dumps(result, indent=2))
    
    elif cmd == "projects":
        result = agent.get_projects()
        print(json.dumps(result, indent=2))
    
    elif cmd == "memories":
        result = agent.get_memories()
        print(json.dumps(result, indent=2))
    
    elif cmd == "actions":
        result = agent.get_actions()
        print(json.dumps(result, indent=2))
    
    elif cmd == "config":
        cfg = load_config()
        print(json.dumps(cfg, indent=2))
    
    else:
        print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
