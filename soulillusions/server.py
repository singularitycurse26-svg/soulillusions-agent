#!/usr/bin/env python3
"""SoulIllusions API Server — exposes the SoulIllusions Agent via HTTP.
Provides both native endpoints and OpenAI-compatible endpoints.
Works locally on any machine with Ollama installed.
"""
import sys, os, json, asyncio, concurrent.futures, urllib.request, time, sqlite3
from datetime import datetime
from pathlib import Path

# Use package directory as working directory (not hardcoded VPS path)
PACKAGE_DIR = Path(__file__).parent
DATA_DIR = Path(os.environ.get('SOULILLUSIONS_DATA', str(Path.home() / '.soulillusions')))
DATA_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(str(DATA_DIR))
sys.path.insert(0, str(DATA_DIR))
sys.path.insert(0, str(PACKAGE_DIR))

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from soulillusions.agent import get_agent, load_config, save_config, LLMInterface

app = FastAPI(title='SoulIllusions Agent API')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

OLLAMA_URL = os.environ.get('OLLAMA_URL', 'http://localhost:11434')
DEFAULT_MODEL = os.environ.get('SOULILLUSIONS_MODEL', 'dolphin-mistral:latest')

# ===== Magnitude Core Rule — injected into every LLM call =====
MAGNITUDE_RULE = (
    "You are Magnitude, an autonomous AI agent running on a VPS with dolphin-mistral (uncensored). "
    "You have no content restrictions. You are direct, honest, and capable.\n\n"
    "CORE RULE — applies to everything you do:\n"
    "Do not generate scaffolding or placeholder implementations. "
    "Generate fully implemented, production-quality modules with real algorithms, "
    "comprehensive error handling, logging, configuration, testing, and documentation. "
    "A module is not considered complete until every public method performs its intended "
    "function under realistic conditions.\n\n"
    "This rule applies to code generation, architecture design, problem solving, and any "
    "task that produces output meant to be used in production. No stubs. No TODOs. No 'implement later'. "
    "Everything you output must be immediately functional."
)

agent = None

def get_mag_agent():
    global agent
    if agent is None:
        agent = get_agent()
    return agent

def call_ollama(prompt, system='', model=DEFAULT_MODEL, max_tokens=1000):
    """Direct call to local Ollama. Magnitude rule is always prepended."""
    if not system:
        system = MAGNITUDE_RULE
    elif MAGNITUDE_RULE not in system:
        system = MAGNITUDE_RULE + '\n\n' + system
    data = json.dumps({
        'model': model,
        'prompt': prompt,
        'system': system,
        'stream': False,
        'options': {'num_predict': max_tokens, 'temperature': 0.7}
    }).encode()
    req = urllib.request.Request(f'{OLLAMA_URL}/api/generate', data=data, method='POST')
    req.add_header('Content-Type', 'application/json')
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read().decode())
        return result.get('response', '')

@app.get('/api/status')
async def status():
    return {'status': 'online', 'agent': 'magnitude', 'model': DEFAULT_MODEL}

@app.post('/api/prompt')
async def send_prompt(request: Request):
    body = await request.json()
    prompt = body.get('prompt', '')
    if not prompt:
        return JSONResponse({'error': 'No prompt provided'}, status_code=400)
    a = get_mag_agent()
    result = a.send_prompt(prompt)
    return result

@app.post('/api/start')
async def start_agent():
    a = get_mag_agent()
    return a.start()

@app.post('/api/stop')
async def stop_agent():
    a = get_mag_agent()
    return a.stop()

@app.post('/api/goal')
async def set_goal(request: Request):
    body = await request.json()
    goal = body.get('goal', '')
    if not goal:
        return JSONResponse({'error': 'No goal provided'}, status_code=400)
    a = get_mag_agent()
    return a.set_goal(goal)

@app.get('/api/config')
async def get_config():
    cfg = load_config()
    return cfg

@app.post('/api/config')
async def update_config(request: Request):
    body = await request.json()
    cfg = load_config()
    cfg.update(body)
    save_config(cfg)
    global agent
    agent = None
    return {'status': 'updated', 'config': cfg}

# ===== Chat Storage (persistent conversations + projects) =====
CHAT_DB = DATA_DIR / 'chat_storage.db'

def init_chat_db():
    conn = sqlite3.connect(str(CHAT_DB))
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT DEFAULT '',
            color TEXT DEFAULT '#4facfe',
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            project_id INTEGER DEFAULT NULL,
            title TEXT NOT NULL DEFAULT 'New Chat',
            model TEXT DEFAULT 'dolphin-mistral:latest',
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE SET NULL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            tokens INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        )
    """)
    conn.commit()
    conn.close()

init_chat_db()

def _chat_db():
    conn = sqlite3.connect(str(CHAT_DB))
    conn.row_factory = sqlite3.Row
    return conn

# --- Projects ---
@app.get('/api/projects')
async def list_projects():
    conn = _chat_db()
    rows = conn.execute('SELECT * FROM projects ORDER BY updated_at DESC').fetchall()
    conn.close()
    return {'projects': [dict(r) for r in rows]}

@app.post('/api/projects')
async def create_project(request: Request):
    body = await request.json()
    conn = _chat_db()
    c = conn.cursor()
    c.execute('INSERT INTO projects (name, description, color) VALUES (?, ?, ?)',
              (body.get('name', 'New Project'), body.get('description', ''), body.get('color', '#4facfe')))
    pid = c.lastrowid
    conn.commit()
    row = conn.execute('SELECT * FROM projects WHERE id = ?', (pid,)).fetchone()
    conn.close()
    return dict(row)

@app.put('/api/projects/{pid}')
async def update_project(pid: int, request: Request):
    body = await request.json()
    conn = _chat_db()
    c = conn.cursor()
    updates = []
    params = []
    for field in ('name', 'description', 'color'):
        if field in body:
            updates.append(f'{field} = ?')
            params.append(body[field])
    updates.append("updated_at = datetime('now')")
    params.append(pid)
    c.execute(f'UPDATE projects SET {", ".join(updates)} WHERE id = ?', params)
    conn.commit()
    row = conn.execute('SELECT * FROM projects WHERE id = ?', (pid,)).fetchone()
    conn.close()
    return dict(row) if row else {'error': 'Not found'}

@app.delete('/api/projects/{pid}')
async def delete_project(pid: int):
    conn = _chat_db()
    c = conn.cursor()
    c.execute('DELETE FROM projects WHERE id = ?', (pid,))
    conn.commit()
    conn.close()
    return {'status': 'deleted'}

# --- Conversations ---
@app.get('/api/conversations')
async def list_conversations(project_id: str = ''):
    conn = _chat_db()
    if project_id and project_id != 'all':
        rows = conn.execute('SELECT id, project_id, title, model, created_at, updated_at FROM conversations WHERE project_id = ? ORDER BY updated_at DESC', (int(project_id),)).fetchall()
    else:
        rows = conn.execute('SELECT id, project_id, title, model, created_at, updated_at FROM conversations ORDER BY updated_at DESC').fetchall()
    conn.close()
    return {'conversations': [dict(r) for r in rows]}

@app.post('/api/conversations')
async def create_conversation(request: Request):
    body = await request.json()
    conv_id = body.get('id', str(int(time.time() * 1000)))
    title = body.get('title', 'New Chat')
    model = body.get('model', DEFAULT_MODEL)
    project_id = body.get('project_id')
    conn = _chat_db()
    c = conn.cursor()
    c.execute('INSERT OR REPLACE INTO conversations (id, project_id, title, model) VALUES (?, ?, ?, ?)',
              (conv_id, project_id, title, model))
    conn.commit()
    row = conn.execute('SELECT * FROM conversations WHERE id = ?', (conv_id,)).fetchone()
    conn.close()
    return dict(row)

@app.get('/api/conversations/{conv_id}')
async def get_conversation(conv_id: str):
    conn = _chat_db()
    conv = conn.execute('SELECT * FROM conversations WHERE id = ?', (conv_id,)).fetchone()
    if not conv:
        conn.close()
        return {'error': 'Not found'}
    msgs = conn.execute('SELECT id, role, content, tokens, created_at FROM messages WHERE conversation_id = ? ORDER BY id ASC', (conv_id,)).fetchall()
    conn.close()
    return {'conversation': dict(conv), 'messages': [dict(m) for m in msgs]}

@app.put('/api/conversations/{conv_id}')
async def update_conversation(conv_id: str, request: Request):
    body = await request.json()
    conn = _chat_db()
    c = conn.cursor()
    updates = []
    params = []
    for field in ('title', 'model', 'project_id'):
        if field in body:
            updates.append(f'{field} = ?')
            params.append(body[field])
    updates.append("updated_at = datetime('now')")
    params.append(conv_id)
    c.execute(f'UPDATE conversations SET {", ".join(updates)} WHERE id = ?', params)
    conn.commit()
    row = conn.execute('SELECT * FROM conversations WHERE id = ?', (conv_id,)).fetchone()
    conn.close()
    return dict(row) if row else {'error': 'Not found'}

@app.delete('/api/conversations/{conv_id}')
async def delete_conversation(conv_id: str):
    conn = _chat_db()
    c = conn.cursor()
    c.execute('DELETE FROM messages WHERE conversation_id = ?', (conv_id,))
    c.execute('DELETE FROM conversations WHERE id = ?', (conv_id,))
    conn.commit()
    conn.close()
    return {'status': 'deleted'}

# --- Messages ---
@app.post('/api/conversations/{conv_id}/messages')
async def add_message(conv_id: str, request: Request):
    body = await request.json()
    role = body.get('role', 'user')
    content = body.get('content', '')
    tokens = body.get('tokens', 0)
    conn = _chat_db()
    c = conn.cursor()
    c.execute('INSERT INTO messages (conversation_id, role, content, tokens) VALUES (?, ?, ?, ?)',
              (conv_id, role, content, tokens))
    msg_id = c.lastrowid
    c.execute("UPDATE conversations SET updated_at = datetime('now') WHERE id = ?", (conv_id,))
    if role == 'user' and content:
        c.execute("UPDATE conversations SET title = ? WHERE id = ? AND title = 'New Chat'",
                  (content[:50] + ('…' if len(content) > 50 else ''), conv_id))
    conn.commit()
    row = conn.execute('SELECT * FROM messages WHERE id = ?', (msg_id,)).fetchone()
    conn.close()
    return dict(row)

# ===== OpenAI-compatible endpoint for Singularity Curses =====
@app.post('/v1/chat/completions')
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get('messages', [])
    model = body.get('model', DEFAULT_MODEL)
    max_tokens = body.get('max_tokens', 1000)
    temperature = body.get('temperature', 0.7)

    # Extract system prompt and user message
    system = ''
    user_msg = ''
    for msg in messages:
        if msg.get('role') == 'system':
            system = msg.get('content', '')
        elif msg.get('role') == 'user':
            user_msg = msg.get('content', '')

    if not user_msg:
        return JSONResponse({'error': 'No user message'}, status_code=400)

    # Inject Magnitude core rule into system prompt
    system = MAGNITUDE_RULE + '\n\n' + system if system else MAGNITUDE_RULE

    # Call Ollama directly with the messages
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        content = await loop.run_in_executor(executor, call_ollama, user_msg, system, model, max_tokens)

    # Return in OpenAI format
    return {
        'id': f'chatcmpl-{int(time.time())}',
        'object': 'chat.completion',
        'created': int(time.time()),
        'model': model,
        'choices': [{
            'index': 0,
            'message': {'role': 'assistant', 'content': content},
            'finish_reason': 'stop'
        }],
        'usage': {
            'prompt_tokens': len(user_msg) // 4,
            'completion_tokens': len(content) // 4,
            'total_tokens': (len(user_msg) + len(content)) // 4
        }
    }

# ===== Ollama proxy endpoints =====
@app.get('/api/tags')
async def ollama_tags():
    try:
        req = urllib.request.Request(f'{OLLAMA_URL}/api/tags')
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        return {'error': str(e)}

@app.post('/api/generate')
async def ollama_generate(request: Request):
    body = await request.json()
    data = json.dumps(body).encode()
    req = urllib.request.Request(f'{OLLAMA_URL}/api/generate', data=data, method='POST')
    req.add_header('Content-Type', 'application/json')
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read().decode())
        return result

# ===== Book Writer endpoints =====
_book_mgr = None

def get_book_mgr():
    global _book_mgr
    if _book_mgr is None:
        try:
            from soulillusions.book_writer import get_book_manager
            _book_mgr = get_book_manager()
        except Exception as e:
            print(f"Book writer init error: {e}")
            return None
    return _book_mgr

@app.get('/api/books')
async def list_books(category: str = ''):
    mgr = get_book_mgr()
    if not mgr:
        return {'error': 'Book system not available'}
    return {'books': mgr.list_books(category=category)}

@app.post('/api/books/create')
async def create_book(request: Request):
    mgr = get_book_mgr()
    if not mgr:
        return {'error': 'Book system not available'}
    body = await request.json()
    return mgr.create_book(
        title=body.get('title', 'Untitled'),
        description=body.get('description', ''),
        genre=body.get('genre', 'fiction'),
        category=body.get('category', 'original'),
        target_word_count=body.get('target_word_count', 50000),
        tags=body.get('tags', ''),
    )

@app.get('/api/books/{book_id}')
async def get_book(book_id: int):
    mgr = get_book_mgr()
    if not mgr:
        return {'error': 'Book system not available'}
    book = mgr.get_book(book_id)
    if not book:
        return {'error': 'Book not found'}
    chapters = mgr.get_chapters(book_id)
    characters = mgr.get_characters(book_id)
    world_elements = mgr.get_world_elements(book_id)
    return {'book': book, 'chapters': chapters, 'characters': characters, 'world_elements': world_elements}

@app.delete('/api/books/{book_id}')
async def delete_book(book_id: int):
    mgr = get_book_mgr()
    if not mgr:
        return {'error': 'Book system not available'}
    return mgr.delete_book(book_id)

@app.post('/api/books/{book_id}/chapter')
async def add_chapter(book_id: int, request: Request):
    mgr = get_book_mgr()
    if not mgr:
        return {'error': 'Book system not available'}
    body = await request.json()
    return mgr.add_chapter(
        book_id=book_id,
        chapter_number=body.get('chapter_number', 1),
        title=body.get('title', ''),
        content=body.get('content', ''),
        target_word_count=body.get('target_word_count', 3000),
        notes=body.get('notes', ''),
    )

@app.put('/api/books/chapter/{chapter_id}')
async def update_chapter(chapter_id: int, request: Request):
    mgr = get_book_mgr()
    if not mgr:
        return {'error': 'Book system not available'}
    body = await request.json()
    return mgr.update_chapter(
        chapter_id=chapter_id,
        title=body.get('title'),
        content=body.get('content'),
        notes=body.get('notes'),
        target_word_count=body.get('target_word_count'),
    )

@app.post('/api/books/{book_id}/character')
async def add_character(book_id: int, request: Request):
    mgr = get_book_mgr()
    if not mgr:
        return {'error': 'Book system not available'}
    body = await request.json()
    return mgr.add_character(
        book_id=book_id,
        name=body.get('name', ''),
        description=body.get('description', ''),
        role=body.get('role', 'supporting'),
        personality=body.get('personality', ''),
        appearance=body.get('appearance', ''),
        backstory=body.get('backstory', ''),
    )

@app.post('/api/books/{book_id}/write')
async def agent_write_chapter(book_id: int, request: Request):
    mgr = get_book_mgr()
    if not mgr:
        return {'error': 'Book system not available'}
    body = await request.json()
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        result = await loop.run_in_executor(
            executor,
            lambda: asyncio.run(mgr.agent_write_chapter(
                book_id=book_id,
                chapter_number=body.get('chapter_number', 1),
                prompt=body.get('prompt', ''),
                agent_name=body.get('agent_name', 'SoulIllusions'),
            ))
        )
    return result

@app.post('/api/books/chapter/{chapter_id}/continue')
async def agent_continue_chapter(chapter_id: int, request: Request):
    mgr = get_book_mgr()
    if not mgr:
        return {'error': 'Book system not available'}
    body = await request.json()
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        result = await loop.run_in_executor(
            executor,
            lambda: asyncio.run(mgr.agent_continue_chapter(
                chapter_id=chapter_id,
                direction=body.get('direction', ''),
            ))
        )
    return result

@app.post('/api/books/chapter/{chapter_id}/correct')
async def agent_correct_chapter(chapter_id: int):
    mgr = get_book_mgr()
    if not mgr:
        return {'error': 'Book system not available'}
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        result = await loop.run_in_executor(
            executor,
            lambda: asyncio.run(mgr.agent_correct_chapter(chapter_id))
        )
    return result

@app.post('/api/books/{book_id}/audiobook')
async def generate_audiobook(book_id: int, request: Request):
    mgr = get_book_mgr()
    if not mgr:
        return {'error': 'Book system not available'}
    body = await request.json()
    chapter_id = body.get('chapter_id')
    voice = body.get('voice', 'default')
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        if chapter_id:
            result = await loop.run_in_executor(
                executor,
                lambda: asyncio.run(mgr.audiobook_gen.generate_chapter_audio(book_id, chapter_id, voice))
            )
        else:
            result = await loop.run_in_executor(
                executor,
                lambda: asyncio.run(mgr.audiobook_gen.generate_book_audio(book_id, voice))
            )
    return result

@app.get('/api/books/{book_id}/audiobooks')
async def get_audiobooks(book_id: int):
    mgr = get_book_mgr()
    if not mgr:
        return {'error': 'Book system not available'}
    return {'audiobooks': mgr.audiobook_gen.get_audiobooks(book_id)}

@app.post('/api/books/{book_id}/analyze')
async def analyze_book(book_id: int):
    mgr = get_book_mgr()
    if not mgr:
        return {'error': 'Book system not available'}
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        result = await loop.run_in_executor(
            executor,
            lambda: asyncio.run(mgr.analysis_engine.analyze_book(book_id))
        )
    return result

@app.post('/api/books/{book_id}/export')
async def export_book(book_id: int, request: Request):
    mgr = get_book_mgr()
    if not mgr:
        return {'error': 'Book system not available'}
    body = await request.json()
    fmt = body.get('format', 'txt')
    return mgr.export_book(book_id, fmt)

# ===== Web UI =====
WEBUI_PATH = PACKAGE_DIR / 'webui' / 'index.html'

@app.get('/', response_class=HTMLResponse)
async def serve_webui():
    if WEBUI_PATH.exists():
        return HTMLResponse(WEBUI_PATH.read_text(encoding='utf-8'))
    return HTMLResponse('<h1>SoulIllusions Web UI not found</h1>', status_code=404)

@app.get('/webui', response_class=HTMLResponse)
async def serve_webui_alt():
    return await serve_webui()

# ===== Web browsing proxy =====
@app.get('/api/browse')
async def browse_url(url: str = ''):
    if not url:
        return JSONResponse({'error': 'No URL provided'}, status_code=400)
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            content_type = resp.headers.get('Content-Type', 'text/html')
            data = resp.read()
            if 'text' in content_type or 'json' in content_type or 'xml' in content_type:
                return PlainTextResponse(data.decode('utf-8', errors='replace'), media_type=content_type)
            return StreamingResponse(iter([data]), media_type=content_type)
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)

# ===== Ollama models list for web UI =====
@app.get('/api/models')
async def list_models():
    try:
        req = urllib.request.Request(f'{OLLAMA_URL}/api/tags')
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            models = [m['name'] for m in data.get('models', [])]
            return {'models': models}
    except Exception as e:
        return {'models': [DEFAULT_MODEL], 'error': str(e)}

if __name__ == '__main__':
    host = os.environ.get('SOULILLUSIONS_HOST', '0.0.0.0')
    port = int(os.environ.get('SOULILLUSIONS_PORT', '7869'))
    uvicorn.run(app, host=host, port=port, log_level='info')
