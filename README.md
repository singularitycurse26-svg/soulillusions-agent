# SoulIllusions Agent

Autonomous AI agent with a full web UI — chat, voice assistant, book writer, audiobook generator, and autonomous task execution. Runs entirely locally with Ollama. No API keys, no cloud, no VPS required.

## Quick Start

```bash
# 1. Install Ollama (if not already installed)
#    Windows: https://ollama.com/download
#    Linux:   curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull the model
ollama pull dolphin-mistral

# 3. Install SoulIllusions
pip install git+https://github.com/YOUR_USERNAME/soulillusions-agent.git

# 4. Start the web UI
soulillusions serve

# 5. Open browser
#    http://localhost:7869
```

## Features

### Chat
- Persistent conversation history (survives restarts)
- Projects to organize conversations (like OpenAI)
- Powered by dolphin-mistral (uncensored) via Ollama
- OpenAI-compatible API endpoint at `/v1/chat/completions`

### Jarvis Voice Assistant
- Speech recognition in browser
- Text-to-speech responses
- Wake word "Jarvis" activation
- Continuous listening mode

### Web Browser
- Built-in browser panel with iframe navigation
- Back/forward/refresh controls
- URL bar with history

### Book Writer
- Create books with chapters, characters, and world elements
- AI-assisted chapter writing with dolphin-mistral
- Continue chapters with AI direction
- Spell and grammar correction
- Audiobook generation via TTS (pyttsx3 or espeak)
- Book analysis engine for text-to-video and text-to-game pipelines
- Export to TXT, HTML, or JSON

### Agent Control
- Autonomous ReAct loop (Reason → Act → Observe → repeat)
- File-centric state management (InfiAgent architecture)
- Full system access: shell, files, Python execution
- Project management: create, complete, improve
- Persistent memory via SQLite
- Sub-agents for specialized tasks (coder, trader, researcher, browser)
- Configurable context refresh interval

### Magnitude Core Rule
Every LLM response follows the production-quality rule:
> Do not generate scaffolding or placeholder implementations. Generate fully implemented, production-quality modules with real algorithms, comprehensive error handling, logging, configuration, testing, and documentation. A module is not considered complete until every public method performs its intended function under realistic conditions.

## CLI Commands

```bash
soulillusions serve          # Start web UI server
soulillusions start          # Start autonomous agent
soulillusions stop           # Stop agent
soulillusions status         # Show agent status
soulillusions prompt <text>  # Send direct prompt to LLM
soulillusions goal <text>    # Set a goal for the agent
soulillusions projects       # List projects
soulillusions memories       # List memories
soulillusions actions        # Show recent actions
soulillusions config         # Show configuration
soulillusions install        # Install Ollama + pull model
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SOULILLUSIONS_DATA` | `~/.soulillusions` | Data directory for DBs, books, workspace |
| `SOULILLUSIONS_HOST` | `0.0.0.0` | Server bind host |
| `SOULILLUSIONS_PORT` | `7869` | Server port |
| `SOULILLUSIONS_MODEL` | `dolphin-mistral:latest` | Default LLM model |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama API URL |

## Data Storage

All data is stored in `~/.soulillusions/` (or the directory set by `SOULILLUSIONS_DATA`):

- `agent.db` — Agent sessions, actions, projects, memory, skills
- `chat_storage.db` — Conversations, messages, projects
- `books.db` — Books, chapters, characters, world elements
- `agent_config.json` — Agent configuration
- `agent_workspace/` — Agent workspace (projects, memory, logs)
- `book_library/` — Book exports and audiobooks

## Architecture

```
soulillusions/
├── __init__.py          # Package init
├── cli.py               # CLI entry point (soulillusions command)
├── server.py            # FastAPI web server + all API endpoints
├── agent.py             # SoulIllusionsAgent — autonomous ReAct loop
├── book_writer.py       # Book writer, audiobook generator, analysis engine
├── default_config.json  # Default agent configuration
└── webui/
    └── index.html       # Full web UI (chat, Jarvis, browser, books, agent)
```

## Requirements

- Python 3.9+
- [Ollama](https://ollama.com) with at least one model installed
- Recommended model: `dolphin-mistral` (uncensored, 7B parameters)
- For audiobook TTS: `pip install soulillusions-agent[tts]` (installs pyttsx3)

## License

MIT
