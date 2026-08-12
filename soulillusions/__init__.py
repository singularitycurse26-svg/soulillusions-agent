"""
SoulIllusions Agent — Autonomous AI Agent with Web UI
=====================================================
A fully self-contained AI agent system with:
- Local LLM inference via Ollama (dolphin-mistral, uncensored)
- Web UI with chat, Jarvis voice, browser, books, and agent control
- Book writer with AI-assisted writing, audiobook generation, and analysis
- Persistent chat history and projects (SQLite)
- Autonomous ReAct loop with file-centric state management
- OpenAI-compatible API endpoint

Usage:
    pip install soulillusions-agent
    soulillusions serve    # Start the web UI server
    soulillusions start    # Start the autonomous agent
    soulillusions status   # Check agent status
"""

__version__ = "1.0.0"
__author__ = "SoulIllusions"

from soulillusions.agent import get_agent, load_config, save_config
