#!/usr/bin/env python3
"""SoulIllusions CLI entry point."""
import sys
import os


def main():
    if len(sys.argv) < 2:
        print("""
SoulIllusions Agent — Autonomous AI Agent with Web UI
=====================================================
Version: 1.0.0

Commands:
  serve          Start the web UI server (http://localhost:7869)
  start          Start the autonomous agent loop
  stop           Stop the running agent
  status         Show agent status
  prompt <text>  Send a direct prompt to the LLM
  goal <text>    Set a goal for the agent
  projects       List all projects
  memories       List memory items
  actions        Show recent agent actions
  config         Show current configuration
  install        Install Ollama and pull dolphin-mistral model

Environment Variables:
  SOULILLUSIONS_DATA    Data directory (default: ~/.soulillusions)
  SOULILLUSIONS_HOST    Server host (default: 0.0.0.0)
  SOULILLUSIONS_PORT    Server port (default: 7869)
  SOULILLUSIONS_MODEL   Default LLM model (default: dolphin-mistral:latest)
  OLLAMA_URL            Ollama API URL (default: http://localhost:11434)

Quick Start:
  1. Install Ollama:     curl -fsSL https://ollama.com/install.sh | sh
  2. Pull model:         ollama pull dolphin-mistral
  3. Start SoulIllusions: soulillusions serve
  4. Open browser:       http://localhost:7869
""")
        return

    cmd = sys.argv[1]

    if cmd == 'serve':
        from soulillusions.server import app
        import uvicorn
        host = os.environ.get('SOULILLUSIONS_HOST', '0.0.0.0')
        port = int(os.environ.get('SOULILLUSIONS_PORT', '7869'))
        print(f"SoulIllusions starting on http://{host}:{port}")
        print(f"Data directory: {os.environ.get('SOULILLUSIONS_DATA', os.path.expanduser('~/.soulillusions'))}")
        print(f"Ollama URL: {os.environ.get('OLLAMA_URL', 'http://localhost:11434')}")
        print(f"Model: {os.environ.get('SOULILLUSIONS_MODEL', 'dolphin-mistral:latest')}")
        print()
        uvicorn.run(app, host=host, port=port, log_level='info')

    elif cmd == 'install':
        print("Installing Ollama and pulling dolphin-mistral model...")
        import subprocess
        if sys.platform == 'win32':
            print("On Windows, please install Ollama from: https://ollama.com/download")
            print("Then run: ollama pull dolphin-mistral")
        else:
            subprocess.run(['curl', '-fsSL', 'https://ollama.com/install.sh', '|', 'sh'], shell=False)
            subprocess.run(['ollama', 'pull', 'dolphin-mistral'])
        print("Done! Now run: soulillusions serve")

    else:
        from soulillusions.agent import get_agent
        agent = get_agent()

        if cmd == 'start':
            result = agent.start()
            print(f"Agent started: {result}")
            if result.get("status") == "started":
                print("\nAgent running in background. Press Ctrl+C to stop.")
                import time
                try:
                    while True:
                        time.sleep(10)
                        status = agent.get_status()
                        actions = status.get('session', {}).get('actions', 0) if status.get('session') else 0
                        print(f"[{time.strftime('%H:%M:%S')}] Actions: {actions}")
                except KeyboardInterrupt:
                    agent.stop()
                    print("Agent stopped.")

        elif cmd == 'stop':
            result = agent.stop()
            print(f"Agent: {result}")

        elif cmd == 'status':
            import json
            result = agent.get_status()
            print(json.dumps(result, indent=2))

        elif cmd == 'prompt':
            prompt = " ".join(sys.argv[2:])
            if not prompt:
                print("Usage: soulillusions prompt <text>")
                return
            result = agent.send_prompt(prompt)
            print(result.get("response", ""))

        elif cmd == 'goal':
            goal = " ".join(sys.argv[2:])
            if not goal:
                print("Usage: soulillusions goal <text>")
                return
            import json
            result = agent.set_goal(goal)
            print(json.dumps(result, indent=2))

        elif cmd == 'projects':
            import json
            result = agent.get_projects()
            print(json.dumps(result, indent=2))

        elif cmd == 'memories':
            import json
            result = agent.get_memories()
            print(json.dumps(result, indent=2))

        elif cmd == 'actions':
            import json
            result = agent.get_actions()
            print(json.dumps(result, indent=2))

        elif cmd == 'config':
            import json
            cfg = load_config()
            print(json.dumps(cfg, indent=2))

        else:
            print(f"Unknown command: {cmd}")
            print("Run 'soulillusions' for help.")


if __name__ == '__main__':
    main()
