"""
SoulIllusions Book Writer System
=================================
Complete book writing, storage, audiobook, and analysis system.

Features:
- Book creation with chapters, scenes, characters, and world-building
- AI-assisted writing (uses LLM from soulillusions_agent.py)
- Agent integration: both SoulIllusions Agent and Prime Agent can help write
- Spell checking and chapter length management
- Audiobook generation via TTS (text-to-speech)
- Book library with search, categories, and ratings
- Book analysis engine: extracts detailed data for text-to-video and text-to-game
- Feeds book data to video and game generation pipelines
- Export to multiple formats (TXT, HTML, PDF-ready, EPUB-ready)

Architecture:
- SQLite database for books, chapters, characters, scenes
- File storage for audiobook audio files
- LLM interface for AI-assisted writing
- Analysis engine that produces structured data for downstream pipelines
"""

import os, sys, json, time, re, asyncio, sqlite3, subprocess, hashlib
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field, asdict

_DATA_DIR = Path(os.environ.get('SOULILLUSIONS_DATA', str(Path.home() / '.soulillusions')))
_DATA_DIR.mkdir(parents=True, exist_ok=True)

SCRIPT_DIR = _DATA_DIR
BOOKS_DB = _DATA_DIR / "books.db"
BOOKS_DIR = _DATA_DIR / "book_library"
AUDIOBOOK_DIR = BOOKS_DIR / "audiobooks"
EXPORT_DIR = BOOKS_DIR / "exports"

BOOKS_DIR.mkdir(exist_ok=True, parents=True)
AUDIOBOOK_DIR.mkdir(exist_ok=True, parents=True)
EXPORT_DIR.mkdir(exist_ok=True, parents=True)


def init_db():
    conn = sqlite3.connect(str(BOOKS_DB))
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS books (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            author TEXT DEFAULT 'SoulIllusions',
            description TEXT,
            genre TEXT DEFAULT 'fiction',
            category TEXT DEFAULT 'original',
            status TEXT DEFAULT 'draft',
            word_count INTEGER DEFAULT 0,
            chapter_count INTEGER DEFAULT 0,
            target_word_count INTEGER DEFAULT 50000,
            rating INTEGER DEFAULT 0,
            tags TEXT,
            cover_description TEXT,
            source_project TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now')),
            completed_at TEXT,
            metadata TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS chapters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            book_id INTEGER NOT NULL,
            chapter_number INTEGER NOT NULL,
            title TEXT,
            content TEXT,
            word_count INTEGER DEFAULT 0,
            target_word_count INTEGER DEFAULT 3000,
            status TEXT DEFAULT 'draft',
            notes TEXT,
            agent_assisted INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (book_id) REFERENCES books(id) ON DELETE CASCADE
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS characters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            book_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            description TEXT,
            role TEXT DEFAULT 'supporting',
            personality TEXT,
            appearance TEXT,
            backstory TEXT,
            relationships TEXT,
            arc TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (book_id) REFERENCES books(id) ON DELETE CASCADE
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS world_elements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            book_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            type TEXT DEFAULT 'location',
            description TEXT,
            importance TEXT DEFAULT 'minor',
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (book_id) REFERENCES books(id) ON DELETE CASCADE
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS audiobooks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            book_id INTEGER NOT NULL,
            chapter_id INTEGER,
            audio_path TEXT,
            voice TEXT DEFAULT 'default',
            duration_seconds INTEGER DEFAULT 0,
            status TEXT DEFAULT 'pending',
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (book_id) REFERENCES books(id) ON DELETE CASCADE
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS book_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            book_id INTEGER NOT NULL,
            analysis_type TEXT NOT NULL,
            analysis_data TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (book_id) REFERENCES books(id) ON DELETE CASCADE
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS agent_writing_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            book_id INTEGER NOT NULL,
            chapter_id INTEGER,
            agent_name TEXT,
            prompt TEXT,
            response TEXT,
            words_written INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (book_id) REFERENCES books(id) ON DELETE CASCADE
        )
    """)
    conn.commit()
    conn.close()


init_db()


# --- LLM Interface (reuse from agent) ---
def _get_llm():
    try:
        from soulillusions.agent import LLMInterface, load_config
        cfg = load_config()
        return LLMInterface(cfg)
    except Exception:
        return None


# --- Spell Check ---
def spell_check(text: str) -> List[Dict]:
    """Basic spell check using available tools."""
    errors = []
    try:
        import subprocess as sp
        result = sp.run(["python", "-m", "spellchecker", "--check", text],
                       capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            errors = data.get("errors", [])
    except Exception:
        pass
    
    if not errors:
        # Fallback: basic common misspelling check
        common_misspellings = {
            "teh": "the", "adn": "and", "nad": "and", "thier": "their",
            "recieve": "receive", "seperate": "separate", "definately": "definitely",
            "occured": "occurred", "untill": "until", "wich": "which",
            "thru": "through", "tho": "though", "alot": "a lot",
            "wont": "won't", "dont": "don't", "cant": "can't",
            "couldnt": "couldn't", "wouldnt": "wouldn't", "shouldnt": "shouldn't",
            "isnt": "isn't", "wasnt": "wasn't", "havent": "haven't",
            "didnt": "didn't", "doesnt": "doesn't",
        }
        words = re.findall(r'\b\w+\b', text.lower())
        for word in words:
            if word in common_misspellings:
                errors.append({
                    "word": word,
                    "suggestion": common_misspellings[word],
                    "type": "common_misspelling"
                })
    return errors


# --- Word Count Helpers ---
def count_words(text: str) -> int:
    return len(re.findall(r'\b\w+\b', text))

def chapter_length_status(word_count: int, target: int = 3000) -> str:
    if word_count == 0:
        return "empty"
    ratio = word_count / target
    if ratio < 0.3:
        return "too_short"
    elif ratio < 0.7:
        return "short"
    elif ratio <= 1.3:
        return "good"
    elif ratio <= 2.0:
        return "long"
    else:
        return "too_long"


# --- Book Analysis Engine ---
class BookAnalysisEngine:
    """Analyzes books to extract structured data for text-to-video and text-to-game pipelines."""
    
    def __init__(self):
        self.llm = _get_llm()
    
    async def analyze_book(self, book_id: int) -> Dict:
        """Full book analysis: characters, plot, scenes, visual descriptions, game elements."""
        book = self._get_book(book_id)
        if not book:
            return {"error": "Book not found"}
        
        chapters = self._get_chapters(book_id)
        characters = self._get_characters(book_id)
        world_elements = self._get_world_elements(book_id)
        
        full_text = "\n\n".join([f"Chapter {ch['chapter_number']}: {ch['title']}\n{ch['content']}" 
                                 for ch in chapters if ch.get("content")])
        
        analysis = {
            "book_id": book_id,
            "title": book["title"],
            "genre": book["genre"],
            "total_words": book["word_count"],
            "chapter_count": len(chapters),
            "analysis_time": datetime.now().isoformat(),
        }
        
        # Character analysis for visual casting
        analysis["characters"] = self._analyze_characters(characters)
        
        # Plot structure for video/game scenes
        analysis["plot_structure"] = self._analyze_plot(chapters)
        
        # Visual elements for text-to-video
        analysis["visual_elements"] = self._extract_visual_descriptions(full_text, chapters)
        
        # Game adaptation data
        analysis["game_adaptation"] = self._extract_game_elements(full_text, characters, world_elements)
        
        # Scene breakdown for video generation
        analysis["scenes"] = self._breakdown_scenes(chapters)
        
        # Themes and mood
        analysis["themes"] = self._extract_themes(full_text)
        
        # Save analysis to DB
        self._save_analysis(book_id, "full_analysis", analysis)
        
        return analysis
    
    def _get_book(self, book_id: int) -> Optional[Dict]:
        conn = sqlite3.connect(str(BOOKS_DB))
        c = conn.cursor()
        c.execute("SELECT * FROM books WHERE id = ?", (book_id,))
        row = c.fetchone()
        conn.close()
        if not row:
            return None
        cols = ["id", "title", "author", "description", "genre", "category", "status",
                "word_count", "chapter_count", "target_word_count", "rating", "tags",
                "cover_description", "source_project", "created_at", "updated_at",
                "completed_at", "metadata"]
        return dict(zip(cols, row))
    
    def _get_chapters(self, book_id: int) -> List[Dict]:
        conn = sqlite3.connect(str(BOOKS_DB))
        c = conn.cursor()
        c.execute("SELECT * FROM chapters WHERE book_id = ? ORDER BY chapter_number", (book_id,))
        rows = c.fetchall()
        conn.close()
        cols = ["id", "book_id", "chapter_number", "title", "content", "word_count",
                "target_word_count", "status", "notes", "agent_assisted", "created_at", "updated_at"]
        return [dict(zip(cols, r)) for r in rows]
    
    def _get_characters(self, book_id: int) -> List[Dict]:
        conn = sqlite3.connect(str(BOOKS_DB))
        c = conn.cursor()
        c.execute("SELECT * FROM characters WHERE book_id = ?", (book_id,))
        rows = c.fetchall()
        conn.close()
        cols = ["id", "book_id", "name", "description", "role", "personality",
                "appearance", "backstory", "relationships", "arc", "created_at"]
        return [dict(zip(cols, r)) for r in rows]
    
    def _get_world_elements(self, book_id: int) -> List[Dict]:
        conn = sqlite3.connect(str(BOOKS_DB))
        c = conn.cursor()
        c.execute("SELECT * FROM world_elements WHERE book_id = ?", (book_id,))
        rows = c.fetchall()
        conn.close()
        cols = ["id", "book_id", "name", "type", "description", "importance", "created_at"]
        return [dict(zip(cols, r)) for r in rows]
    
    def _analyze_characters(self, characters: List[Dict]) -> List[Dict]:
        """Extract visual and personality data for character casting in video/game."""
        result = []
        for ch in characters:
            result.append({
                "name": ch["name"],
                "role": ch["role"],
                "appearance": ch.get("appearance", ""),
                "personality": ch.get("personality", ""),
                "backstory": ch.get("backstory", ""),
                "visual_prompt": self._character_to_visual_prompt(ch),
                "game_role": self._character_to_game_role(ch),
                "relationships": ch.get("relationships", ""),
                "arc": ch.get("arc", ""),
            })
        return result
    
    def _character_to_visual_prompt(self, ch: Dict) -> str:
        """Convert character data to a text-to-image prompt for video generation."""
        parts = []
        if ch.get("appearance"):
            parts.append(ch["appearance"])
        if ch.get("personality"):
            mood = "confident" if "strong" in ch["personality"].lower() else "neutral"
            parts.append(f"{mood} expression")
        parts.append("detailed, cinematic lighting, professional character design")
        return ", ".join(parts)
    
    def _character_to_game_role(self, ch: Dict) -> str:
        """Map character role to game archetype."""
        role = ch.get("role", "supporting").lower()
        if "protagon" in role or "main" in role:
            return "player"
        elif "antagon" in role or "villain" in role:
            return "boss"
        elif "mentor" in role or "guide" in role:
            return "npc_guide"
        elif "love" in role:
            return "npc_ally"
        else:
            return "npc"
    
    def _analyze_plot(self, chapters: List[Dict]) -> Dict:
        """Extract three-act structure and key plot points."""
        total = len(chapters)
        if total == 0:
            return {"acts": [], "plot_points": []}
        
        act1_end = max(1, total // 4)
        act2_end = max(act1_end + 1, total * 3 // 4)
        
        return {
            "acts": [
                {
                    "name": "Act 1: Setup",
                    "chapters": list(range(1, act1_end + 1)),
                    "summary": chapters[0].get("title", "") if chapters else "",
                },
                {
                    "name": "Act 2: Confrontation",
                    "chapters": list(range(act1_end + 1, act2_end + 1)),
                    "summary": chapters[act1_end].get("title", "") if act1_end < total else "",
                },
                {
                    "name": "Act 3: Resolution",
                    "chapters": list(range(act2_end + 1, total + 1)),
                    "summary": chapters[act2_end].get("title", "") if act2_end < total else "",
                },
            ],
            "plot_points": [ch.get("title", f"Chapter {ch['chapter_number']}") for ch in chapters],
            "total_chapters": total,
        }
    
    def _extract_visual_descriptions(self, full_text: str, chapters: List[Dict]) -> List[Dict]:
        """Extract scenes with visual descriptions for text-to-video pipeline."""
        scenes = []
        visual_keywords = ["saw", "looked", "appeared", "stood", "sat", "walked",
                          "ran", "flew", "glowed", "shone", "dark", "bright",
                          "forest", "city", "castle", "ocean", "mountain", "sky",
                          "rain", "sun", "moon", "fire", "snow", "desert"]
        
        for ch in chapters:
            if not ch.get("content"):
                continue
            paragraphs = ch["content"].split("\n\n")
            for i, para in enumerate(paragraphs):
                para_lower = para.lower()
                has_visual = any(kw in para_lower for kw in visual_keywords)
                if has_visual and len(para) > 50:
                    scenes.append({
                        "chapter": ch["chapter_number"],
                        "paragraph": i,
                        "text": para[:500],
                        "visual_prompt": self._text_to_video_prompt(para, ch),
                    })
        return scenes
    
    def _text_to_video_prompt(self, text: str, chapter: Dict) -> str:
        """Convert book text to a text-to-video generation prompt."""
        # Extract key visual elements
        sentences = text.split(". ")
        visual_sentences = [s for s in sentences if len(s) > 20][:3]
        prompt = ". ".join(visual_sentences)
        prompt = re.sub(r'[^\w\s,.!?-]', '', prompt)
        prompt = prompt[:300]  # Keep it concise for video generation
        return f"{prompt}. Cinematic, detailed, atmospheric, high quality."
    
    def _extract_game_elements(self, full_text: str, characters: List[Dict], 
                                world_elements: List[Dict]) -> Dict:
        """Extract game adaptation data from book."""
        return {
            "player_character": next((ch["name"] for ch in characters 
                                      if "protagon" in ch.get("role", "").lower() or 
                                      "main" in ch.get("role", "").lower()), 
                                     characters[0]["name"] if characters else "Hero"),
            "antagonist": next((ch["name"] for ch in characters 
                               if "antagon" in ch.get("role", "").lower() or 
                               "villain" in ch.get("role", "").lower()), None),
            "locations": [{"name": we["name"], "description": we["description"]} 
                          for we in world_elements if we["type"] == "location"],
            "items": [{"name": we["name"], "description": we["description"]} 
                      for we in world_elements if we["type"] == "item"],
            "suggested_genre": self._suggest_game_genre(full_text),
            "key_moments": self._extract_key_moments(full_text),
        }
    
    def _suggest_game_genre(self, text: str) -> str:
        """Suggest game genre based on book content."""
        text_lower = text.lower()
        scores = {
            "shooter": sum(1 for kw in ["gun", "shoot", "war", "battle", "army", "soldier"] if kw in text_lower),
            "platformer": sum(1 for kw in ["jump", "climb", "run", "escape", "chase"] if kw in text_lower),
            "rpg": sum(1 for kw in ["magic", "quest", "kingdom", "dragon", "sword", "spell", "wizard"] if kw in text_lower),
            "puzzle": sum(1 for kw in ["riddle", "puzzle", "mystery", "solve", "clue", "secret"] if kw in text_lower),
            "racing": sum(1 for kw in ["race", "speed", "drive", "car", "fast", "chase"] if kw in text_lower),
            "strategy": sum(1 for kw in ["army", "kingdom", "war", "plan", "tactic", "command"] if kw in text_lower),
        }
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "adventure"
    
    def _extract_key_moments(self, text: str) -> List[str]:
        """Extract key dramatic moments for game levels/scenes."""
        moments = []
        sentences = text.split(". ")
        drama_keywords = ["suddenly", "but then", "however", "shock", "surprise",
                         "betray", "discover", "realize", "fight", "escape", "death"]
        for s in sentences:
            if any(kw in s.lower() for kw in drama_keywords) and len(s) > 30:
                moments.append(s.strip()[:200])
        return moments[:20]
    
    def _breakdown_scenes(self, chapters: List[Dict]) -> List[Dict]:
        """Break chapters into scenes for video generation."""
        scenes = []
        for ch in chapters:
            if not ch.get("content"):
                continue
            paragraphs = ch["content"].split("\n\n")
            for i, para in enumerate(paragraphs):
                if len(para.strip()) > 100:
                    scenes.append({
                        "chapter": ch["chapter_number"],
                        "scene_number": i + 1,
                        "text": para[:500],
                        "word_count": count_words(para),
                        "suggested_duration": min(10, max(3, count_words(para) // 50)),
                    })
        return scenes
    
    def _extract_themes(self, text: str) -> List[str]:
        """Extract major themes from the book."""
        text_lower = text.lower()
        theme_keywords = {
            "love": ["love", "romance", "heart", "passion", "beloved"],
            "adventure": ["adventure", "journey", "quest", "explore", "discover"],
            "conflict": ["war", "battle", "fight", "conflict", "enemy", "versus"],
            "mystery": ["mystery", "secret", "hidden", "unknown", "riddle"],
            "coming_of_age": ["grew", "learned", "changed", "mature", "understood"],
            "redemption": ["redeem", "forgive", "aton", "second chance"],
            "survival": ["survive", "escape", "flee", "endure", "persist"],
            "power": ["power", "throne", "kingdom", "rule", "control", "dominate"],
        }
        themes = []
        for theme, keywords in theme_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 2:
                themes.append(theme)
        return themes
    
    def _save_analysis(self, book_id: int, analysis_type: str, data: dict):
        conn = sqlite3.connect(str(BOOKS_DB))
        c = conn.cursor()
        c.execute("INSERT INTO book_analysis (book_id, analysis_type, analysis_data) VALUES (?, ?, ?)",
                  (book_id, analysis_type, json.dumps(data)))
        conn.commit()
        conn.close()
    
    def get_analysis(self, book_id: int) -> Optional[Dict]:
        conn = sqlite3.connect(str(BOOKS_DB))
        c = conn.cursor()
        c.execute("SELECT analysis_data FROM book_analysis WHERE book_id = ? AND analysis_type = 'full_analysis' ORDER BY created_at DESC LIMIT 1", (book_id,))
        row = c.fetchone()
        conn.close()
        if row:
            return json.loads(row[0])
        return None


# --- TTS / Audiobook Generation ---
class AudiobookGenerator:
    """Generates audiobooks from book chapters using TTS."""
    
    def __init__(self):
        self.output_dir = AUDIOBOOK_DIR
    
    async def generate_chapter_audio(self, book_id: int, chapter_id: int, 
                                      voice: str = "default") -> Dict:
        """Generate audio for a single chapter."""
        conn = sqlite3.connect(str(BOOKS_DB))
        c = conn.cursor()
        c.execute("SELECT title, content FROM chapters WHERE id = ? AND book_id = ?", 
                  (chapter_id, book_id))
        row = c.fetchone()
        if not row:
            conn.close()
            return {"error": "Chapter not found"}
        
        title, content = row
        if not content:
            conn.close()
            return {"error": "Chapter has no content"}
        
        # Try to use available TTS
        audio_path = await self._tts(content, book_id, chapter_id, voice)
        
        if audio_path:
            c.execute("""INSERT INTO audiobooks (book_id, chapter_id, audio_path, voice, status)
                        VALUES (?, ?, ?, ?, 'completed')""",
                      (book_id, chapter_id, str(audio_path), voice))
            conn.commit()
            conn.close()
            return {"status": "completed", "audio_path": str(audio_path), "chapter_title": title}
        else:
            conn.close()
            return {"error": "TTS not available. Install pyttsx3 or espeak."}
    
    async def generate_book_audio(self, book_id: int, voice: str = "default") -> Dict:
        """Generate audio for all chapters in a book."""
        conn = sqlite3.connect(str(BOOKS_DB))
        c = conn.cursor()
        c.execute("SELECT id, title FROM chapters WHERE book_id = ? ORDER BY chapter_number", (book_id,))
        chapters = c.fetchall()
        conn.close()
        
        results = []
        for ch_id, ch_title in chapters:
            result = await self.generate_chapter_audio(book_id, ch_id, voice)
            results.append({"chapter": ch_title, **result})
        
        return {"status": "completed", "chapters": results, "total": len(results)}
    
    async def _tts(self, text: str, book_id: int, chapter_id: int, voice: str) -> Optional[Path]:
        """Generate speech from text using available TTS engine."""
        audio_file = self.output_dir / f"book_{book_id}_ch_{chapter_id}.mp3"
        
        # Try pyttsx3 (offline, free)
        try:
            import pyttsx3
            engine = pyttsx3.init()
            if voice != "default":
                voices = engine.getProperty('voices')
                for v in voices:
                    if voice.lower() in v.name.lower():
                        engine.setProperty('voice', v.id)
                        break
            engine.save_to_file(text, str(audio_file))
            engine.runAndWait()
            if audio_file.exists():
                return audio_file
        except ImportError:
            pass
        
        # Try espeak (Linux/Mac, free)
        try:
            result = subprocess.run(
                ["espeak", "-s", "160", "-w", str(audio_file), text[:3000]],
                capture_output=True, timeout=60
            )
            if audio_file.exists():
                return audio_file
        except Exception:
            pass
        
        # Try GPU backend TTS
        try:
            import urllib.request
            from soulillusions.agent import load_config
            cfg = load_config()
            server_url = cfg.get("server_inference_url", "")
            if not server_url:
                server_cfg = SCRIPT_DIR / "server_config.json"
                if server_cfg.exists():
                    sc = json.loads(server_cfg.read_text())
                    server_url = sc.get("gpu_backend_url", "")
            if server_url:
                payload = json.dumps({"text": text, "voice": voice}).encode()
                req = urllib.request.Request(f"{server_url}/api/tts", data=payload,
                                             headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(req, timeout=120) as resp:
                    result = json.loads(resp.read().decode())
                    if result.get("audio_base64"):
                        import base64
                        audio_file.write_bytes(base64.b64decode(result["audio_base64"]))
                        return audio_file
        except Exception:
            pass
        
        return None
    
    def get_audiobooks(self, book_id: int) -> List[Dict]:
        conn = sqlite3.connect(str(BOOKS_DB))
        c = conn.cursor()
        c.execute("""SELECT a.id, a.book_id, a.chapter_id, a.audio_path, a.voice, 
                     a.duration_seconds, a.status, a.created_at, ch.title as chapter_title
                     FROM audiobooks a LEFT JOIN chapters ch ON a.chapter_id = ch.id
                     WHERE a.book_id = ? ORDER BY ch.chapter_number""", (book_id,))
        rows = c.fetchall()
        conn.close()
        cols = ["id", "book_id", "chapter_id", "audio_path", "voice",
                "duration_seconds", "status", "created_at", "chapter_title"]
        return [dict(zip(cols, r)) for r in rows]


# --- Book Manager ---
class BookManager:
    """Main book management system."""
    
    def __init__(self):
        self.llm = _get_llm()
        self.analysis_engine = BookAnalysisEngine()
        self.audiobook_gen = AudiobookGenerator()
    
    def create_book(self, title: str, description: str = "", genre: str = "fiction",
                    category: str = "original", target_word_count: int = 50000,
                    tags: str = "", source_project: str = "") -> Dict:
        conn = sqlite3.connect(str(BOOKS_DB))
        c = conn.cursor()
        c.execute("""INSERT INTO books (title, description, genre, category, target_word_count, tags, source_project)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                  (title, description, genre, category, target_word_count, tags, source_project))
        book_id = c.lastrowid
        conn.commit()
        conn.close()
        return {"id": book_id, "title": title, "status": "created"}
    
    def add_chapter(self, book_id: int, chapter_number: int, title: str = "",
                    content: str = "", target_word_count: int = 3000,
                    notes: str = "") -> Dict:
        wc = count_words(content)
        conn = sqlite3.connect(str(BOOKS_DB))
        c = conn.cursor()
        c.execute("""INSERT INTO chapters (book_id, chapter_number, title, content, word_count, target_word_count, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                  (book_id, chapter_number, title, content, wc, target_word_count, notes))
        chapter_id = c.lastrowid
        self._update_book_stats(book_id)
        conn.commit()
        conn.close()
        return {"id": chapter_id, "book_id": book_id, "word_count": wc,
                "length_status": chapter_length_status(wc, target_word_count)}
    
    def update_chapter(self, chapter_id: int, title: str = None, content: str = None,
                       notes: str = None, target_word_count: int = None) -> Dict:
        conn = sqlite3.connect(str(BOOKS_DB))
        c = conn.cursor()
        updates = []
        params = []
        if title is not None:
            updates.append("title = ?")
            params.append(title)
        if content is not None:
            updates.append("content = ?")
            params.append(content)
            updates.append("word_count = ?")
            params.append(count_words(content))
        if notes is not None:
            updates.append("notes = ?")
            params.append(notes)
        if target_word_count is not None:
            updates.append("target_word_count = ?")
            params.append(target_word_count)
        updates.append("updated_at = datetime('now')")
        params.append(chapter_id)
        
        c.execute(f"UPDATE chapters SET {', '.join(updates)} WHERE id = ?", params)
        
        c.execute("SELECT book_id, word_count, target_word_count FROM chapters WHERE id = ?", (chapter_id,))
        row = c.fetchone()
        if row:
            self._update_book_stats(row[0])
            wc = row[1] if content is None else count_words(content)
            twc = row[2] if target_word_count is None else target_word_count
            conn.commit()
            conn.close()
            return {"id": chapter_id, "word_count": wc, "length_status": chapter_length_status(wc, twc)}
        conn.close()
        return {"error": "Chapter not found"}
    
    def add_character(self, book_id: int, name: str, description: str = "",
                      role: str = "supporting", personality: str = "",
                      appearance: str = "", backstory: str = "",
                      relationships: str = "", arc: str = "") -> Dict:
        conn = sqlite3.connect(str(BOOKS_DB))
        c = conn.cursor()
        c.execute("""INSERT INTO characters (book_id, name, description, role, personality, appearance, backstory, relationships, arc)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                  (book_id, name, description, role, personality, appearance, backstory, relationships, arc))
        char_id = c.lastrowid
        conn.commit()
        conn.close()
        return {"id": char_id, "name": name, "role": role}
    
    def add_world_element(self, book_id: int, name: str, type: str = "location",
                          description: str = "", importance: str = "minor") -> Dict:
        conn = sqlite3.connect(str(BOOKS_DB))
        c = conn.cursor()
        c.execute("""INSERT INTO world_elements (book_id, name, type, description, importance)
                    VALUES (?, ?, ?, ?, ?)""",
                  (book_id, name, type, description, importance))
        elem_id = c.lastrowid
        conn.commit()
        conn.close()
        return {"id": elem_id, "name": name, "type": type}
    
    async def agent_write_chapter(self, book_id: int, chapter_number: int,
                                   prompt: str, agent_name: str = "SoulIllusions Agent",
                                   context_chapters: int = 1) -> Dict:
        """Have an AI agent write or continue a chapter."""
        if not self.llm:
            return {"error": "LLM not available"}
        
        # Get context from previous chapters
        conn = sqlite3.connect(str(BOOKS_DB))
        c = conn.cursor()
        c.execute("SELECT title, content FROM chapters WHERE book_id = ? AND chapter_number < ? ORDER BY chapter_number DESC LIMIT ?",
                  (book_id, chapter_number, context_chapters))
        prev_chapters = c.fetchall()
        c.execute("SELECT title, description, genre FROM books WHERE id = ?", (book_id,))
        book = c.fetchone()
        c.execute("SELECT name, role, personality, appearance FROM characters WHERE book_id = ?", (book_id,))
        characters = c.fetchall()
        conn.close()
        
        # Build context
        context_parts = []
        if book:
            context_parts.append(f"Book: {book[0]}")
            context_parts.append(f"Genre: {book[2]}")
            context_parts.append(f"Description: {book[1]}")
        if characters:
            char_summary = ", ".join([f"{ch[0]} ({ch[1]})" for ch in characters])
            context_parts.append(f"Characters: {char_summary}")
        if prev_chapters:
            for title, content in prev_chapters:
                context_parts.append(f"Previous chapter '{title}': {content[-1000:] if content else '[empty]'}")
        
        context = "\n".join(context_parts)
        
        system_prompt = f"""You are a professional book writer AI assisting with a novel. 
Write compelling, detailed prose with proper spelling and grammar.
Maintain consistency with previous chapters and character personalities.
Write at least 2000 words for this chapter.
Context:
{context}

User request: {prompt}

Write the chapter content now. Use vivid descriptions, natural dialogue, and engaging narrative."""
        
        response = await self.llm.generate(prompt, system_prompt, max_tokens=8000)
        
        wc = count_words(response)
        
        # Save as writing session
        conn = sqlite3.connect(str(BOOKS_DB))
        c = conn.cursor()
        c.execute("""INSERT INTO agent_writing_sessions (book_id, chapter_id, agent_name, prompt, response, words_written)
                    VALUES (?, NULL, ?, ?, ?, ?)""",
                  (book_id, agent_name, prompt, response, wc))
        conn.commit()
        conn.close()
        
        return {
            "content": response,
            "word_count": wc,
            "agent": agent_name,
            "length_status": chapter_length_status(wc),
            "spell_errors": spell_check(response[:5000]),
        }
    
    async def agent_continue_chapter(self, chapter_id: int, direction: str = "",
                                      agent_name: str = "SoulIllusions Agent") -> Dict:
        """Have an agent continue writing an existing chapter."""
        conn = sqlite3.connect(str(BOOKS_DB))
        c = conn.cursor()
        c.execute("SELECT book_id, title, content, chapter_number FROM chapters WHERE id = ?", (chapter_id,))
        row = c.fetchone()
        conn.close()
        if not row:
            return {"error": "Chapter not found"}
        
        book_id, title, content, ch_num = row
        existing = content or ""
        
        prompt = f"""Continue writing chapter {ch_num}: '{title}'.
Existing content ({count_words(existing)} words):
{existing[-2000:]}

Direction: {direction if direction else "Continue the story naturally."}
Write the next 1000+ words. Maintain tone and style."""
        
        system = "You are a professional novelist. Continue the chapter seamlessly."
        
        if not self.llm:
            return {"error": "LLM not available"}
        
        response = await self.llm.generate(prompt, system, max_tokens=4000)
        new_content = existing + "\n\n" + response
        wc = count_words(new_content)
        
        self.update_chapter(chapter_id, content=new_content)
        
        return {
            "continuation": response,
            "total_content": new_content,
            "word_count": wc,
            "length_status": chapter_length_status(wc),
        }
    
    async def agent_correct_chapter(self, chapter_id: int) -> Dict:
        """Have an agent fix spelling and grammar in a chapter."""
        conn = sqlite3.connect(str(BOOKS_DB))
        c = conn.cursor()
        c.execute("SELECT content FROM chapters WHERE id = ?", (chapter_id,))
        row = c.fetchone()
        conn.close()
        if not row or not row[0]:
            return {"error": "Chapter not found or empty"}
        
        content = row[0]
        errors = spell_check(content[:5000])
        
        if not self.llm:
            return {"error": "LLM not available", "spell_errors": errors}
        
        prompt = f"""Fix spelling and grammar errors in this text. Keep the story exactly the same, only correct errors:

{content[:8000]}"""
        
        system = "You are a professional editor. Fix only spelling and grammar. Do not change the story."
        corrected = await self.llm.generate(prompt, system, max_tokens=8000)
        
        wc = count_words(corrected)
        self.update_chapter(chapter_id, content=corrected)
        
        return {
            "corrected": corrected,
            "word_count": wc,
            "original_errors": errors,
            "status": "corrected"
        }
    
    def get_book(self, book_id: int) -> Optional[Dict]:
        return self.analysis_engine._get_book(book_id)
    
    def get_chapters(self, book_id: int) -> List[Dict]:
        return self.analysis_engine._get_chapters(book_id)
    
    def get_chapter(self, chapter_id: int) -> Optional[Dict]:
        conn = sqlite3.connect(str(BOOKS_DB))
        c = conn.cursor()
        c.execute("SELECT * FROM chapters WHERE id = ?", (chapter_id,))
        row = c.fetchone()
        conn.close()
        if not row:
            return None
        cols = ["id", "book_id", "chapter_number", "title", "content", "word_count",
                "target_word_count", "status", "notes", "agent_assisted", "created_at", "updated_at"]
        return dict(zip(cols, row))
    
    def get_characters(self, book_id: int) -> List[Dict]:
        return self.analysis_engine._get_characters(book_id)
    
    def get_world_elements(self, book_id: int) -> List[Dict]:
        return self.analysis_engine._get_world_elements(book_id)
    
    def list_books(self, limit: int = 50, category: str = "") -> List[Dict]:
        conn = sqlite3.connect(str(BOOKS_DB))
        c = conn.cursor()
        if category:
            c.execute("SELECT id, title, author, genre, category, status, word_count, chapter_count, rating, created_at FROM books WHERE category = ? ORDER BY updated_at DESC LIMIT ?",
                      (category, limit))
        else:
            c.execute("SELECT id, title, author, genre, category, status, word_count, chapter_count, rating, created_at FROM books ORDER BY updated_at DESC LIMIT ?",
                      (limit,))
        rows = c.fetchall()
        conn.close()
        cols = ["id", "title", "author", "genre", "category", "status", "word_count", "chapter_count", "rating", "created_at"]
        return [dict(zip(cols, r)) for r in rows]
    
    def delete_book(self, book_id: int) -> Dict:
        conn = sqlite3.connect(str(BOOKS_DB))
        c = conn.cursor()
        c.execute("DELETE FROM books WHERE id = ?", (book_id,))
        c.execute("DELETE FROM chapters WHERE book_id = ?", (book_id,))
        c.execute("DELETE FROM characters WHERE book_id = ?", (book_id,))
        c.execute("DELETE FROM world_elements WHERE book_id = ?", (book_id,))
        c.execute("DELETE FROM audiobooks WHERE book_id = ?", (book_id,))
        c.execute("DELETE FROM book_analysis WHERE book_id = ?", (book_id,))
        c.execute("DELETE FROM agent_writing_sessions WHERE book_id = ?", (book_id,))
        conn.commit()
        conn.close()
        return {"status": "deleted", "book_id": book_id}
    
    def export_book(self, book_id: int, format: str = "txt") -> Dict:
        """Export book to file."""
        book = self.get_book(book_id)
        if not book:
            return {"error": "Book not found"}
        chapters = self.get_chapters(book_id)
        
        safe_title = re.sub(r'[^a-zA-Z0-9_-]', '_', book["title"].lower())[:50]
        
        if format == "txt":
            filepath = EXPORT_DIR / f"{safe_title}.txt"
            lines = [f"{book['title']}", f"By {book['author']}", "=" * 60, ""]
            for ch in chapters:
                lines.append(f"\nChapter {ch['chapter_number']}: {ch['title']}\n")
                lines.append(ch.get("content", ""))
                lines.append("")
            filepath.write_text("\n".join(lines), encoding="utf-8")
            return {"status": "exported", "file": str(filepath), "format": "txt"}
        
        elif format == "html":
            filepath = EXPORT_DIR / f"{safe_title}.html"
            html_parts = [f"<html><head><title>{book['title']}</title></head><body>"]
            html_parts.append(f"<h1>{book['title']}</h1><p>By {book['author']}</p>")
            for ch in chapters:
                html_parts.append(f"<h2>Chapter {ch['chapter_number']}: {ch['title']}</h2>")
                content = ch.get("content", "")
                paragraphs = content.split("\n\n")
                for p in paragraphs:
                    html_parts.append(f"<p>{p}</p>")
            html_parts.append("</body></html>")
            filepath.write_text("\n".join(html_parts), encoding="utf-8")
            return {"status": "exported", "file": str(filepath), "format": "html"}
        
        elif format == "json":
            filepath = EXPORT_DIR / f"{safe_title}.json"
            data = {
                "book": book,
                "chapters": chapters,
                "characters": self.get_characters(book_id),
                "world_elements": self.get_world_elements(book_id),
            }
            filepath.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
            return {"status": "exported", "file": str(filepath), "format": "json"}
        
        return {"error": f"Unknown format: {format}"}
    
    def get_book_for_video(self, book_id: int) -> Dict:
        """Get book data formatted for text-to-video pipeline."""
        analysis = self.analysis_engine.get_analysis(book_id)
        if not analysis:
            # Run analysis if not done
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, self.analysis_engine.analyze_book(book_id))
                analysis = future.result()
        return {
            "book_id": book_id,
            "title": analysis.get("title", ""),
            "visual_scenes": analysis.get("visual_elements", []),
            "characters": analysis.get("characters", []),
            "plot_structure": analysis.get("plot_structure", {}),
            "scenes": analysis.get("scenes", []),
            "themes": analysis.get("themes", []),
        }
    
    def get_book_for_game(self, book_id: int) -> Dict:
        """Get book data formatted for text-to-game pipeline."""
        analysis = self.analysis_engine.get_analysis(book_id)
        if not analysis:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, self.analysis_engine.analyze_book(book_id))
                analysis = future.result()
        game_data = analysis.get("game_adaptation", {})
        return {
            "book_id": book_id,
            "title": analysis.get("title", ""),
            "player_character": game_data.get("player_character", "Hero"),
            "antagonist": game_data.get("antagonist"),
            "locations": game_data.get("locations", []),
            "items": game_data.get("items", []),
            "suggested_genre": game_data.get("suggested_genre", "adventure"),
            "key_moments": game_data.get("key_moments", []),
            "characters": analysis.get("characters", []),
            "themes": analysis.get("themes", []),
        }
    
    def _update_book_stats(self, book_id: int):
        conn = sqlite3.connect(str(BOOKS_DB))
        c = conn.cursor()
        c.execute("SELECT COUNT(*), SUM(word_count) FROM chapters WHERE book_id = ?", (book_id,))
        row = c.fetchone()
        ch_count, total_wc = row[0], row[1] or 0
        c.execute("UPDATE books SET chapter_count = ?, word_count = ?, updated_at = datetime('now') WHERE id = ?",
                  (ch_count, total_wc, book_id))
        conn.commit()
        conn.close()


# --- Singleton ---
_book_manager: Optional[BookManager] = None

def get_book_manager() -> BookManager:
    global _book_manager
    if _book_manager is None:
        _book_manager = BookManager()
    return _book_manager


# --- CLI ---
def cli():
    print("=" * 50)
    print("  SoulIllusions Book Writer System")
    print("=" * 50)
    
    if len(sys.argv) < 2:
        print("Commands:")
        print("  create <title> [--genre X] [--desc X] [--target N]")
        print("  list [--category X]")
        print("  add-chapter <book_id> <number> [--title X] [--content X]")
        print("  write <book_id> <chapter_num> <prompt>")
        print("  continue <chapter_id> [direction]")
        print("  correct <chapter_id>")
        print("  add-character <book_id> <name> [--role X] [--appearance X]")
        print("  analyze <book_id>")
        print("  audiobook <book_id> [--chapter <id>]")
        print("  export <book_id> [--format txt|html|json]")
        print("  for-video <book_id>")
        print("  for-game <book_id>")
        print("  delete <book_id>")
        return
    
    cmd = sys.argv[1]
    mgr = get_book_manager()
    
    if cmd == "create":
        title = sys.argv[2] if len(sys.argv) > 2 else "Untitled"
        genre = ""
        desc = ""
        target = 50000
        for i, arg in enumerate(sys.argv):
            if arg == "--genre" and i + 1 < len(sys.argv): genre = sys.argv[i + 1]
            elif arg == "--desc" and i + 1 < len(sys.argv): desc = sys.argv[i + 1]
            elif arg == "--target" and i + 1 < len(sys.argv): target = int(sys.argv[i + 1])
        result = mgr.create_book(title, desc, genre, target_word_count=target)
        print(json.dumps(result, indent=2))
    
    elif cmd == "list":
        category = ""
        if "--category" in sys.argv:
            category = sys.argv[sys.argv.index("--category") + 1]
        books = mgr.list_books(category=category)
        print(json.dumps(books, indent=2))
    
    elif cmd == "add-chapter":
        book_id = int(sys.argv[2])
        ch_num = int(sys.argv[3])
        title = ""
        content = ""
        for i, arg in enumerate(sys.argv):
            if arg == "--title" and i + 1 < len(sys.argv): title = sys.argv[i + 1]
            elif arg == "--content" and i + 1 < len(sys.argv): content = sys.argv[i + 1]
        result = mgr.add_chapter(book_id, ch_num, title, content)
        print(json.dumps(result, indent=2))
    
    elif cmd == "write":
        book_id = int(sys.argv[2])
        ch_num = int(sys.argv[3])
        prompt = " ".join(sys.argv[4:])
        result = asyncio.run(mgr.agent_write_chapter(book_id, ch_num, prompt))
        print(json.dumps(result, indent=2, default=str))
    
    elif cmd == "continue":
        ch_id = int(sys.argv[2])
        direction = " ".join(sys.argv[3:]) if len(sys.argv) > 3 else ""
        result = asyncio.run(mgr.agent_continue_chapter(ch_id, direction))
        print(json.dumps(result, indent=2, default=str))
    
    elif cmd == "correct":
        ch_id = int(sys.argv[2])
        result = asyncio.run(mgr.agent_correct_chapter(ch_id))
        print(json.dumps(result, indent=2, default=str))
    
    elif cmd == "add-character":
        book_id = int(sys.argv[2])
        name = sys.argv[3]
        role = "supporting"
        appearance = ""
        for i, arg in enumerate(sys.argv):
            if arg == "--role" and i + 1 < len(sys.argv): role = sys.argv[i + 1]
            elif arg == "--appearance" and i + 1 < len(sys.argv): appearance = sys.argv[i + 1]
        result = mgr.add_character(book_id, name, role=role, appearance=appearance)
        print(json.dumps(result, indent=2))
    
    elif cmd == "analyze":
        book_id = int(sys.argv[2])
        result = asyncio.run(mgr.analysis_engine.analyze_book(book_id))
        print(json.dumps(result, indent=2, default=str))
    
    elif cmd == "audiobook":
        book_id = int(sys.argv[2])
        if "--chapter" in sys.argv:
            ch_id = int(sys.argv[sys.argv.index("--chapter") + 1])
            result = asyncio.run(mgr.audiobook_gen.generate_chapter_audio(book_id, ch_id))
        else:
            result = asyncio.run(mgr.audiobook_gen.generate_book_audio(book_id))
        print(json.dumps(result, indent=2, default=str))
    
    elif cmd == "export":
        book_id = int(sys.argv[2])
        fmt = "txt"
        if "--format" in sys.argv:
            fmt = sys.argv[sys.argv.index("--format") + 1]
        result = mgr.export_book(book_id, fmt)
        print(json.dumps(result, indent=2))
    
    elif cmd == "for-video":
        book_id = int(sys.argv[2])
        result = mgr.get_book_for_video(book_id)
        print(json.dumps(result, indent=2, default=str))
    
    elif cmd == "for-game":
        book_id = int(sys.argv[2])
        result = mgr.get_book_for_game(book_id)
        print(json.dumps(result, indent=2, default=str))
    
    elif cmd == "delete":
        book_id = int(sys.argv[2])
        result = mgr.delete_book(book_id)
        print(json.dumps(result, indent=2))
    
    else:
        print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    cli()
