import json
import os
from datetime import datetime
from typing import List, Dict, Any

class ChatChunkSaver:
    """Utility to collect and persist *important* parts of a chat session.

    The class maintains an internal list of *chunks*, where each chunk is a
    dictionary containing a minimal yet useful representation of a message.

    Expected usage::

        saver = ChatChunkSaver()
        saver.add_message(role="user", content="Hello", important=True)
        saver.add_message(role="assistant", content="Hi there!", important=False)
        saver.save_to_file("my_chat_chunks.json")

    Only messages flagged as ``important`` are stored.  This mirrors the
    request to "save only same important data of previous chat".
    """

    def __init__(self) -> None:
        # Holds the accumulated chunks.
        self.chunks: List[Dict[str, Any]] = []

    def _now_iso(self) -> str:
        """Return the current UTC timestamp in ISO‑8601 format."""
        return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    def add_message(self, *, role: str, content: str, important: bool = False) -> None:
        """Add a message to the internal list.

        Parameters
        ----------
        role: str
            Either ``"user"`` or ``"assistant"``.
        content: str
            Raw text of the message.
        important: bool, default ``False``
            When ``True`` the message is kept; otherwise it is ignored.
        """
        if not important:
            # Skip non‑important messages as per the request.
            return

        chunk = {
            "role": role,
            "content": content,
            "timestamp": self._now_iso(),
        }
        self.chunks.append(chunk)

    def to_list(self) -> List[Dict[str, Any]]:
        """Return a shallow copy of the stored chunks list."""
        return list(self.chunks)

    def save_to_file(self, filepath: str, *, indent: int = 2) -> None:
        """Serialise the collected chunks to a JSON file.

        The directory hierarchy is created automatically if it does not exist.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=indent)
        print(f"✅ Saved {len(self.chunks)} chat chunk(s) to {filepath}")

    def load_from_file(self, filepath: str) -> None:
        """Load previously saved chunks, replacing any existing data.

        If the file does not exist the method leaves ``self.chunks`` unchanged.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        print(f"🔄 Loaded {len(self.chunks)} chat chunk(s) from {filepath}")

# ---------------------------------------------------------------------------
# Example helper – convenient one‑liner to extract "important" messages from a
# conversation transcript that follows the typical OpenAI/chat‑format.
# ---------------------------------------------------------------------------

def extract_important_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Return only messages that contain a ``"important": true`` flag.

    The input list should be a sequence of dictionaries with at least the keys
    ``role`` and ``content``.  An optional ``important`` boolean key is honoured.
    If the flag is missing, the message is considered *not* important.
    """
    return [msg for msg in messages if msg.get("important")]

# ---------------------------------------------------------------------------
# When executed as a script, the module reads a JSON transcript (if supplied),
# filters important items, and writes them to ``important_chat_chunks.json``.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python chat_save.py <transcript.json> [output.json]")
        sys.exit(1)

    transcript_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "important_chat_chunks.json"

    with open(transcript_path, "r", encoding="utf-8") as f:
        raw_messages = json.load(f)

    saver = ChatChunkSaver()
    for msg in extract_important_messages(raw_messages):
        saver.add_message(role=msg["role"], content=msg["content"], important=True)

    saver.save_to_file(output_path)
