"""
Registry + persistence layer for stateful prompt looping.
"""

import re
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypedDict, Optional, List

logger = logging.getLogger("prompt_registry")


class LoopState(TypedDict):
    session_id: str
    task: str
    draft: str
    critique: str
    iteration: int  


class MemoryStore(ABC):
    @abstractmethod
    def get(self, session_id: str) -> Optional[LoopState]: ...

    @abstractmethod
    def save(self, session_id: str, state: LoopState) -> None: ...

    @abstractmethod
    def clear(self, session_id: str) -> None: ...

    @abstractmethod
    def list_sessions(self) -> List[str]: ...


class InMemoryStore(MemoryStore):
    def __init__(self):
        self._data: dict[str, LoopState] = {}

    def get(self, session_id: str) -> Optional[LoopState]:
        return self._data.get(session_id)

    def save(self, session_id: str, state: LoopState) -> None:
        self._data[session_id] = state

    def clear(self, session_id: str) -> None:
        self._data.pop(session_id, None)

    def list_sessions(self) -> List[str]:
        return list(self._data.keys())


class JSONFileStore(MemoryStore):
    def __init__(self, directory: str = ".loop_memory"):
        self.dir = Path(directory)
        self.dir.mkdir(exist_ok=True)

    def _path(self, session_id: str) -> Path:
        safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", session_id)
        return self.dir / f"{safe_id}.json"

    def get(self, session_id: str) -> Optional[LoopState]:
        path = self._path(session_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except Exception:
            logger.exception("Failed reading memory file for session %s", session_id)
            return None

    def save(self, session_id: str, state: LoopState) -> None:
        self._path(session_id).write_text(json.dumps(state))

    def clear(self, session_id: str) -> None:
        self._path(session_id).unlink(missing_ok=True)

    def list_sessions(self) -> List[str]:
        return [p.stem for p in self.dir.glob("*.json")]


DEFAULT_STORE: MemoryStore = InMemoryStore()
