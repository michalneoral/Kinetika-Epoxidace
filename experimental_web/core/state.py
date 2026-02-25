from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, MutableMapping, Any

from nicegui import app


@dataclass
class SessionState:
    """Typed view of session state (used mainly for documentation/type hints)."""
    current_experiment_id: Optional[int] = None
    current_experiment_name: str = ""
    current_excel_file_id: Optional[int] = None
    current_excel_filename: str = ""
    current_excel_sheet: str = ""
    data_version: int = 0


class StateProxy:
    """Attribute-style access to a dict stored in app.storage.user.

    NiceGUI persists user storage by serializing to JSON-like structures.
    That means custom objects (like dataclasses) may come back as dict/ObservableDict.
    This proxy keeps the rest of the code ergonomic (st.current_experiment_id = ...).
    """

    def __init__(self, backing: MutableMapping[str, Any]) -> None:
        self._b = backing

    @property
    def current_experiment_id(self) -> Optional[int]:
        v = self._b.get("current_experiment_id", None)
        return int(v) if isinstance(v, str) and v.isdigit() else v

    @current_experiment_id.setter
    def current_experiment_id(self, value: Optional[int]) -> None:
        self._b["current_experiment_id"] = value

    @property
    def current_experiment_name(self) -> str:
        return str(self._b.get("current_experiment_name", ""))

    @current_experiment_name.setter
    def current_experiment_name(self, value: str) -> None:
        self._b["current_experiment_name"] = value or ""

    @property
    def current_excel_file_id(self) -> Optional[int]:
        v = self._b.get("current_excel_file_id", None)
        return int(v) if isinstance(v, str) and v.isdigit() else v

    @current_excel_file_id.setter
    def current_excel_file_id(self, value: Optional[int]) -> None:
        self._b["current_excel_file_id"] = value

    @property
    def current_excel_filename(self) -> str:
        return str(self._b.get("current_excel_filename", ""))

    @current_excel_filename.setter
    def current_excel_filename(self, value: str) -> None:
        self._b["current_excel_filename"] = value or ""

    @property
    def current_excel_sheet(self) -> str:
        return str(self._b.get("current_excel_sheet", ""))

    @current_excel_sheet.setter
    def current_excel_sheet(self, value: str) -> None:
        self._b["current_excel_sheet"] = value or ""

    @property
    def data_version(self) -> int:
        v = self._b.get("data_version", 0)
        try:
            return int(v)
        except Exception:
            return 0

    @data_version.setter
    def data_version(self, value: int) -> None:
        self._b["data_version"] = int(value)


def get_state() -> StateProxy:
    """Return per-user session state.

    Stored as a plain dict in app.storage.user for persistence and compatibility.
    """
    if "state" not in app.storage.user:
        app.storage.user["state"] = {"current_experiment_id": None, "current_experiment_name": "", "current_excel_file_id": None, "current_excel_filename": "", "current_excel_sheet": "", "data_version": 0}
    else:
        # migration/normalization: if older versions stored something else, convert to dict
        s = app.storage.user["state"]
        if isinstance(s, SessionState):
            app.storage.user["state"] = {
                "current_experiment_id": s.current_experiment_id,
                "current_experiment_name": s.current_experiment_name,
                "current_excel_file_id": None,
                "current_excel_filename": "",
                "current_excel_sheet": "",
                "data_version": 0,
            }
        elif isinstance(s, dict):
            s.setdefault("current_experiment_id", None)
            s.setdefault("current_experiment_name", "")
            s.setdefault("current_excel_file_id", None)
            s.setdefault("current_excel_filename", "")
            s.setdefault("current_excel_sheet", "")
            s.setdefault("data_version", 0)
        else:
            # ObservableDict behaves like dict but is not a real dict type; treat as mapping
            try:
                s.setdefault("current_experiment_id", None)  # type: ignore[attr-defined]
                s.setdefault("current_experiment_name", "")  # type: ignore[attr-defined]
                s.setdefault("current_excel_file_id", None)  # type: ignore[attr-defined]
                s.setdefault("current_excel_filename", "")  # type: ignore[attr-defined]
                s.setdefault("current_excel_sheet", "")  # type: ignore[attr-defined]
                s.setdefault("data_version", 0)  # type: ignore[attr-defined]
            except Exception:
                app.storage.user["state"] = {
                    "current_experiment_id": None,
                    "current_experiment_name": "",
                    "current_excel_file_id": None,
                    "current_excel_filename": "",
                    "current_excel_sheet": "",
                    "data_version": 0,
                }

    return StateProxy(app.storage.user["state"])
