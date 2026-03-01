import io
import json
import asyncio
from typing import Any, Optional, Awaitable

import pandas as pd


def schedule(coro: Awaitable[Any]) -> None:
    """
    Naplánuje coroutine do běžícího event loopu (NiceGUI).
    Funguje v sync callbacku (timer, on_change, atd.).
    """
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(coro)
    except RuntimeError:
        # fallback (typicky jen mimo NiceGUI loop)
        asyncio.run(coro)


def df_to_json_bytes(df: pd.DataFrame) -> bytes:
    payload = {
        "columns": list(df.columns),
        "index": list(df.index),
        "data": df.to_numpy().tolist(),
    }
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def df_from_json_bytes(b: bytes) -> pd.DataFrame:
    obj = json.loads(b.decode("utf-8"))
    df = pd.DataFrame(obj["data"], columns=obj["columns"])
    try:
        df.index = obj["index"]
    except Exception:
        pass
    return df


def safe_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


def safe_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return default
