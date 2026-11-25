import json
import os
from functools import cache
from pathlib import Path

import numpy as np
import requests

CONFIG = Path(__file__).with_name("space.txt")


class Unavailable(Exception):
    pass


def space_url():
    if url := os.environ.get("SPACE_URL"):
        return url.rstrip("/")
    if not CONFIG.exists():
        raise Unavailable("no Space is configured")
    return (
        f"https://{CONFIG.read_text().strip().replace('/', '-').replace('_', '-').lower()}.hf.space"
    )


def last_event(text):
    """Return the name and data of the last server-sent event."""
    lines = [line.split(": ", 1) for line in text.splitlines() if ": " in line]
    names = [value for key, value in lines if key == "event"]
    values = [value for key, value in lines if key == "data"]
    return names[-1], json.loads(values[-1])


@cache
def vectors(title, abstract):
    url = f"{space_url()}/gradio_api/call/encode"
    headers = {"Authorization": f"Bearer {t}"} if (t := os.environ.get("HF_TOKEN")) else {}
    try:
        call = requests.post(url, json={"data": [title, abstract]}, headers=headers, timeout=10)
        call.raise_for_status()
        reply = requests.get(f"{url}/{call.json()['event_id']}", headers=headers, timeout=30)
        reply.raise_for_status()
        name, data = last_event(reply.content.decode())
        if name != "complete":
            hint = "" if headers else "; set HF_TOKEN for more quota"
            raise Unavailable(f"the Space is out of quota or failed{hint}")
        result = {key: np.array(data[0][key], dtype=np.float32) for key in ("base", "adapter")}
        if any(vector.shape != (768,) for vector in result.values()):
            raise Unavailable("unexpected response")
        return result
    except requests.RequestException as error:
        status = getattr(error.response, "status_code", None)
        raise Unavailable(f"HTTP {status}" if status else type(error).__name__) from error
    except (ValueError, TypeError, KeyError, IndexError) as error:
        raise Unavailable("unexpected response") from error


def embed(title, abstract, adapter):
    return vectors(title, abstract)["adapter" if adapter else "base"]
