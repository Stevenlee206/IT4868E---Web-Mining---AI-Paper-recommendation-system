import os
from functools import cache

import numpy as np
import requests

URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "allenai/specter2_base/pipeline/feature-extraction"
)


class Unavailable(Exception):
    pass


def request(title, abstract):
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise Unavailable("HF_TOKEN is not set")
    try:
        response = requests.post(
            URL,
            headers={"Authorization": f"Bearer {token}"},
            json={"inputs": f"{title} [SEP] {abstract}", "truncate": True},
            timeout=10,
        )
        response.raise_for_status()
        # First row: the CLS token, or the pooled vector if the server pools.
        vector = np.array(response.json(), dtype=np.float32).reshape(-1, 768)[0]
    except requests.RequestException as error:
        status = getattr(error.response, "status_code", None)
        raise Unavailable(f"HTTP {status}" if status else type(error).__name__) from error
    except (ValueError, TypeError) as error:
        raise Unavailable("unexpected response") from error
    return vector


@cache
def verified(title, abstract, expected):
    """Compare the API with our stored vectors once, in case it pools differently."""
    vector = request(title, abstract)
    cosine = vector @ np.frombuffer(expected, np.float32) / np.linalg.norm(vector)
    if not cosine > 0.98:
        raise Unavailable("API vectors do not match the stored ones")


def embed(title, abstract, check):
    (known_title, known_abstract), expected = check
    verified(known_title, known_abstract, expected.tobytes())
    return request(title, abstract)
