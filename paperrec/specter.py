"""SPECTER2 paper features with a resumable on-disk cache.

Install with: python -m pip install -e ".[neural]"

The encoder is pretrained and frozen, and learns nothing from this project's saves
or splits, so one feature matrix serves every scenario and fold. It was trained
elsewhere on citation-informed data, which may overlap these papers; report this
as an external pretrained-model assumption.

Encoding runs in small batches, longest texts first so memory problems appear
early. Finished batches are flushed to disk, so an interrupted run resumes.

CLI (from the project root):
    python -m paperrec.specter
"""

import json
import os
from pathlib import Path

import numpy as np

from .models.lexical import _json_hash

MODEL_NAME = "allenai/specter2_base"
ADAPTER_NAME = "allenai/specter2"
ADAPTER_ROLE = "proximity"
DIMENSIONS = 768
CACHE_VERSION = 1


def default_device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def make_batch_encoder(*, device, max_length):
    """Load SPECTER2 and return a function from papers to CLS embeddings."""
    import torch
    from adapters import AutoAdapterModel
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoAdapterModel.from_pretrained(MODEL_NAME)
    model.load_adapter(ADAPTER_NAME, source="hf", load_as=ADAPTER_ROLE, set_active=True)
    model = model.to(device).eval()

    def encode(papers) -> np.ndarray:
        texts = [f"{p.title}{tokenizer.sep_token}{p.abstract}" for p in papers]
        inputs = tokenizer(
            texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        ).to(device)
        with torch.inference_mode():
            output = model(**inputs).last_hidden_state[:, 0]
        return output.cpu().numpy().astype(np.float32)

    return encode


def _encode_to_disk(papers, encode, directory: Path, batch_size: int, progress=None) -> np.ndarray:
    final_path = directory / "embeddings.npy"
    partial_path = directory / "embeddings.partial.npy"
    progress_path = directory / "progress.json"
    directory.mkdir(parents=True, exist_ok=True)

    done = 0
    if partial_path.exists() and progress_path.exists():
        done = json.loads(progress_path.read_text(encoding="utf-8"))["done"]
        stored = np.lib.format.open_memmap(partial_path, mode="r+")
    else:
        stored = np.lib.format.open_memmap(
            partial_path, mode="w+", dtype=np.float32, shape=(len(papers), DIMENSIONS)
        )

    lengths = np.array([len(p.title) + len(p.abstract) for p in papers])
    order = np.argsort(-lengths, kind="stable")
    while done < len(papers):
        batch = order[done : done + batch_size]
        vectors = encode([papers[i] for i in batch])
        if vectors.shape != (len(batch), DIMENSIONS) or not np.isfinite(vectors).all():
            raise ValueError(f"Encoder returned invalid vectors with shape {vectors.shape}")
        stored[batch] = vectors
        stored.flush()
        done += len(batch)
        # Written only after the flush, so progress never claims unsaved vectors.
        temporary = progress_path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"done": done}), encoding="utf-8")
        os.replace(temporary, progress_path)
        if progress is not None:
            progress(done, len(papers))
    del stored
    os.replace(partial_path, final_path)
    progress_path.unlink()
    return np.load(final_path)


def specter2_features(
    papers,
    *,
    cache_root="artifacts/features/specter2",
    batch_size=8,
    max_length=512,
    device=None,
    encoder=None,
    progress=None,
) -> np.ndarray:
    """Return a float32 (papers, 768) matrix, encoding only what is not cached."""
    identity = {
        "format_version": CACHE_VERSION,
        "model": MODEL_NAME,
        "adapter": ADAPTER_NAME,
        "role": ADAPTER_ROLE,
        "max_length": max_length,
        "text_fingerprint": _json_hash([(p.paper_id, p.title, p.abstract) for p in papers]),
    }
    directory = Path(cache_root) / _json_hash(identity)
    final_path = directory / "embeddings.npy"
    if final_path.exists():
        features = np.load(final_path)
        if features.shape != (len(papers), DIMENSIONS) or not np.isfinite(features).all():
            raise ValueError(f"Invalid SPECTER2 cache at {final_path}")
        return features
    if encoder is None:
        encoder = make_batch_encoder(device=device or default_device(), max_length=max_length)
    features = _encode_to_disk(papers, encoder, directory, batch_size, progress)
    (directory / "metadata.json").write_text(json.dumps(identity, indent=2), encoding="utf-8")
    return features


if __name__ == "__main__":
    import argparse
    from time import perf_counter

    from .data import load_citeulike_a

    parser = argparse.ArgumentParser(description="Encode CiteULike-a papers with SPECTER2")
    parser.add_argument("--data", type=Path, default=Path("data/raw/citeulike-a"))
    parser.add_argument("--cache-root", type=Path, default=Path("artifacts/features/specter2"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default=None, help="Default: mps when available, else cpu")
    args = parser.parse_args()
    start = perf_counter()

    def report(done, total):
        if done == total or done % (args.batch_size * 50) == 0:
            print(f"{done}/{total} papers, {perf_counter() - start:.0f}s", flush=True)

    matrix = specter2_features(
        load_citeulike_a(args.data).papers,
        cache_root=args.cache_root,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
        progress=report,
    )
    print(f"Features ready: {matrix.shape}, {perf_counter() - start:.0f}s", flush=True)
