"""Extract and export paper representations for WP3 (SPECTER2+GCN) and CTR.

Usage:
    python -m paperrec.extract_embeddings --models all --output-dir artifacts/embeddings
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .data import load_citeulike_a


def export_specter2(papers, output_path: Path, batch_size: int = 64):
    """Generate 768-d SPECTER2 embeddings for all papers and save to .npy."""
    print(f"--> Extracting SPECTER2 embeddings for {len(papers)} papers...", flush=True)
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"    Using device: {device}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
    model = AutoModel.from_pretrained("allenai/specter2_base").to(device)
    model.eval()

    texts = [f"{p.title.strip()} [SEP] {p.abstract.strip()}" for p in papers]
    all_embeddings = []

    total_batches = (len(texts) + batch_size - 1) // batch_size
    for i, start_idx in enumerate(range(0, len(texts), batch_size), start=1):
        batch_texts = texts[start_idx : start_idx + batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            output = model(**inputs)
            batch_vecs = output.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(batch_vecs)

        if i % 20 == 0 or i == total_batches:
            print(f"    Batch {i}/{total_batches} done", flush=True)

    embeddings = np.vstack(all_embeddings).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = embeddings / norms

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, normalized)
    print(f"--> [WP3 Delivery] Saved SPECTER2 embeddings to {output_path} (shape={normalized.shape})", flush=True)


def export_lda(papers, output_path: Path, n_topics: int = 100, seed: int = 42):
    """Generate paper-topic mixture Theta (K=100) and save to .npy for CTR."""
    print(f"--> Extracting LDA topic distributions (K={n_topics}) for CTR...", flush=True)
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer

    texts = [p.text for p in papers]
    vectorizer = CountVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b\w\w+\b",
        min_df=5,
        max_df=0.5,
    )
    counts = vectorizer.fit_transform(texts)

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=20,
        learning_method="online",
        random_state=seed,
        batch_size=128,
        n_jobs=1,
    )
    theta = lda.fit_transform(counts).astype(np.float32)

    # Normalize rows to sum to 1.0
    row_sums = theta.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    theta = theta / row_sums

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, theta)
    print(f"--> [CTR Delivery] Saved LDA topic distributions to {output_path} (shape={theta.shape})", flush=True)


def export_lsi(papers, output_path: Path, n_components: int = 200, seed: int = 42):
    """Generate LSI embeddings (K=200) and save to .npy."""
    print(f"--> Extracting LSI embeddings (K={n_components})...", flush=True)
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = [p.text for p in papers]
    tfidf = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b\w\w+\b",
        min_df=2,
        max_df=0.8,
        norm="l2",
    )
    x_tfidf = tfidf.fit_transform(texts)

    svd = TruncatedSVD(n_components=n_components, random_state=seed, algorithm="randomized")
    lsi_vecs = svd.fit_transform(x_tfidf).astype(np.float32)

    norms = np.linalg.norm(lsi_vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = lsi_vecs / norms

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, normalized)
    print(f"--> Saved LSI embeddings to {output_path} (shape={normalized.shape})", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/citeulike-a"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/embeddings"))
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["all", "specter2", "lda", "lsi"],
        default=["all"],
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--k-lsi", type=int, default=200)
    parser.add_argument("--k-lda", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    models = set(args.models)
    if "all" in models:
        models = {"specter2", "lda", "lsi"}

    dataset = load_citeulike_a(args.data)
    papers = dataset.papers
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if "specter2" in models:
        export_specter2(papers, output_dir / "specter2_embeddings.npy", batch_size=args.batch_size)

    if "lda" in models:
        export_lda(papers, output_dir / "lda_topics.npy", n_topics=args.k_lda, seed=args.seed)

    if "lsi" in models:
        export_lsi(papers, output_dir / "lsi_embeddings.npy", n_components=args.k_lsi, seed=args.seed)

    print("\n[Done] All requested embeddings exported successfully.", flush=True)


if __name__ == "__main__":
    main()
