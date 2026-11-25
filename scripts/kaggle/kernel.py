"""Runs inside Kaggle. launch.py replaces __GROUP__ before pushing."""

import os
import shutil
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

GROUP = "__GROUP__"
INPUT, WORK, REPO = Path("/kaggle/input"), Path("/kaggle/working"), Path("/tmp/paperrec")
OUTPUT, FEATURES, LOGS = WORK / "benchmark", WORK / "features", WORK / "logs"
JOBS = {
    "encode": [],
    "wmf": ["wmf"],
    "cpu": ["tfidf", "bm25", "lsi", "lda", "item-knn", "node2vec-citation", "node2vec-bipartite"],
    "gcn": ["specter2-base", "specter2-adapter", "gcn-bce", "nograph-bce", "gcn-bce-add"],
}

ENV = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


def sh(name, *command, env=None):
    start = time.time()
    with (LOGS / f"{name}.log").open("w") as log:
        code = subprocess.run(
            command, check=False, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT
        ).returncode
    print(f"{name}: exit {code}, {time.time() - start:.0f}s", flush=True)
    print("\n".join((LOGS / f"{name}.log").read_text().splitlines()[-12:]), flush=True)
    return code


def benchmark(name, *args, env=None):
    return sh(
        name, sys.executable, "-m", "paperrec.benchmark", *args,
        "--features", str(FEATURES), "--output", str(OUTPUT), env=env,
    )  # fmt: skip


def main():
    for folder in (REPO, OUTPUT, FEATURES, LOGS):
        folder.mkdir(parents=True, exist_ok=True)
    bundle = next(INPUT.rglob("bundle.bin"))
    with tarfile.open(bundle) as archive:
        archive.extractall(REPO, filter="data")
    packages = ["gensim>=4.4,<5", "psutil"]
    if GROUP in ("encode", "gcn"):
        packages += ["transformers>=4.40,<5", "adapters>=1,<2"]
    codes = [sh("pip", sys.executable, "-m", "pip", "install", "-q", *packages)]
    sh("versions", sys.executable, "-m", "pip", "freeze")

    if GROUP == "encode":
        codes.append(sh("gpu-info", "nvidia-smi"))
        codes.append(
            sh("encode-adapter", sys.executable, "-m", "paperrec.specter",
               "--cache-root", str(FEATURES / "specter2"), "--batch-size", "32")
        )  # fmt: skip
        codes.append(
            sh("encode-base", sys.executable, "-m", "paperrec.extract_embeddings",
               "--models", "specter2", "--output-dir", str(FEATURES / "base"))
        )  # fmt: skip
        shutil.move(FEATURES / "base" / "specter2_embeddings.npy", FEATURES / "specter2_base.npy")
    elif GROUP == "wmf":
        codes.append(benchmark("tune-wmf", "tune", "--models", "wmf"))
    elif GROUP == "cpu":
        codes.append(benchmark("tune-item-knn", "tune", "--models", "item-knn"))
    else:
        source = next(INPUT.rglob("specter2_base.npy")).parent
        shutil.copytree(source, FEATURES, dirs_exist_ok=True)

    tasks = [
        (model, scenario)
        for scenario in ("warm", "cold")
        for model in JOBS[GROUP]
        if not (model == "node2vec-bipartite" and scenario == "cold")
    ]

    def job(task):
        model, scenario = task
        name = f"{scenario}-{model}"
        return benchmark(name, "run", "--model", model, "--scenario", scenario, env=ENV)

    with ThreadPoolExecutor(1 if GROUP == "wmf" else os.cpu_count()) as pool:
        codes += pool.map(job, tasks)
    if GROUP == "gcn":
        codes.append(benchmark("export-demo", "export-demo"))
    if GROUP != "encode":
        shutil.rmtree(FEATURES)  # Only the encode kernel's features are inputs elsewhere.
    if any(codes):
        sys.exit(f"Failed steps: {codes}")


main()
