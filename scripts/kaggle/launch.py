"""Run the benchmark on Kaggle and download the results.

    python scripts/kaggle/launch.py          # upload, run all kernels, wait, download
    python scripts/kaggle/launch.py wmf gcn  # rerun some kernels, then wait for all
    python scripts/kaggle/launch.py download # only fetch finished results

Needs the Kaggle CLI (`pip install kaggle`) and a configured account. The dataset and
kernels are private. Nothing heavy runs locally.
"""

import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GROUPS = ("encode", "wmf", "control", "cpu", "gcn")
GPU_GROUPS = ("encode", "wmf", "control")
DATASET = "paperrec-benchmark-bundle"
RESULTS = ROOT / "results" / "kaggle"


def kaggle(*args, check=True):
    done = subprocess.run(["kaggle", *args], check=False, capture_output=True, text=True)
    if check and done.returncode:
        raise RuntimeError(f"kaggle {' '.join(args)}\n{done.stdout}{done.stderr}")
    return done.stdout + done.stderr


def username():
    line = next(x for x in kaggle("config", "view").splitlines() if "username" in x)
    return line.split(":")[1].strip()


def upload(user):
    """Bundle the code, raw data and splits as one private dataset version."""
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        # Not named .tar.gz, because Kaggle unpacks archives it recognizes.
        with tarfile.open(folder / "bundle.bin", "w:gz") as archive:
            for path in ("paperrec", "data/raw/citeulike-a", "artifacts/splits"):
                archive.add(ROOT / path, path, filter=skip_caches)
        metadata = {"title": DATASET, "id": f"{user}/{DATASET}", "licenses": [{"name": "unknown"}]}
        (folder / "dataset-metadata.json").write_text(json.dumps(metadata))
        exists = DATASET in kaggle("datasets", "list", "--mine", "--search", DATASET)
        action = ("version", "-m", "update") if exists else ("create",)
        print(kaggle("datasets", *action, "-p", str(folder)).strip(), flush=True)
    while "ready" not in kaggle("datasets", "status", f"{user}/{DATASET}", check=False).lower():
        time.sleep(20)
    time.sleep(30)  # The listing lags behind the status.


def skip_caches(info):
    return None if "__pycache__" in info.name else info


def push(user, group):
    slug = f"paperrec-{group}"
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        source = (Path(__file__).parent / "kernel.py").read_text()
        (folder / "kernel.py").write_text(source.replace("__GROUP__", group))
        metadata = {
            "id": f"{user}/{slug}",
            "title": slug,
            "code_file": "kernel.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": group in GPU_GROUPS,
            "enable_internet": True,
            "dataset_sources": [f"{user}/{DATASET}"],
            "kernel_sources": [f"{user}/paperrec-encode"] if group == "gcn" else [],
        }
        (folder / "kernel-metadata.json").write_text(json.dumps(metadata))
        extra = ("--accelerator", "NvidiaTeslaT4") if group in GPU_GROUPS else ()
        print(kaggle("kernels", "push", "-p", str(folder), *extra).strip(), flush=True)


def wait(user, group):
    while True:
        status = kaggle("kernels", "status", f"{user}/paperrec-{group}", check=False)
        state = re.search(r"KernelWorkerStatus\.(\w+)", status)
        state = state.group(1) if state else "UNKNOWN"  # For example a network error.
        if state not in ("UNKNOWN", "QUEUED", "RUNNING", "NEW_SCRIPT"):
            print(f"{group}: {state}", flush=True)
            if state != "COMPLETE":
                raise RuntimeError(f"{group} did not complete; see its log on Kaggle")
            return
        time.sleep(60)


def download(user, groups=GROUPS):
    for group in groups:
        target = RESULTS / group
        shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True)
        ref = f"{user}/paperrec-{group}"
        # Everything except the large feature files.
        kaggle("kernels", "output", ref, "-p", str(target), "--file-pattern", r"^(?!.*features/)")
        print(
            f"Downloaded {group}: {sum(f.is_file() for f in target.rglob('*'))} files", flush=True
        )
    for group in groups:
        source = RESULTS / group / "benchmark"
        if source.exists():
            shutil.copytree(source, ROOT / "results" / "benchmark", dirs_exist_ok=True)


def main():
    user = username()
    if sys.argv[1:] == ["download"]:
        return download(user)
    groups = sys.argv[1:] or GROUPS
    upload(user)
    for group in groups:
        if group != "gcn":
            push(user, group)
    if "gcn" in groups:  # Needs the encoded features.
        wait(user, "encode")
        push(user, "gcn")
    for group in GROUPS:
        wait(user, group)
    download(user)


if __name__ == "__main__":
    main()
