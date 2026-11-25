"""Create or update the encoder Space. Reads a write token from HF_TOKEN.

python deploy_space.py
"""

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

HERE = Path(__file__).parent

token = os.environ.get("HF_TOKEN")
if not token:
    sys.exit("Set HF_TOKEN to a token with write access first.")

api = HfApi(token=token)
try:
    repo = f"{api.whoami()['name']}/specter2-encoder"
    api.create_repo(
        repo, repo_type="space", space_sdk="gradio", space_hardware="zero-a10g", exist_ok=True
    )
    api.upload_folder(folder_path=HERE / "space", repo_id=repo, repo_type="space")
except HfHubHTTPError as error:
    sys.exit(f"Hugging Face refused: {error}")

(HERE / "space.txt").write_text(repo + "\n")
print(f"Uploaded to https://huggingface.co/spaces/{repo}")
print("The first build takes a few minutes. Its status is shown on that page.")
