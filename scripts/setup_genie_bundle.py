"""
Set up the genie_bundle folder for on-device LLM (Agent tactical query).

- Ensures genie_bundle/ exists and has genie_config.json.
- Downloads tokenizer.json from Hugging Face (Llama 3.2 tokenizer).

Run from repo root:
    python scripts/setup_genie_bundle.py

You still need to add (from Qualcomm QAIRT / qai-hub-models export):
- genie-t2t-run.exe and QNN libs
- llama_v3_2_3b_instruct_part_*.bin files

See genie_bundle/README.md for full steps.
"""

import os
import sys
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
GENIE_BUNDLE = PROJECT_ROOT / "genie_bundle"
CONFIG_NAME = "genie_config.json"
TOKENIZER_NAME = "tokenizer.json"

# Llama 3.2 tokenizer (same as 3B/8B); use a repo that doesn't require gated access when possible
HF_TOKENIZER_REPO = "HuggingFaceFW/llama-3.2-tokenizer"
HF_TOKENIZER_PATH = "tokenizer.json"


def ensure_genie_bundle():
    GENIE_BUNDLE.mkdir(parents=True, exist_ok=True)
    config_path = GENIE_BUNDLE / CONFIG_NAME
    if not config_path.exists():
        # Write a minimal config if missing (normally we ship one in repo)
        minimal = {
            "dialog": {
                "version": 1,
                "type": "basic",
                "context": {"version": 1, "size": 4096, "n-vocab": 128256, "bos-token": 128000, "eos-token": 128009},
                "sampler": {"version": 1, "seed": 42, "temp": 0.8, "top-k": 40, "top-p": 0.95},
                "tokenizer": {"version": 1, "path": TOKENIZER_NAME},
                "engine": {
                    "version": 1,
                    "n-threads": 3,
                    "backend": {
                        "version": 1,
                        "type": "QnnHtp",
                        "QnnHtp": {
                            "version": 1,
                            "use-mmap": True,
                            "spill-fill-bufsize": 0,
                            "mmap-budget": 0,
                            "poll": True,
                            "cpu-mask": "0xe0",
                            "kv-dim": 128,
                            "pos-id-dim": 64,
                            "allow-async-init": False,
                            "rope-theta": 500000,
                        },
                        "extensions": "htp_backend_ext_config.json",
                    },
                },
                "model": {
                    "version": 1,
                    "type": "binary",
                    "binary": {
                        "version": 1,
                        "ctx-bins": [
                            "llama_v3_2_3b_instruct_part_1_of_2.bin",
                            "llama_v3_2_3b_instruct_part_2_of_2.bin",
                        ],
                    },
                },
            }
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump({"dialog": minimal["dialog"]}, f, indent=2)
        print(f"Wrote {config_path}")
    else:
        print(f"Config present: {config_path}")


def download_tokenizer():
    dest = GENIE_BUNDLE / TOKENIZER_NAME
    if dest.exists():
        print(f"Tokenizer already present: {dest}")
        return True
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Install huggingface_hub to download tokenizer: py -m pip install huggingface_hub")
        return False
    print(f"Downloading {TOKENIZER_NAME} from Hugging Face ({HF_TOKENIZER_REPO})...")
    try:
        path = hf_hub_download(
            repo_id=HF_TOKENIZER_REPO,
            filename=HF_TOKENIZER_PATH,
            local_dir=str(GENIE_BUNDLE),
            local_dir_use_symlinks=False,
        )
        print(f"Saved tokenizer to {path}")
        return True
    except Exception as e:
        # Fallback: try meta-llama/Llama-3.2-3B (may require login)
        print(f"Download failed: {e}")
        try:
            path = hf_hub_download(
                repo_id="meta-llama/Llama-3.2-3B",
                filename="tokenizer.json",
                local_dir=str(GENIE_BUNDLE),
                local_dir_use_symlinks=False,
            )
            print(f"Saved tokenizer to {path}")
            return True
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            print("You can manually download tokenizer.json from Hugging Face (Llama 3.2) into genie_bundle/")
            return False


def main():
    print("Genie bundle setup")
    print(f"Bundle dir: {GENIE_BUNDLE}\n")
    ensure_genie_bundle()
    ok = download_tokenizer()
    print()
    if ok:
        print("Tokenizer ready. For full Genie support, add genie-t2t-run.exe and model .bin files (see genie_bundle/README.md).")
    else:
        print("Tokenizer not installed. Agent tab will still work using vector search + manuals only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
