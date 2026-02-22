"""
Set up Llama-v3.2-3B-Instruct per Qualcomm AI Hub instructions and export for on-device (Genie).

Follows: https://aihub.qualcomm.com/compute/models/llama_v3_2_3b_instruct

Steps:
  1. Install qai-hub-models[llama-v3-2-3b-instruct] (Python 3.10–3.13).
  2. Configure Qualcomm AI Hub: qai-hub configure --api_token YOUR_TOKEN
  3. Export for Snapdragon X Elite into genie_bundle/ (optional: run CLI demo to verify).

Run from repo root:
    python scripts/setup_llama_qualcomm.py

Environment:
  QAI_HUB_API_TOKEN  Optional. If set, skips interactive token prompt for configure.
"""

import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
GENIE_BUNDLE = PROJECT_ROOT / "genie_bundle"

# Python version supported by qai-hub-models: 3.10 <= x < 3.14
MIN_PY = (3, 10)
MAX_PY = (3, 14)


def check_python():
    v = sys.version_info
    if (MIN_PY <= (v.major, v.minor) < MAX_PY):
        return True
    print(f"Python {v.major}.{v.minor} detected. qai-hub-models requires 3.10 <= version < 3.14.")
    return False


def run(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> bool:
    cwd = cwd or PROJECT_ROOT
    env = env or os.environ.copy()
    print(f"Running: {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=cwd, env=env)
    return r.returncode == 0


def install_package(use_user: bool = False) -> bool:
    """Install qai-hub-models with Llama 3.2 3B Instruct extra."""
    cmd = [sys.executable, "-m", "pip", "install", "-U", "qai-hub-models[llama-v3-2-3b-instruct]"]
    if use_user:
        cmd.append("--user")
    return run(cmd)


def configure_qai_hub() -> bool:
    """Configure Qualcomm AI Hub Workbench (API token)."""
    token = os.environ.get("QAI_HUB_API_TOKEN", "").strip()
    if not token and sys.stdin.isatty():
        print("\nQualcomm AI Hub needs an API token.")
        print("Get it from: https://app.aihub.qualcomm.com -> Account -> Settings -> API Token")
        try:
            token = input("Paste your API token (or press Enter to skip configure): ").strip()
        except EOFError:
            token = ""
    if not token:
        print("Skipping qai-hub configure (set QAI_HUB_API_TOKEN to configure non-interactively). Export may fail if not already configured.")
        return True
    return run([sys.executable, "-m", "qai_hub", "configure", "--api_token", token])


def run_demo() -> bool:
    """Run CLI demo to verify the model works end-to-end."""
    print("\n--- Running CLI demo (verify model) ---")
    return run([
        sys.executable, "-m", "qai_hub_models.models.llama_v3_2_3b_instruct.demo",
    ])


def export_for_genie() -> bool:
    """Export model for on-device deployment into genie_bundle/."""
    GENIE_BUNDLE.mkdir(parents=True, exist_ok=True)
    print(f"\n--- Exporting to {GENIE_BUNDLE} (chipset: qualcomm-snapdragon-x-elite) ---")
    return run([
        sys.executable, "-m", "qai_hub_models.models.llama_v3_2_3b_instruct.export",
        "--chipset", "qualcomm-snapdragon-x-elite",
        "--skip-inferencing",
        "--skip-profiling",
        "--output-dir", str(GENIE_BUNDLE),
    ])


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Setup Llama-v3.2-3B-Instruct (Qualcomm AI Hub)")
    parser.add_argument("--user", action="store_true", help="Use pip --user (avoids Windows Access Denied)")
    args = parser.parse_args()

    print("Llama-v3.2-3B-Instruct setup (Qualcomm AI Hub)\n")
    if not check_python():
        return 1

    # 1) Install
    if not install_package(use_user=args.user):
        print("Install failed. On Windows try: python scripts/setup_llama_qualcomm.py --user")
        return 1
    print("Install OK.\n")

    # 2) Configure (optional if token in env)
    if not configure_qai_hub():
        print("Configure failed or skipped. Export may fail.")
    else:
        print("Configure OK.\n")

    # 3) Optional: run demo to verify (skip when non-interactive)
    run_demo_choice = os.environ.get("RUN_LLAMA_DEMO", "").strip().lower()
    if run_demo_choice not in ("0", "no", "false") and sys.stdin.isatty():
        try:
            run_demo_choice = input("Run CLI demo to verify model? [y/N]: ").strip().lower()
        except EOFError:
            run_demo_choice = "n"
    if run_demo_choice in ("y", "yes", "1", "true"):
        run_demo()

    # 4) Export for Genie (on-device)
    if not export_for_genie():
        print("Export failed. Check token and Qualcomm AI Hub access.")
        return 1
    print("\nExport OK. genie_bundle/ should contain model binaries (.bin).")
    print("Next: add genie-t2t-run.exe and QNN libs to genie_bundle/ (see genie_bundle/README.md).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
