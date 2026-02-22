# Genie bundle (on-device LLM for Agent tactical query)

This folder holds the Qualcomm Genie runtime and **Llama-v3.2-3B-Instruct** model binaries (w4a16/w8a16 quantized for on-device) so the Agent tab can run on-device Q&A without the cloud.

Model: state-of-the-art LLM for language understanding and generation. See [Qualcomm AI Hub – Llama-v3.2-3B-Instruct](https://aihub.qualcomm.com/compute/models/llama_v3_2_3b_instruct).

## What you need in this folder

- **genie_config.json** — already included (Llama 3.2 3B instruct).
- **tokenizer.json** — run `python scripts/setup_genie_bundle.py` from the repo root to download it.
- **genie-t2t-run.exe** and Qualcomm libs — from QAIRT SDK (see below).
- **llama_v3_2_3b_instruct_part_1_of_2.bin**, **part_2_of_2.bin** — from Qualcomm AI Hub export (see below).
- **htp_backend_ext_config.json** — optional; copy from QAIRT/ai-hub-apps if you use HTP.

## Quick setup (tokenizer only)

From repo root:

```powershell
python -m pip install huggingface_hub
python scripts/setup_genie_bundle.py
```
(Use `python` from your Python 3.12 install. If you have the launcher: `py -3.12 -m pip ...`)

This creates `genie_bundle/` (if needed), keeps `genie_config.json`, and downloads **tokenizer.json** from Hugging Face. The Agent tab will still work without Genie (vector search + manuals or Ollama); Genie is optional for full on-device LLM answers.

## Full Genie setup (exe + model binaries)

### Option A: One script (recommended)

From repo root (Python 3.10–3.13 required):

```powershell
python scripts/setup_llama_qualcomm.py
```

This will:
1. Install `qai-hub-models[llama-v3-2-3b-instruct]`
2. Configure Qualcomm AI Hub (prompt for API token, or set `QAI_HUB_API_TOKEN`)
3. Optionally run the CLI demo to verify the model
4. Export the model for Snapdragon X Elite into `genie_bundle/`

Get your API token: [Qualcomm AI Hub Workbench](https://app.aihub.qualcomm.com) → Account → Settings → API Token.

### Option B: Manual steps

1. **Install QAIRT SDK**  
   - [QAIRT SDK](https://developer.qualcomm.com/software/qualcomm-ai-research-toolkit) and follow the Windows/Linux setup guide.

2. **Use Python 3.10–3.13** (3.12 recommended).  
   Check: `py -3.12 --version`. `qai-hub-models` does not support 3.9 or 3.14+.

3. **Install and export Llama 3.2 3B** (from repo root):
   ```powershell
   pip install "qai-hub-models[llama-v3-2-3b-instruct]"
   qai-hub configure --api_token YOUR_TOKEN
   mkdir -Force genie_bundle
   python -m qai_hub_models.models.llama_v3_2_3b_instruct.export --chipset qualcomm-snapdragon-x-elite --skip-inferencing --skip-profiling --output-dir genie_bundle
   ```

4. **Copy Genie runtime into this folder**  
   - Copy `genie-t2t-run.exe` and required DLLs from `$QNN_SDK_ROOT` (or QAIRT) into this `genie_bundle/` folder.  
   - See [Qualcomm ai-hub-apps llm_on_genie](https://github.com/quic/ai-hub-apps/tree/master/tutorials/llm_on_genie) for the exact file list.

5. **Point config at your part files**  
   If the export produced different part names (e.g. `*_part_1_of_3.bin`), edit `genie_config.json` → `dialog.model.binary.ctx-bins` to match.

Once `genie-t2t-run.exe`, `genie_config.json`, `tokenizer.json`, and the `.bin` files are in place, the backend will use Genie for Agent tactical answers automatically.
