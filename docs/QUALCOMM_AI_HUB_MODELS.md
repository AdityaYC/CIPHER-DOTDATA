# Qualcomm AI Hub — models you can use

[Qualcomm AI Hub](https://aihub.qualcomm.com/models) has **332 model variants (167 models)** optimized for Qualcomm devices (e.g. Snapdragon X Elite). Below are the ones most relevant for this project.

---

## Text generation / agentic talk (LLMs)

Use these for the **Agent** tab (Q&A over video/manuals) when running on Snapdragon or when exporting for Genie.

| Model | Pip extra | Notes |
|-------|-----------|--------|
| **Llama-v3.2-3B-Instruct** | `qai-hub-models[llama-v3-2-3b-instruct]` | Best fit for agentic chat: 3B, 4K context, Snapdragon X Elite. Same family as in `genie_bundle`. |
| **Llama-v3.2-1B-Instruct** | `qai-hub-models[llama-v3-2-1b-instruct]` | Smallest; fastest on device. |
| **Llama-v3.1-8B-Instruct** | `qai-hub-models[llama-v3-1-8b-instruct]` | Higher quality, heavier. |
| **Llama-v3-8B-Instruct** | `qai-hub-models[llama-v3-8b-instruct]` | 8B option. |
| **Llama-v2-7B-Chat** | `qai-hub-models[llama-v2-7b-chat]` | Older 7B chat model. |
| **Qwen2-5-7B-Instruct** | (see [AI Hub](https://aihub.qualcomm.com/models)) | Alternative 7B instruct. |
| **Baichuan2-7B** | (see [AI Hub](https://aihub.qualcomm.com/models)) | General-purpose LLM. |

**Recommendation for agentic talk on Qualcomm:** **Llama-v3.2-3B-Instruct** — matches the Genie bundle setup and is validated on Snapdragon X Elite.

### Export for Genie / on-device

From repo root (with Python 3.12 and [qai-hub API token](https://app.aihub.qualcomm.com/docs/hub/faq.html)):

```powershell
pip install -U "qai-hub-models[llama-v3-2-3b-instruct]"
python -m qai_hub configure --api_token YOUR_TOKEN
python -m qai_hub_models.models.llama_v3_2_3b_instruct.export --chipset qualcomm-snapdragon-x-elite --skip-inferencing --skip-profiling --output-dir genie_bundle
```

Then copy the Genie runtime (`genie-t2t-run.exe` and DLLs) into `genie_bundle/` as in [genie_bundle/README.md](../genie_bundle/README.md).

---

## Vision / object detection

Useful for drone-style perception (already using YOLO; these are Qualcomm-optimized alternatives or additions).

| Model | Use case |
|-------|----------|
| **YOLOv5 / v6 / v7 / v8 / v10 / v11** | Object detection (alternatives to current YOLO pipeline). |
| **CenterNet-2D** | Object detection. |
| **BEVDet / BEVFusion** | Bird’s-eye view (driver assistance; can inspire tactical map). |
| **SAM2** | Segmentation. |
| **Depth-Anything** | Depth estimation (for “depth on boxes” or 3D). |
| **MobileVit / EfficientVit** | Lightweight image classification/backbones. |

Install examples:

```powershell
pip install "qai-hub-models[yolo-v8]"
# or
pip install "qai-hub-models[depth-anything]"
```

---

## Other

- **Whisper** (tiny/base/small/large-v3-turbo): speech-to-text for voice queries (alternative to current Whisper usage).
- **BERT / ALBERT**: if you add NLP (e.g. NER, re-ranking) later.

---

## Links

- **All models:** https://aihub.qualcomm.com/models  
- **Llama-v3.2-3B-Instruct:** https://aihub.qualcomm.com/compute/models/llama_v3_2_3b_instruct  
- **LLM Chat Windows app:** https://aihub.qualcomm.com/apps/chatapp_windows  
- **GitHub (export scripts):** https://github.com/quic/ai-hub-models  
- **PyPI:** https://pypi.org/project/qai-hub-models/

---

## Summary

- **Agent (agentic talk) on Qualcomm:** use **Llama-v3.2-3B-Instruct** from Qualcomm AI Hub; export to `genie_bundle` and run with Genie, or use Ollama on a non-Snapdragon machine (see [AGENT_OLLAMA.md](AGENT_OLLAMA.md)).  
- **Vision/detection:** explore **YOLO v8/v10/v11**, **CenterNet-2D**, or **Depth-Anything** via `qai-hub-models` if you want Qualcomm-optimized models alongside or instead of current YOLO.
