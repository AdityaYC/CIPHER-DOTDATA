# Agent tab — use Ollama for agentic talk

The Agent tab (video + questions) uses a local LLM when available. **Recommended:** Ollama + Llama.

## Which model to use for agentic talk

| Model | Command | Use when |
|-------|---------|----------|
| **llama3.2** (3B) | `ollama run llama3.2` | Default: fast, good for Q&A and agentic answers. |
| **llama3.1** (8B) | `ollama run llama3.1` | Higher quality when your machine can run it. |

## Setup

1. Install [Ollama](https://ollama.com) (one-time).
2. In a terminal, run one of:
   - `ollama run llama3.2`
   - `ollama run llama3.1`
3. Leave Ollama running (it serves `http://localhost:11434`).
4. Start (or restart) the app backend so it can use Ollama.

The backend tries **Ollama first**, then Genie (Qualcomm bundle) if present, then vector-search/manual fallback.

## Optional

- Use a different model: set env `OLLAMA_AGENT_MODEL` (e.g. `OLLAMA_AGENT_MODEL=llama3.1`).
- Ollama on another host: set `OLLAMA_HOST=http://other-pc:11434`.
