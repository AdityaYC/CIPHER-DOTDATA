"""Drone Vision – parallel building-exploration agents powered by a vision LLM.

Usage:
    modal deploy agents/agents.py

Requires app.py to be deployed first (provides the GetImage class).
"""

import base64
import json
import time
import uuid

import modal
# from starlette.responses import StreamingResponse, Response

# ---------------------------------------------------------------------------
# Model registry – add new vision-language models here
# ---------------------------------------------------------------------------

SUPPORTED_MODELS: dict[str, str] = {
    "qwen3-vl-30b-a3b-thinking-fp8": "Qwen/Qwen3-VL-30B-A3B-Thinking-FP8",
    "qwen3-vl-2b": "Qwen/Qwen3-VL-2B-Instruct",
    # "qwen3-vl-8b": "Qwen/Qwen3-VL-8B-Instruct",
    # "qwen2.5-vl-3b": "Qwen/Qwen2.5-VL-3B-Instruct",
}
DEFAULT_MODEL = "qwen3-vl-30b-a3b-thinking-fp8"
MODEL_ID = SUPPORTED_MODELS[DEFAULT_MODEL]

MAX_STEPS = 15

# ---------------------------------------------------------------------------
# Trajectory bounds (from trajectory_postprocessed.csv) – keeps agents
# inside the mapped building volume.
# ---------------------------------------------------------------------------

BOUNDS = {
    "x": (-200, 200),
    "y": (-200, 200),
    "z": (-100, 100),
}

# ---------------------------------------------------------------------------
# Modal resources
# ---------------------------------------------------------------------------

app = modal.App("vision-agents-old")

model_vol = modal.Volume.from_name("vision-model-cache", create_if_missing=True)
cancel_dict = modal.Dict.from_name("vision-cancel", create_if_missing=True)

MODEL_DIR = "/model-cache"


def download_model():
    """Pre-download model weights into the volume (runs at image build)."""
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL_ID, local_dir=f"{MODEL_DIR}/{MODEL_ID}")


# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------

agent_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.11.0",
        "transformers",
        "qwen-vl-utils==0.0.14",
        "Pillow",
        "torch",
        "huggingface_hub",
    )
    .run_function(download_model, volumes={MODEL_DIR: model_vol})
)

# ---------------------------------------------------------------------------
# System prompt template
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a building-exploration agent.  You navigate by requesting camera
views at specific 3D positions and yaw angles.

## World info
- yaw is in degrees (0-360).  0 = initial forward direction.

## Your task
The user asked: "{query}"
Explore the building to find what they asked for.

## How to respond
Each turn you will receive an image from your current position.  Look at
the image, consider your trajectory so far, then output **only** a JSON
object (no markdown, no extra text) in one of these two forms:

If you have NOT found the target:
{{"action": "move", "x": <float>, "y": <float>, "z": <float>, "yaw": <float>, "reasoning": "<1-2 sentences>"}}

If you CAN SEE the target in the current image:
{{"action": "found", "description": "<what and where you see it>", "confidence": "<low|medium|high>", "evidence": ["<visual cue 1>", "<visual cue 2>"]}}

Use "found" only when confidence is HIGH based on direct visual evidence in
the current image.  If confidence is not high, choose "move" to collect a
better viewpoint.

Before using "found", self-check all of these:
1) The object's identity matches the query (not just similar-looking).
2) Its location in the image is explicit (left/center/right + nearby context).
3) You can cite at least two concrete visual attributes (shape/color/text/context).

Do NOT use "found" if the object is partially occluded, blurry, too far, or
ambiguous with similar objects.  In those cases, "move" by rotating slightly,
moving closer, or changing viewpoint.

For "found.description", include:
- what the object is,
- where it is in the frame relative to landmarks,
- the phrase "high confidence".

Do NOT revisit positions you have already been to.
"""


# ---------------------------------------------------------------------------
# AgentRunner class
# ---------------------------------------------------------------------------


@app.cls(
    image=agent_image,
    gpu="H200",
    volumes={MODEL_DIR: model_vol},
    timeout=600,
    scaledown_window=300,
)
class AgentRunner:
    """Hosts the VLM and exposes send_agent for multi-turn exploration."""

    @modal.enter()
    def setup(self):
        from vllm import LLM

        self.llm = LLM(
            model=f"{MODEL_DIR}/{MODEL_ID}",
            trust_remote_code=True,
            max_model_len=16384,
            dtype="half",
            enable_prefix_caching=True,
        )
        self.get_image_cls = modal.Cls.from_name("vision-old", "GetImage")

    # ------------------------------------------------------------------ #

    @modal.method()
    def send_agent(
        self,
        query: str,
        start_x: float,
        start_y: float,
        start_z: float,
        start_yaw: float,
        agent_id: int,
        session_key: str,
    ) -> dict:
        """Run one exploration agent.

        Each turn the LLM receives a fresh two-message prompt:
          1. System: original query/instructions + a text summary of
             every prior step (position, reasoning).
          2. User: only the CURRENT image + current position.

        This keeps context small (one image per call) while giving the
        LLM full memory of where it has been and what it decided.

        Returns dict with keys:
            found, agent_id, description, final_image_b64,
            steps, trajectory
        """
        from vllm import SamplingParams

        x, y, z, yaw = start_x, start_y, start_z, start_yaw
        trajectory: list[dict] = []
        history_lines: list[str] = []   # text-only memory of past turns
        last_image_b64 = ""
        sampling = SamplingParams(temperature=0.7, max_tokens=300)

        print(f"\n{'='*60}")
        print(f"[Agent {agent_id}] START  query={query!r}")
        print(f"[Agent {agent_id}]        pos=({x:.2f}, {y:.2f}, {z:.2f})  yaw={yaw:.1f}")
        print(f"{'='*60}")

        base_sys_text = SYSTEM_PROMPT.format(query=query)

        for step in range(MAX_STEPS):
            # -- cancel check -------------------------------------------
            try:
                if cancel_dict[session_key]:
                    print(f"[Agent {agent_id}] CANCELLED at step {step}")
                    return self._result(False, agent_id, "Cancelled",
                                        last_image_b64, step, trajectory)
            except KeyError:
                pass

            # -- call getImageRemote ------------------------------------
            print(f"\n[Agent {agent_id}] Step {step}")
            print(f"[Agent {agent_id}]   getImageRemote(x={x:.4f}, y={y:.4f}, z={z:.4f}, yaw={yaw:.2f})")

            get_image = self.get_image_cls()
            result = get_image.getImageRemote.remote(x, y, z, yaw)

            src_idx = result["source_idx"]
            src_ts = result.get("source_timestamp_s", result.get("source_filename", "?"))
            img_bytes: bytes = result["image_png"]
            img_b64 = base64.b64encode(img_bytes).decode("ascii")
            last_image_b64 = img_b64

            print(f"[Agent {agent_id}]   -> source_idx={src_idx}  source={src_ts}  image_bytes={len(img_bytes)}")

            trajectory.append({
                "x": x, "y": y, "z": z, "yaw": yaw,
                "step": step, "source": src_ts,
            })

            # -- build system prompt with trajectory summary ------------
            sys_text = base_sys_text
            if history_lines:
                sys_text += (
                    "\n\n## Trajectory so far\n"
                    + "\n".join(history_lines)
                )

            # -- fresh prompt: system + current image only --------------
            messages = [
                {"role": "system", "content": [{"type": "text", "text": sys_text}]},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                        },
                        {
                            "type": "text",
                            "text": (
                                f"Position: ({x:.2f}, {y:.2f}, {z:.2f}), yaw={yaw:.1f}. "
                                f"Step {step}/{MAX_STEPS}."
                            ),
                        },
                    ],
                },
            ]

            # -- VLM inference ------------------------------------------
            outputs = self.llm.chat(messages, sampling_params=sampling)
            raw_text = outputs[0].outputs[0].text.strip()
            print(f"[Agent {agent_id}]   LLM reasoning: {raw_text}")

            # -- parse JSON action --------------------------------------
            action = self._parse_action(raw_text)
            if action is None:
                print(f"[Agent {agent_id}]   (parse failed – rotating 30 deg)")
                history_lines.append(
                    f"- Step {step}: pos=({x:.2f},{y:.2f},{z:.2f}) yaw={yaw:.1f} "
                    f"— could not decide, rotated 30 deg"
                )
                yaw = (yaw + 30) % 360
                continue

            print(f"[Agent {agent_id}]   Parsed action: {action}")

            if action.get("action") == "found":
                ok, validation_msg = self._validate_found_action(action)
                if not ok:
                    print(f"[Agent {agent_id}]   (reject found: {validation_msg})")
                    history_lines.append(
                        f"- Step {step}: pos=({x:.2f},{y:.2f},{z:.2f}) yaw={yaw:.1f} "
                        f"— rejected found ({validation_msg}), rotated 20 deg"
                    )
                    yaw = (yaw + 20) % 360
                    continue

                desc = str(action.get("description", ""))
                print(f"\n[Agent {agent_id}] *** FOUND at step {step}: {desc} ***\n")
                return self._result(True, agent_id, desc,
                                    last_image_b64, step, trajectory)

            if action.get("action") == "move":
                reasoning = action.get("reasoning", "")
                history_lines.append(
                    f"- Step {step}: pos=({x:.2f},{y:.2f},{z:.2f}) yaw={yaw:.1f} — {reasoning}"
                )
                x = self._clamp(float(action.get("x", x)), *BOUNDS["x"])
                y = self._clamp(float(action.get("y", y)), *BOUNDS["y"])
                z = self._clamp(float(action.get("z", z)), *BOUNDS["z"])
                yaw = float(action.get("yaw", yaw)) % 360
            else:
                history_lines.append(
                    f"- Step {step}: pos=({x:.2f},{y:.2f},{z:.2f}) yaw={yaw:.1f} "
                    f"— unknown action, rotated 30 deg"
                )
                yaw = (yaw + 30) % 360

        print(f"[Agent {agent_id}] Max steps reached")
        return self._result(False, agent_id,
                            "Max steps reached without finding target",
                            last_image_b64, MAX_STEPS, trajectory)

    # ------------------------------------------------------------------ #
    # Streaming version
    # ------------------------------------------------------------------ #

    @modal.method()
    def send_agent_streaming(
        self,
        query: str,
        start_x: float,
        start_y: float,
        start_z: float,
        start_yaw: float,
        agent_id: int,
        session_key: str,
    ):
        """Generator version of send_agent — yields step-by-step events."""
        from vllm import SamplingParams
        from io import BytesIO
        from PIL import Image

        x, y, z, yaw = start_x, start_y, start_z, start_yaw
        trajectory = []
        history_lines = []
        last_image_b64 = ""
        sampling = SamplingParams(temperature=0.7, max_tokens=300)

        base_sys_text = SYSTEM_PROMPT.format(query=query)

        for step in range(MAX_STEPS):
            # Cancel check
            try:
                if cancel_dict[session_key]:
                    return
            except KeyError:
                pass

            # Get image
            get_image = self.get_image_cls()
            result = get_image.getImageRemote.remote(x, y, z, yaw)
            img_bytes = result["image_png"]
            img_b64 = base64.b64encode(img_bytes).decode("ascii")
            last_image_b64 = img_b64

            # Downscale for streaming (256x256)
            img = Image.open(BytesIO(img_bytes))
            img_small = img.resize((256, 256), Image.LANCZOS)
            buf = BytesIO()
            img_small.save(buf, format="PNG")
            small_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

            trajectory.append({"x": x, "y": y, "z": z, "yaw": yaw, "step": step})

            # Build system prompt with trajectory summary
            sys_text = base_sys_text
            if history_lines:
                sys_text += "\n\n## Trajectory so far\n" + "\n".join(history_lines)

            # Fresh prompt: system + current image only
            messages = [
                {"role": "system", "content": [{"type": "text", "text": sys_text}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                        {"type": "text", "text": f"Position: ({x:.2f}, {y:.2f}, {z:.2f}), yaw={yaw:.1f}. Step {step}/{MAX_STEPS}."},
                    ],
                },
            ]

            # VLM inference
            outputs = self.llm.chat(messages, sampling_params=sampling)
            raw_text = outputs[0].outputs[0].text.strip()

            # Parse action
            action = self._parse_action(raw_text)
            reasoning = ""
            action_type = "move"

            if action is None:
                reasoning = "(parse failed - rotating)"
                history_lines.append(
                    f"- Step {step}: pos=({x:.2f},{y:.2f},{z:.2f}) yaw={yaw:.1f} "
                    f"— could not decide, rotated 30 deg"
                )
                yaw = (yaw + 30) % 360
            elif action.get("action") == "found":
                ok, validation_msg = self._validate_found_action(action)
                if not ok:
                    reasoning = f"(rejected found: {validation_msg})"
                    history_lines.append(
                        f"- Step {step}: pos=({x:.2f},{y:.2f},{z:.2f}) yaw={yaw:.1f} "
                        f"— rejected found ({validation_msg}), rotated 20 deg"
                    )
                    yaw = (yaw + 20) % 360
                else:
                    action_type = "found"
                    reasoning = action.get("description", "")
            elif action.get("action") == "move":
                reasoning = action.get("reasoning", "")
                action_type = "move"
                history_lines.append(
                    f"- Step {step}: pos=({x:.2f},{y:.2f},{z:.2f}) yaw={yaw:.1f} — {reasoning}"
                )
            else:
                reasoning = raw_text[:100]
                history_lines.append(
                    f"- Step {step}: pos=({x:.2f},{y:.2f},{z:.2f}) yaw={yaw:.1f} "
                    f"— unknown action, rotated 30 deg"
                )
                yaw = (yaw + 30) % 360

            # Yield the step event
            yield {
                "type": "agent_step",
                "agent_id": agent_id,
                "step": step,
                "total_steps": MAX_STEPS,
                "pose": {"x": x, "y": y, "z": z, "yaw": yaw},
                "image_b64": small_b64,
                "reasoning": reasoning,
                "action": action_type,
            }

            if action_type == "found":
                yield {
                    "type": "agent_found",
                    "agent_id": agent_id,
                    "description": reasoning,
                    "final_image_b64": last_image_b64,
                    "steps": step + 1,
                    "trajectory": trajectory,
                }
                return

            # Apply move
            if action and action.get("action") == "move":
                x = self._clamp(float(action.get("x", x)), *BOUNDS["x"])
                y = self._clamp(float(action.get("y", y)), *BOUNDS["y"])
                z = self._clamp(float(action.get("z", z)), *BOUNDS["z"])
                yaw = float(action.get("yaw", yaw)) % 360

        # Max steps reached
        yield {
            "type": "agent_done",
            "agent_id": agent_id,
            "found": False,
            "steps": MAX_STEPS,
            "trajectory": trajectory,
        }

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _result(found, agent_id, description, image_b64, steps, trajectory):
        return {
            "found": found,
            "agent_id": agent_id,
            "description": description,
            "final_image_b64": image_b64,
            "steps": steps,
            "trajectory": trajectory,
        }

    @staticmethod
    def _clamp(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    @staticmethod
    def _parse_action(text: str) -> dict | None:
        """Best-effort JSON extraction from LLM output."""
        # Strip markdown fences if present
        if "```" in text:
            parts = text.split("```")
            if len(parts) >= 2:
                text = parts[1]
                if text.startswith("json"):
                    text = text[4:]
        # Try to find JSON object boundaries
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _validate_found_action(action: dict) -> tuple[bool, str]:
        """Require high-confidence structured evidence before accepting found."""
        desc = str(action.get("description", "")).strip()
        if not desc:
            return False, "missing description"

        confidence = str(action.get("confidence", "")).strip().lower()
        if confidence != "high":
            return False, f'confidence must be "high" (got {confidence or "missing"})'

        evidence = action.get("evidence")
        if not isinstance(evidence, list):
            return False, "missing evidence list"

        evidence_items = [str(item).strip() for item in evidence if str(item).strip()]
        if len(evidence_items) < 2:
            return False, "need at least 2 evidence items"

        return True, "ok"


# ---------------------------------------------------------------------------
# spawn_agent – Modal function that runs a single agent
# ---------------------------------------------------------------------------


@app.function(image=agent_image, gpu="H200", volumes={MODEL_DIR: model_vol}, timeout=600)
def spawn_agent(
    query: str,
    x: float,
    y: float,
    z: float,
    yaw: float,
    agent_id: int,
    session_key: str,
) -> dict:
    """Run a single exploration agent as a standalone Modal function.

    This is the unit of parallelism — call .spawn() N times from main()
    to search concurrently.

    Returns dict with keys:
        found, agent_id, description, final_image_b64, steps, trajectory
    """
    runner = AgentRunner()
    return runner.send_agent.remote(
        query=query,
        start_x=x, start_y=y, start_z=z,
        start_yaw=yaw,
        agent_id=agent_id,
        session_key=session_key,
    )


# ---------------------------------------------------------------------------
# SSE streaming endpoint
# ---------------------------------------------------------------------------


@app.function(image=agent_image, timeout=600)
@modal.fastapi_endpoint(method="POST")
def stream_agents(request: dict):
    """SSE endpoint for streaming agent exploration to the frontend."""
    query = request["query"]
    start_x = request.get("start_x", 0.0)
    start_y = request.get("start_y", 0.0)
    start_z = request.get("start_z", 0.0)
    start_yaw = request.get("start_yaw", 0.0)
    num_agents = request.get("num_agents", 2)

    def event_generator():
        session_key = str(uuid.uuid4())
        cancel_dict[session_key] = False

        # Emit agent_started events
        agent_configs = []
        for i in range(num_agents):
            agent_yaw = (start_yaw + (i % 2) * 180) % 360
            agent_configs.append((i, agent_yaw))
            event = {
                "type": "agent_started",
                "agent_id": i,
                "start_pose": {"x": start_x, "y": start_y, "z": start_z, "yaw": agent_yaw},
            }
            yield f"data: {json.dumps(event)}\n\n"

        # Spawn agents and poll for completion
        runner = AgentRunner()
        handles = []
        for agent_id, agent_yaw in agent_configs:
            h = runner.send_agent.spawn(
                query=query,
                start_x=start_x, start_y=start_y, start_z=start_z,
                start_yaw=agent_yaw,
                agent_id=agent_id,
                session_key=session_key,
            )
            handles.append(h)

        completed = [False] * num_agents
        results = [None] * num_agents
        winner = None

        while not all(completed):
            time.sleep(2)
            for i, h in enumerate(handles):
                if completed[i]:
                    continue
                try:
                    r = h.get(timeout=0)
                except TimeoutError:
                    continue
                except Exception:
                    completed[i] = True
                    error_event = {
                        "type": "error",
                        "agent_id": i,
                        "message": "Agent encountered an error",
                    }
                    yield f"data: {json.dumps(error_event)}\n\n"
                    continue

                completed[i] = True
                results[i] = r

                # Emit trajectory steps retroactively
                for step_data in r.get("trajectory", []):
                    step_event = {
                        "type": "agent_step",
                        "agent_id": i,
                        "step": step_data["step"],
                        "total_steps": MAX_STEPS,
                        "pose": {
                            "x": step_data["x"], "y": step_data["y"],
                            "z": step_data["z"], "yaw": step_data["yaw"],
                        },
                        "image_b64": "",
                        "reasoning": "",
                        "action": "move",
                    }
                    yield f"data: {json.dumps(step_event)}\n\n"

                if r["found"]:
                    found_event = {
                        "type": "agent_found",
                        "agent_id": i,
                        "description": r["description"],
                        "final_image_b64": r.get("final_image_b64", ""),
                        "steps": r["steps"],
                        "trajectory": r["trajectory"],
                    }
                    yield f"data: {json.dumps(found_event)}\n\n"
                    if winner is None:
                        winner = i
                        cancel_dict[session_key] = True
                else:
                    done_event = {
                        "type": "agent_done",
                        "agent_id": i,
                        "found": False,
                        "steps": r["steps"],
                        "trajectory": r["trajectory"],
                    }
                    yield f"data: {json.dumps(done_event)}\n\n"

        # Session complete
        complete_event = {
            "type": "session_complete",
            "winner_agent_id": winner,
            "description": results[winner]["description"] if winner is not None else "No target found",
        }
        yield f"data: {json.dumps(complete_event)}\n\n"

        # Cleanup
        try:
            del cancel_dict[session_key]
        except KeyError:
            pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        },
    )


@app.function(image=agent_image)
@modal.fastapi_endpoint(method="OPTIONS")
def stream_agents_options():
    """Handle CORS preflight requests for the streaming endpoint."""
    return Response(
        content="",
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        },
    )


# ---------------------------------------------------------------------------
# Local entrypoint – launches N agents in parallel, first success wins
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(
    query: str = "find the nearest bathroom",
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    yaw: float = 0.0,
    n: int = 1,
):
    """Launch n parallel agents.  First one to find the target wins.

    Usage:
        modal run agents/agents.py --query "find the nearest bathroom" --n 3
    """
    session_key = str(uuid.uuid4())
    cancel_dict[session_key] = False

    print(f"\n{'#'*60}")
    print(f"# Launching {n} agent(s)  query={query!r}")
    print(f"# start=({x:.2f}, {y:.2f}, {z:.2f})  yaw={yaw:.1f}")
    print(f"# session={session_key}")
    print(f"{'#'*60}\n")

    # Spawn N agents with different starting yaw offsets
    handles = []
    for i in range(n):
        agent_yaw = (yaw + (i % 2) * 180) % 360
        print(f"Spawning agent {i}  yaw={agent_yaw:.1f}")
        h = spawn_agent.spawn(
            query=query,
            x=x, y=y, z=z,
            yaw=agent_yaw,
            agent_id=i,
            session_key=session_key,
        )
        handles.append(h)

    # Poll until one agent finds the target or all finish
    completed = [False] * n
    results: list[dict | None] = [None] * n
    winner = None

    while not all(completed):
        time.sleep(2)
        for i, h in enumerate(handles):
            if completed[i]:
                continue
            try:
                r = h.get(timeout=0)
            except TimeoutError:
                continue
            except Exception as exc:
                print(f"Agent {i} errored: {exc}")
                completed[i] = True
                continue

            completed[i] = True
            results[i] = r
            print(f"\nAgent {i} finished: found={r['found']}  steps={r['steps']}")
            if r.get("description"):
                print(f"  description: {r['description']}")

            if r["found"] and winner is None:
                winner = r
                # Signal all other agents to stop
                cancel_dict[session_key] = True
                print(f"\n>>> Agent {i} found the target – cancelling others <<<\n")

    # Cleanup
    try:
        del cancel_dict[session_key]
    except KeyError:
        pass

    # Print final result
    final = winner
    if final is None:
        final = max(
            (r for r in results if r is not None),
            key=lambda r: r["steps"],
            default=None,
        )

    if final:
        print(f"\n{'='*60}")
        if final["found"]:
            print(f"RESULT: Agent {final['agent_id']} found the target in {final['steps']} steps")
        else:
            print(f"RESULT: No agent found the target. Best effort from agent {final['agent_id']}.")
        print(f"  description: {final['description']}")
        print(f"  trajectory points: {len(final['trajectory'])}")
        if final["final_image_b64"]:
            print(f"  final image: {len(final['final_image_b64'])} chars base64")
        print(f"{'='*60}\n")
    else:
        print("\nAll agents failed.")
