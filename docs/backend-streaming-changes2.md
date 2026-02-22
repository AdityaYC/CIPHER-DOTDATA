# Backend Changes Required for Agent Streaming

The frontend now has a full streaming agent UI. It expects to connect to a backend SSE endpoint that streams agent exploration events in real-time. Here's what needs to change in `agents/agents.py`.

---

## 1. New SSE Streaming Endpoint

Create a new Modal web endpoint that accepts a POST request and returns a streaming SSE response.

**File:** `agents/agents.py` (or a new `agents/stream.py`)

```python
from starlette.responses import StreamingResponse
import json

@app.function(image=agent_image, timeout=600)
@modal.web_endpoint(method="POST")
def stream_agents(request: dict):
    """SSE endpoint for streaming agent exploration to the frontend."""
    query = request["query"]
    start_x = request.get("start_x", 0.0)
    start_y = request.get("start_y", 0.0)
    start_z = request.get("start_z", 0.0)
    start_yaw = request.get("start_yaw", 0.0)
    num_agents = request.get("num_agents", 2)

    def event_generator():
        # ... see section 3 below
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
```

---

## 2. New `send_agent_streaming()` Method

Add a generator version of `AgentRunner.send_agent()` that yields events after each step instead of only returning a final result.

**Add to `AgentRunner` class:**

```python
@modal.method()
def send_agent_streaming(
    self,
    query: str,
    start_x: float, start_y: float, start_z: float, start_yaw: float,
    agent_id: int,
    session_key: str,
):
    """Generator version of send_agent — yields step-by-step events."""
    from vllm import SamplingParams
    import base64
    from io import BytesIO
    from PIL import Image

    x, y, z, yaw = start_x, start_y, start_z, start_yaw
    trajectory = []
    last_image_b64 = ""
    sampling = SamplingParams(temperature=0.7, max_tokens=300)

    sys_text = SYSTEM_PROMPT.format(query=query)
    messages = [{"role": "system", "content": [{"type": "text", "text": sys_text}]}]

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

        # Optionally downscale for streaming (256x256)
        img = Image.open(BytesIO(img_bytes))
        img_small = img.resize((256, 256), Image.LANCZOS)
        buf = BytesIO()
        img_small.save(buf, format="PNG")
        small_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        trajectory.append({"x": x, "y": y, "z": z, "yaw": yaw, "step": step})

        # Append image to conversation
        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                {"type": "text", "text": f"Position: ({x:.2f}, {y:.2f}, {z:.2f}), yaw={yaw:.1f}. Step {step}/{MAX_STEPS}."},
            ],
        })

        # VLM inference
        outputs = self.llm.chat(messages, sampling_params=sampling)
        raw_text = outputs[0].outputs[0].text.strip()
        messages.append({"role": "assistant", "content": raw_text})

        # Parse action
        action = self._parse_action(raw_text)
        reasoning = ""
        action_type = "move"

        if action is None:
            reasoning = "(parse failed - rotating)"
            yaw = (yaw + 30) % 360
        elif action.get("action") == "found":
            action_type = "found"
            reasoning = action.get("description", "")
        elif action.get("action") == "move":
            reasoning = action.get("reasoning", "")
            action_type = "move"
        else:
            reasoning = raw_text[:100]
            yaw = (yaw + 30) % 360

        # >>> YIELD the step event <<<
        yield {
            "type": "agent_step",
            "agent_id": agent_id,
            "step": step,
            "total_steps": MAX_STEPS,
            "pose": {"x": x, "y": y, "z": z, "yaw": yaw},
            "image_b64": small_b64,      # small image for streaming
            "reasoning": reasoning,
            "action": action_type,
        }

        if action_type == "found":
            # Yield found event with full-res image
            yield {
                "type": "agent_found",
                "agent_id": agent_id,
                "description": reasoning,
                "final_image_b64": last_image_b64,  # full resolution
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
```

---

## 3. Orchestrator Function

The `event_generator()` inside the SSE endpoint needs to:

1. Spawn N agents in parallel
2. Merge their streaming outputs into a single SSE stream
3. Cancel remaining agents when one finds the target

```python
def event_generator():
    import time
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

    # NOTE: The exact parallelism strategy depends on Modal's streaming
    # support. Options:
    #
    # Option A: Use modal.Function.map() if the streaming method supports it
    # Option B: Run agents sequentially (simpler, less parallel)
    # Option C: Use threading + modal .spawn() with periodic polling
    #
    # The simplest approach that works with Modal's current API:
    # Run agents via .spawn(), poll for results, and emit events.
    #
    # For true step-by-step streaming, you may need to use a
    # shared Modal Dict or Queue to pass intermediate results
    # from agent containers back to the streaming endpoint.

    # --- Approach using Modal Dict as message queue ---
    # Each agent writes its step events to:
    #   msg_dict[f"{session_key}:{agent_id}:{step}"] = event_json
    # The orchestrator polls this dict and yields events as they appear.

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

    # Poll for completion (non-streaming fallback)
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
                    "pose": {"x": step_data["x"], "y": step_data["y"],
                             "z": step_data["z"], "yaw": step_data["yaw"]},
                    "image_b64": "",  # images not available retroactively
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
```

---

## 4. CORS Headers

The streaming endpoint MUST include these headers for the frontend to connect:

```
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: POST, OPTIONS
Access-Control-Allow-Headers: Content-Type
```

These are already included in the `StreamingResponse` headers above. You may also need to handle OPTIONS preflight requests:

```python
@app.function(image=agent_image)
@modal.web_endpoint(method="OPTIONS")
def stream_agents_options():
    return Response(
        content="",
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        },
    )
```

---

## 5. Image Size Optimization

To reduce bandwidth during streaming:

- **Step events**: Send 256x256 images (~10-30KB base64) — good enough for the agent card thumbnails and main viewport during exploration
- **Found events**: Send full-resolution images — this is the final result the user cares about

```python
from PIL import Image
from io import BytesIO

def downscale_image(img_bytes: bytes, size=(256, 256)) -> str:
    img = Image.open(BytesIO(img_bytes))
    img = img.resize(size, Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")
```

---

## 6. True Real-Time Streaming (Advanced)

The approach above (polling `.spawn()` handles) sends events in batches when an agent completes, not step-by-step. For true real-time streaming, consider:

**Option A: Modal Dict as message queue**
- Each agent writes step events to a shared `modal.Dict` with keys like `{session_key}:{agent_id}:step:{n}`
- The orchestrator polls this dict every ~500ms and yields new events

**Option B: Generator-based streaming with `send_agent_streaming()`**
- If Modal supports streaming return values from `.remote()` calls (check Modal docs for generator support), use the `send_agent_streaming()` method above directly
- This is the cleanest approach but depends on Modal's support for streaming generators across containers

**Option C: WebSocket via Modal**
- Use FastAPI WebSocket support with Modal's `@modal.asgi_app()` decorator
- More complex but gives bidirectional communication

---

## 7. Event Type Reference

The frontend expects these exact event shapes (TypeScript types):

```typescript
type AgentStartedEvent = {
  type: "agent_started";
  agent_id: number;
  start_pose: { x: number; y: number; z: number; yaw: number };
};

type AgentStepEvent = {
  type: "agent_step";
  agent_id: number;
  step: number;
  total_steps: number;        // MAX_STEPS (15)
  pose: { x: number; y: number; z: number; yaw: number };
  image_b64: string;          // base64 PNG (256x256 for steps)
  reasoning: string;          // LLM reasoning text
  action: "move" | "found";
};

type AgentFoundEvent = {
  type: "agent_found";
  agent_id: number;
  description: string;
  final_image_b64: string;    // full resolution base64 PNG
  steps: number;
  trajectory: Array<{ x: number; y: number; z: number; yaw: number; step: number }>;
};

type AgentDoneEvent = {
  type: "agent_done";
  agent_id: number;
  found: false;
  steps: number;
  trajectory: Array<{ x: number; y: number; z: number; yaw: number; step: number }>;
};

type SessionCompleteEvent = {
  type: "session_complete";
  winner_agent_id: number | null;
  description: string;
};

type AgentErrorEvent = {
  type: "error";
  agent_id?: number;
  message: string;
};
```

Each event is sent as an SSE line:
```
data: {"type":"agent_step","agent_id":0,"step":3,...}\n\n
```

---

## 8. Frontend Config

The frontend reads the streaming endpoint URL from the `VITE_AGENT_STREAM_URL` environment variable. Default is:
```
https://your-deployment--agents-stream.modal.run
```

Update this to match whatever URL Modal assigns to the new endpoint after deployment.
