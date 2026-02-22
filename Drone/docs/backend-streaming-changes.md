# Backend API: Session-Based Agent Streaming

The frontend and Poke MCP server both need to connect to agent exploration sessions by ID. The backend exposes two endpoints: one to create a session (starts the agents), and one to observe a session's event stream (SSE with replay).

---

## Endpoints

### 1. Create Session

```
POST /sessions
Content-Type: application/json
```

**Request body:**
```json
{
  "query": "where is the nearest bathroom?",
  "start_x": 0.0,
  "start_y": 0.0,
  "start_z": 0.0,
  "start_yaw": 0.0,
  "num_agents": 2
}
```

**Response (200):**
```json
{
  "session_id": "a1b2c3d4-e5f6-..."
}
```

**Behavior:**
- Generate a UUID session_id
- Spawn N agents in the background (same logic as current `main()` in agents.py)
- Return immediately with the session_id (do NOT block until agents finish)
- Store session metadata in `modal.Dict` keyed by session_id

**Implementation sketch:**
```python
session_events = modal.Dict.from_name("session-events", create_if_missing=True)
session_meta = modal.Dict.from_name("session-meta", create_if_missing=True)

@app.function(image=agent_image, timeout=600)
@modal.web_endpoint(method="POST")
def create_session(request: dict):
    session_id = str(uuid.uuid4())
    query = request["query"]
    num_agents = request.get("num_agents", 2)

    # Store session metadata
    session_meta[session_id] = {
        "query": query,
        "num_agents": num_agents,
        "status": "running",
        "created_at": time.time(),
    }

    # Initialize empty event list
    session_events[session_id] = []

    # Spawn agents in background
    cancel_dict[session_id] = False
    for i in range(num_agents):
        agent_yaw = (request.get("start_yaw", 0.0) + (i % 2) * 180) % 360

        # Emit agent_started event
        _append_event(session_id, {
            "type": "agent_started",
            "agent_id": i,
            "start_pose": {
                "x": request.get("start_x", 0.0),
                "y": request.get("start_y", 0.0),
                "z": request.get("start_z", 0.0),
                "yaw": agent_yaw,
            },
        })

        # Spawn the agent (runs async in background)
        run_agent_with_events.spawn(
            session_id=session_id,
            query=query,
            x=request.get("start_x", 0.0),
            y=request.get("start_y", 0.0),
            z=request.get("start_z", 0.0),
            yaw=agent_yaw,
            agent_id=i,
        )

    return {"session_id": session_id}
```

---

### 2. Observe Session (SSE)

```
GET /sessions/{session_id}/stream
Accept: text/event-stream
```

**Response:** SSE stream (`text/event-stream`)

**Behavior:**
1. Look up `session_id` in the events dict
2. **Replay** all events that have already been emitted (catches up late joiners)
3. **Poll** for new events and stream them as they arrive
4. Close the stream after emitting `session_complete`
5. Return 404 if session_id is unknown

**Critical requirements:**
- **Event replay**: When a client connects, send ALL prior events for this session first. This is essential because the frontend may open the link seconds after poke-mcp created the session.
- **Multiple concurrent readers**: Multiple clients (poke-mcp + frontend) can GET the same session stream simultaneously. Each reader maintains its own cursor into the event list.
- **Polling interval**: Poll the events dict every ~500ms for new events.

**Implementation sketch:**
```python
from starlette.responses import StreamingResponse

@app.function(image=agent_image, timeout=600)
@modal.web_endpoint(method="GET")
def stream_session(session_id: str):
    def event_generator():
        try:
            events = session_events[session_id]
        except KeyError:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Session not found'})}\n\n"
            return

        cursor = 0
        while True:
            # Get current event list
            events = session_events[session_id]

            # Emit any new events since our cursor
            while cursor < len(events):
                yield f"data: {json.dumps(events[cursor])}\n\n"
                event = events[cursor]
                cursor += 1

                # Stop after session_complete
                if event.get("type") == "session_complete":
                    return

            # Poll interval
            time.sleep(0.5)

            # Safety timeout (10 minutes)
            # ... check elapsed time and break if too long

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        },
    )
```

---

## 3. Agent Runner with Events

Modify the agent execution to append events to the session's event list instead of (or in addition to) returning a final result.

```python
@app.function(image=agent_image, gpu="H200", volumes={MODEL_DIR: model_vol}, timeout=600)
def run_agent_with_events(
    session_id: str,
    query: str,
    x: float, y: float, z: float, yaw: float,
    agent_id: int,
):
    """Run one agent and append events to the session event list."""
    runner = AgentRunner()
    # ... same logic as send_agent, but after each step:

    # After each VLM inference step:
    _append_event(session_id, {
        "type": "agent_step",
        "agent_id": agent_id,
        "step": step,
        "total_steps": MAX_STEPS,
        "pose": {"x": x, "y": y, "z": z, "yaw": yaw},
        "image_b64": small_b64,      # 256x256 for streaming
        "reasoning": reasoning,
        "action": action_type,
    })

    # On found:
    _append_event(session_id, {
        "type": "agent_found",
        "agent_id": agent_id,
        "description": description,
        "final_image_b64": full_res_b64,
        "steps": step + 1,
        "trajectory": trajectory,
    })
    # Cancel other agents
    cancel_dict[session_id] = True

    # On done (max steps):
    _append_event(session_id, {
        "type": "agent_done",
        "agent_id": agent_id,
        "found": False,
        "steps": MAX_STEPS,
        "trajectory": trajectory,
    })

    # Check if all agents done → emit session_complete
    # (Use a counter in modal.Dict to track completions)
```

**Helper to append events atomically:**
```python
def _append_event(session_id: str, event: dict):
    """Append an event to the session's event list."""
    events = session_events[session_id]
    events.append(event)
    session_events[session_id] = events  # Write back to Dict
```

**Emitting session_complete:**
Use an atomic counter in `modal.Dict` to track how many agents have finished. When the last agent finishes (or one finds the target and cancels others), emit:

```python
completion_key = f"{session_id}:completed_count"
# Increment completed count (use a separate dict or key pattern)
# When count == num_agents OR a winner was found and all cancelled:
_append_event(session_id, {
    "type": "session_complete",
    "winner_agent_id": winner_id,  # or null
    "description": winner_description or "No target found",
})
```

---

## 4. CORS

Both endpoints need CORS headers for the frontend to connect from a different origin:

```
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: GET, POST, OPTIONS
Access-Control-Allow-Headers: Content-Type
```

Handle OPTIONS preflight for the POST endpoint:
```python
@app.function(image=agent_image)
@modal.web_endpoint(method="OPTIONS")
def sessions_options():
    from starlette.responses import Response
    return Response(
        content="",
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        },
    )
```

---

## 5. Image Size Optimization

- **Step events**: Send 256x256 PNG (~10-30KB base64) — good for thumbnails and live viewport
- **Found events**: Send full-resolution image — this is the final result

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

## 6. Session Cleanup

Sessions should auto-expire. Simple approach:
- Store `created_at` timestamp in session metadata
- Background cleanup function (or lazy cleanup on access) deletes sessions older than 15 minutes
- The event dict entries get deleted along with the metadata

---

## 7. Event Type Reference

These event types are unchanged from the frontend's TypeScript definitions:

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
  total_steps: number;
  pose: { x: number; y: number; z: number; yaw: number };
  image_b64: string;
  reasoning: string;
  action: "move" | "found";
};

type AgentFoundEvent = {
  type: "agent_found";
  agent_id: number;
  description: string;
  final_image_b64: string;
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

## 8. Frontend Configuration

The frontend reads endpoints from environment variables:

```
VITE_AGENT_API_URL=https://your-deployment--agents-api.modal.run
```

The frontend will call:
- `POST {VITE_AGENT_API_URL}/sessions` to create a session
- `GET {VITE_AGENT_API_URL}/sessions/{session_id}/stream` to observe

If `VITE_AGENT_API_URL` is not set, the frontend falls back to `VITE_AGENT_STREAM_URL` (the old single-endpoint URL) for backward compatibility.
