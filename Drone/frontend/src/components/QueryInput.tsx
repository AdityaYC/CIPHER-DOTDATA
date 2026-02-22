import { useState, useRef, useCallback } from "react";
import { Search, Mic, Square } from "lucide-react";
import { DEFAULT_AGENT_COUNT } from "../config";

const MAX_VOICE_SEC = 8;

type QueryInputProps = {
  onSubmit: (query: string, numAgents: number) => void;
  onVoiceSubmit?: (blob: Blob, numAgents: number) => void;
  disabled: boolean;
  placeholder?: string;
};

export function QueryInput({ onSubmit, disabled, onVoiceSubmit, placeholder = "DESCRIBE WHAT TO FIND..." }: QueryInputProps) {
  const [value, setValue] = useState("");
  const [numAgents, setNumAgents] = useState(DEFAULT_AGENT_COUNT);
  const [agentInputValue, setAgentInputValue] = useState(String(DEFAULT_AGENT_COUNT));
  const [recording, setRecording] = useState(false);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const stopTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const numAgentsRef = useRef(numAgents);
  numAgentsRef.current = numAgents;

  const stopRecording = useCallback(() => {
    if (stopTimeoutRef.current) {
      clearTimeout(stopTimeoutRef.current);
      stopTimeoutRef.current = null;
    }
    const rec = recorderRef.current;
    if (!rec || rec.state === "inactive") return;
    recorderRef.current = null;
    setRecording(false);
    // Request any buffered data before stop so onstop receives full recording
    if (typeof rec.requestData === "function") rec.requestData();
    rec.stop();
  }, []);

  const handleVoiceClick = useCallback(async () => {
    if (disabled || !onVoiceSubmit) return;
    if (recording) {
      stopRecording();
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mime = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : "audio/webm";
      const rec = new MediaRecorder(stream);
      chunksRef.current = [];
      rec.ondataavailable = (e) => {
        if (e.data.size) chunksRef.current.push(e.data);
      };
      rec.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(chunksRef.current, { type: mime });
        // Always submit so user gets backend message (e.g. "Audio was empty" or transcription)
        onVoiceSubmit(blob, numAgentsRef.current);
      };
      rec.start(100);
      recorderRef.current = rec;
      setRecording(true);
      stopTimeoutRef.current = setTimeout(() => stopRecording(), MAX_VOICE_SEC * 1000);
    } catch (e) {
      console.warn("Microphone access failed:", e);
    }
  }, [disabled, onVoiceSubmit, recording, stopRecording]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = value.trim();
    if (trimmed && !disabled) {
      onSubmit(trimmed, numAgents);
    }
  };

  return (
    <form className="agent-query-bar" onSubmit={handleSubmit}>
      <input
        className="agent-query-input"
        type="text"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        placeholder={placeholder}
        disabled={disabled}
      />
      <input
        className="agent-count-input"
        type="number"
        min={1}
        max={8}
        value={agentInputValue}
        onChange={(e) => setAgentInputValue(e.target.value)}
        onBlur={() => {
          const n = parseInt(agentInputValue, 10);
          const clamped = isNaN(n) ? DEFAULT_AGENT_COUNT : Math.max(1, Math.min(8, n));
          setNumAgents(clamped);
          setAgentInputValue(String(clamped));
        }}
        disabled={disabled}
        title="Number of agents"
      />
      {onVoiceSubmit && (
        <button
          type="button"
          className={`agent-query-mic ${recording ? "recording" : ""}`}
          onClick={handleVoiceClick}
          disabled={disabled}
          title={recording ? "Stop recording" : "Ask by voice (hold or click again to stop)"}
        >
          {recording ? (
            <Square size={16} strokeWidth={2.5} style={{ marginRight: 6, verticalAlign: "middle" }} />
          ) : (
            <Mic size={16} strokeWidth={2.5} style={{ marginRight: 6, verticalAlign: "middle" }} />
          )}
          {recording ? "STOP" : "VOICE"}
        </button>
      )}
      <button
        className="agent-query-submit"
        type="submit"
        disabled={disabled || !value.trim()}
      >
        <Search size={16} strokeWidth={2.5} style={{ marginRight: 6, verticalAlign: "middle" }} />
        SEARCH
      </button>
    </form>
  );
}
