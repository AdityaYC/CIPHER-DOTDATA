import { useEffect, useRef, useState } from "react";

interface IPhoneFeedProps {
  streamUrl?: string;
  className?: string;
}

export function IPhoneFeed({ 
  streamUrl = "http://localhost:8002/stream",
  className = ""
}: IPhoneFeedProps) {
  const imgRef = useRef<HTMLImageElement>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;

    // Method 1: Use MJPEG stream (most reliable for continuous video)
    if (imgRef.current) {
      imgRef.current.src = streamUrl;
      
      imgRef.current.onload = () => {
        if (mounted) {
          setIsConnected(true);
          setError(null);
        }
      };

      imgRef.current.onerror = () => {
        if (mounted) {
          setIsConnected(false);
          setError("Cannot connect to iPhone camera stream");
        }
      };
    }

    return () => {
      mounted = false;
    };
  }, [streamUrl]);

  return (
    <div className={`iphone-feed ${className}`}>
      {error && (
        <div className="feed-error">
          <p>{error}</p>
          <p className="feed-error-hint">
            Make sure iPhone stream server is running on port 8002
          </p>
        </div>
      )}
      {!isConnected && !error && (
        <div className="feed-loading">
          <div className="loading-spinner" />
          <p>Connecting to iPhone camera...</p>
        </div>
      )}
      <img
        ref={imgRef}
        alt="iPhone Camera Feed"
        className={`feed-image ${isConnected ? 'connected' : ''}`}
        style={{ display: isConnected ? 'block' : 'none' }}
      />
      {isConnected && (
        <div className="feed-status">
          <span className="status-indicator live" />
          <span>LIVE</span>
        </div>
      )}
    </div>
  );
}
