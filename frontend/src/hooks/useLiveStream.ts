import { useEffect, useRef, useState } from "react";
import { authStore } from "../api/client";

const WS_URL = import.meta.env.VITE_WS_URL || "/ws/stream";

export type LiveSnapshot = {
  ts: string;
  telemetry: Array<Record<string, unknown>>;
  trade_flow: Array<Record<string, unknown>>;
  gnn_predictions: Array<Record<string, unknown>>;
};

export function useLiveStream() {
  const [data, setData] = useState<LiveSnapshot | null>(null);
  const [status, setStatus] = useState("disconnected");
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const token = authStore.getToken();
    if (!token) {
      return;
    }

    // Close stale connection from a previous effect run (Strict Mode)
    if (wsRef.current) {
      wsRef.current.close();
    }

    const ws = new WebSocket(`${WS_URL}?token=${token}`);
    wsRef.current = ws;
    setStatus("connecting");

    ws.onopen = () => {
      if (wsRef.current === ws) setStatus("connected");
    };
    ws.onmessage = (event) => {
      if (wsRef.current === ws) setData(JSON.parse(event.data));
    };
    ws.onerror = () => {
      if (wsRef.current === ws) setStatus("error");
    };
    ws.onclose = () => {
      if (wsRef.current === ws) {
        wsRef.current = null;
        setStatus("disconnected");
      }
    };

    return () => {
      if (wsRef.current === ws) {
        wsRef.current = null;
        ws.close();
      }
    };
  }, []);

  return { data, status };
}
