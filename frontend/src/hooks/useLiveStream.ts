import { useEffect, useState } from "react";
import { authStore } from "../api/client";

const WS_URL = import.meta.env.VITE_WS_URL || "ws://localhost:8000/ws/stream";

export type LiveSnapshot = {
  ts: string;
  telemetry: Array<Record<string, unknown>>;
  trade_flow: Array<Record<string, unknown>>;
  gnn_predictions: Array<Record<string, unknown>>;
};

export function useLiveStream() {
  const [data, setData] = useState<LiveSnapshot | null>(null);
  const [status, setStatus] = useState("disconnected");

  useEffect(() => {
    const token = authStore.getToken();
    if (!token) {
      return;
    }
    const ws = new WebSocket(`${WS_URL}?token=${token}`);
    setStatus("connecting");

    ws.onopen = () => setStatus("connected");
    ws.onmessage = (event) => {
      setData(JSON.parse(event.data));
    };
    ws.onerror = () => setStatus("error");
    ws.onclose = () => setStatus("disconnected");

    return () => {
      ws.close();
    };
  }, []);

  return { data, status };
}
