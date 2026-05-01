import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { authStore, login } from "../api/client";

export default function Login() {
  const navigate = useNavigate();
  const [username, setUsername] = useState("tony");
  const [password, setPassword] = useState("operator");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const response = await login(username, password);
      authStore.setAuth(response.access_token, response.role, response.node_id);
      navigate("/");
    } catch (err) {
      setError((err as Error).message || "Login failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <div className="card login-panel">
        <h2>Sign in</h2>
        <p className="muted">Use the Phase 6 demo users to access each dashboard.</p>
        <form onSubmit={handleSubmit} className="grid" style={{ gap: "12px" }}>
          <label>
            <div className="muted">Username</div>
            <input
              className="input"
              value={username}
              onChange={(event) => setUsername(event.target.value)}
            />
          </label>
          <label>
            <div className="muted">Password</div>
            <input
              className="input"
              type="password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
            />
          </label>
          {error && <div className="warning">{error}</div>}
          <button className="button" type="submit" disabled={loading}>
            {loading ? "Signing in..." : "Sign in"}
          </button>
        </form>
        <div className="muted" style={{ marginTop: "16px" }}>
          <div>Operator: tony / operator</div>
          <div>Homeowner: reggie / homeowner</div>
          <div>Community: luigi / community</div>
        </div>
      </div>
    </div>
  );
}
