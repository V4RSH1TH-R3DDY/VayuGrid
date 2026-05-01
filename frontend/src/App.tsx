import { NavLink, Navigate, Route, Routes, useNavigate } from "react-router-dom";
import { authStore } from "./api/client";
import CommunityDashboard from "./views/CommunityDashboard";
import HomeownerDashboard from "./views/HomeownerDashboard";
import Login from "./views/Login";
import OperatorDashboard from "./views/OperatorDashboard";

const RequireAuth = ({ children }: { children: JSX.Element }) => {
  const token = authStore.getToken();
  if (!token) {
    return <Navigate to="/login" replace />;
  }
  return children;
};

const RoleRedirect = () => {
  const role = authStore.getRole();
  const nodeId = authStore.getNodeId();
  if (role === "operator") {
    return <Navigate to="/operator" replace />;
  }
  if (role === "homeowner") {
    return <Navigate to={`/homeowner/${nodeId || 1}`} replace />;
  }
  if (role === "community") {
    return <Navigate to="/community" replace />;
  }
  return <Navigate to="/login" replace />;
};

export default function App() {
  const navigate = useNavigate();
  const token = authStore.getToken();
  const role = authStore.getRole();

  return (
    <div>
      <header>
        <div className="navbar">
          <div style={{ fontWeight: 700 }}>VayuGrid Dashboards</div>
          {token && (
            <nav className="nav-links">
              <NavLink to="/operator">Operator</NavLink>
              <NavLink to="/homeowner">Homeowner</NavLink>
              <NavLink to="/community">Community</NavLink>
            </nav>
          )}
          {token && (
            <button
              className="button secondary"
              onClick={() => {
                authStore.clear();
                navigate("/login");
              }}
            >
              Logout ({role})
            </button>
          )}
        </div>
      </header>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="/" element={<RoleRedirect />} />
        <Route
          path="/operator"
          element={
            <RequireAuth>
              <OperatorDashboard />
            </RequireAuth>
          }
        />
        <Route
          path="/homeowner"
          element={
            <RequireAuth>
              <HomeownerDashboard />
            </RequireAuth>
          }
        />
        <Route
          path="/homeowner/:nodeId"
          element={
            <RequireAuth>
              <HomeownerDashboard />
            </RequireAuth>
          }
        />
        <Route
          path="/community"
          element={
            <RequireAuth>
              <CommunityDashboard />
            </RequireAuth>
          }
        />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </div>
  );
}
