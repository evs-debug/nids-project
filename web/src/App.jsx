import { useEffect, useState } from "react";
import { api } from "./api.js";
import Dashboard from "./Dashboard.jsx";
import Simulator from "./Simulator.jsx";
import Upload from "./Upload.jsx";
import Alerts from "./Alerts.jsx";

const TABS = [
  { id: "dashboard", label: "Overview", view: Dashboard },
  { id: "simulator", label: "Live simulator", view: Simulator },
  { id: "upload", label: "Classify a file", view: Upload },
  { id: "alerts", label: "Alert history", view: Alerts },
];

export default function App() {
  const [tab, setTab] = useState("dashboard");
  const [backend, setBackend] = useState("checking");

  useEffect(() => {
    api.health().then(() => setBackend("ok")).catch(() => setBackend("down"));
  }, []);

  const View = TABS.find((t) => t.id === tab).view;

  return (
    <div className="shell">
      <header className="top">
        <div>
          <h1>NIDS Console</h1>
          <p className="muted">Network intrusion detection with explainable attack classification</p>
        </div>
        <span className={`status status-${backend}`}>
          {backend === "ok" ? "Backend connected" : backend === "down" ? "Backend unreachable" : "Connecting"}
        </span>
      </header>
      <nav className="tabs" role="tablist">
        {TABS.map((t) => (
          <button
            key={t.id}
            role="tab"
            aria-selected={tab === t.id}
            className={tab === t.id ? "tab active" : "tab"}
            onClick={() => setTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </nav>
      {backend === "down" ? (
        <div className="card error">
          The API is not responding. Start it with <code>uvicorn api.main:app --port 8000</code> and check that
          Postgres is running, then reload.
        </div>
      ) : (
        <main>
          <View />
        </main>
      )}
    </div>
  );
}
