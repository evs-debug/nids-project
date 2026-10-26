import { useState } from "react";
import { api } from "./api.js";
import { DetailPanel, DetectionTable, Scoreboard } from "./components.jsx";

export default function Upload() {
  const [rows, setRows] = useState([]);
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [selected, setSelected] = useState(null);

  async function handle(file) {
    if (!file) return;
    setBusy(true);
    setError("");
    setSelected(null);
    try {
      const res = await api.classify(file);
      setRows(res.detections);
      setName(file.name);
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  const counts = rows.reduce((acc, r) => ({ ...acc, [r.xgboost]: (acc[r.xgboost] || 0) + 1 }), {});

  return (
    <div className="split">
      <section className="stack">
        <div className="card">
          <h2>Classify your own flows</h2>
          <p className="muted">
            Upload a CSV with the 41 NSL-KDD flow features (a header row is expected, with an optional
            <code> label </code> column). Raw NSL-KDD files without a header also work. Up to 1000 flows at a time.
          </p>
          <div className="controls">
            <label className="file">
              <input type="file" accept=".csv,.txt" disabled={busy} onChange={(e) => handle(e.target.files[0])} />
            </label>
            <a className="button-link" href="/api/sample.csv?n=40" download>Download a sample file</a>
          </div>
          {busy && <p className="muted">Classifying</p>}
          {error && <p className="error-text">{error}</p>}
        </div>

        {rows.length > 0 && (
          <div className="card">
            <h3>{name}</h3>
            <p className="muted small">
              {rows.length} flows. XGBoost verdicts:{" "}
              {Object.entries(counts).map(([k, v]) => `${k} ${v}`).join(", ")}.
            </p>
            <Scoreboard rows={rows} />
            <DetectionTable rows={rows} selectedId={selected} onSelect={setSelected} empty="" />
          </div>
        )}
      </section>
      {selected && <DetailPanel id={selected} onClose={() => setSelected(null)} />}
    </div>
  );
}
