import { useEffect, useRef, useState } from "react";
import { api } from "./api.js";
import { DetailPanel, DetectionTable, Scoreboard } from "./components.jsx";

const KEEP = 150;
const RATIOS = [
  { value: "", label: "Natural mix" },
  { value: "0.05", label: "5% attacks" },
  { value: "0.2", label: "20% attacks" },
  { value: "0.5", label: "50% attacks" },
];

export default function Simulator() {
  const [conditions, setConditions] = useState([]);
  const [condition, setCondition] = useState("clean");
  const [ratio, setRatio] = useState("");
  const [rows, setRows] = useState([]);
  const [running, setRunning] = useState(false);
  const [selected, setSelected] = useState(null);
  const [error, setError] = useState("");
  const busy = useRef(false);
  const params = useRef({});
  params.current = { condition, ratio };

  useEffect(() => {
    api.conditions().then(setConditions).catch((e) => setError(e.message));
  }, []);

  async function step() {
    if (busy.current) return;
    busy.current = true;
    try {
      const { condition, ratio } = params.current;
      const body = { condition, n: 8, attack_ratio: ratio ? Number(ratio) : null };
      const res = await api.simulate(body);
      setRows((prev) => [...res.detections.slice().reverse(), ...prev].slice(0, KEEP));
      setError("");
    } catch (e) {
      setError(e.message);
      setRunning(false);
    } finally {
      busy.current = false;
    }
  }

  useEffect(() => {
    if (!running) return undefined;
    step();
    const timer = setInterval(step, 1800);
    return () => clearInterval(timer);
  }, [running]);

  const alerts = rows.filter((r) => r.xgboost !== "normal").length;

  return (
    <div className="split">
      <section className="stack">
        <div className="card">
          <h2>Live traffic simulator</h2>
          <p className="muted">
            Draws flows the models never trained on, degrades them under the chosen network condition, and runs
            all three detectors. The percentages below count how often each detector matched the true label.
          </p>
          <div className="controls">
            <label>
              Network condition
              <select value={condition} onChange={(e) => setCondition(e.target.value)}>
                {conditions.map((c) => <option key={c.id} value={c.id}>{c.label}</option>)}
              </select>
            </label>
            <label>
              Traffic mix
              <select value={ratio} onChange={(e) => setRatio(e.target.value)}>
                {RATIOS.map((r) => <option key={r.value} value={r.value}>{r.label}</option>)}
              </select>
            </label>
            <button className={running ? "danger" : "primary"} onClick={() => setRunning((v) => !v)}>
              {running ? "Stop stream" : "Start stream"}
            </button>
            <button onClick={step} disabled={running}>Run one batch</button>
            <button className="ghost" onClick={() => { setRows([]); setSelected(null); }}>Clear view</button>
          </div>
          {error && <p className="error-text">{error}</p>}
        </div>

        <div className="card">
          <Scoreboard rows={rows} />
          <p className="muted small">
            {rows.length} flows shown, {alerts} raised as alerts by XGBoost. Rows shaded red are flows a detector got wrong.
          </p>
          <DetectionTable
            rows={rows}
            selectedId={selected}
            onSelect={setSelected}
            empty="Nothing yet. Press Start stream, or Run one batch."
          />
        </div>
      </section>
      {selected && <DetailPanel id={selected} onClose={() => setSelected(null)} />}
    </div>
  );
}
