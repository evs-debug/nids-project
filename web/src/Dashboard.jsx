import { useEffect, useState } from "react";
import { api, CLASSES, MODELS } from "./api.js";
import { Verdict } from "./components.jsx";

const METRICS = [
  { id: "macro_f1", label: "Macro-F1" },
  { id: "detection_rate", label: "Detection rate" },
  { id: "alert_precision", label: "Alert precision" },
];

function BenchmarkPlot({ results }) {
  const [metric, setMetric] = useState("macro_f1");
  if (!results.length) return <p className="muted pad">Computing benchmarks</p>;

  const all = results.flatMap((r) => MODELS.map((m) => r.scores[m.id][metric]));
  const lo = Math.max(0, Math.floor((Math.min(...all) - 0.02) * 10) / 10);
  const pos = (v) => `${((v - lo) / (1 - lo)) * 100}%`;
  const ticks = [];
  for (let t = lo; t <= 1.0001; t += 0.1) ticks.push(Number(t.toFixed(1)));

  return (
    <div>
      <div className="controls tight">
        {METRICS.map((m) => (
          <button key={m.id} className={metric === m.id ? "chip active" : "chip"} onClick={() => setMetric(m.id)}>
            {m.label}
          </button>
        ))}
        <span className="legend">
          {MODELS.map((m) => (
            <span key={m.id}><i className={`dot m-${m.id}`} />{m.label}</span>
          ))}
        </span>
      </div>
      <div className="plot">
        <div className="plot-row head">
          <span />
          <span />
          {MODELS.map((m) => <span key={m.id} className="plot-val muted small">{m.label.split(" ")[0]}</span>)}
        </div>
        {results.map((r) => {
          const vals = MODELS.map((m) => r.scores[m.id][metric]);
          return (
            <div key={r.condition} className="plot-row">
              <span className="plot-label">{r.label}</span>
              <span className="plot-track">
                <span className="plot-line" style={{ left: pos(Math.min(...vals)), right: `calc(100% - ${pos(Math.max(...vals))})` }} />
                {MODELS.map((m) => (
                  <i key={m.id} className={`dot big m-${m.id}`} style={{ left: pos(r.scores[m.id][metric]) }} title={`${m.label} ${r.scores[m.id][metric]}`} />
                ))}
              </span>
              {MODELS.map((m) => (
                <span key={m.id} className={m.id === "xgboost" ? "plot-val num strong" : "plot-val num"}>
                  {r.scores[m.id][metric].toFixed(3)}
                </span>
              ))}
            </div>
          );
        })}
        <div className="plot-row axis">
          <span />
          <span className="plot-track ticks">
            {ticks.map((t) => <span key={t} style={{ left: pos(t) }}>{t.toFixed(1)}</span>)}
          </span>
        </div>
      </div>
      <p className="muted small">
        Scores on 25,195 held-out flows the models never trained on, one run with a fixed seed. Each condition
        degrades those flows the same way for all three detectors. The case study averages three seeds.
      </p>
    </div>
  );
}

export default function Dashboard() {
  const [stats, setStats] = useState(null);
  const [results, setResults] = useState([]);
  const [error, setError] = useState("");

  useEffect(() => {
    api.stats().then(setStats).catch((e) => setError(e.message));
    api.conditions()
      .then((cs) => Promise.all(cs.map((c) => api.benchmark(c.id))))
      .then(setResults)
      .catch((e) => setError(e.message));
  }, []);

  const peak = stats ? Math.max(1, ...CLASSES.map((c) => stats.by_class[c])) : 1;

  return (
    <div className="stack">
      <div className="card">
        <h2>How the detectors compare</h2>
        <p className="muted">
          XGBoost holds up as traffic degrades; the rule-based baseline falls apart. On the official NSL-KDD
          test set, which contains attack types never seen in training, the rules do better. See the case study.
        </p>
        {error && <p className="error-text">{error}</p>}
        <BenchmarkPlot results={results} />
      </div>

      <div className="grid-2">
        <div className="card">
          <h3>Stored detections</h3>
          {!stats ? <p className="muted">Loading</p> : (
            <>
              <div className="score-row">
                <div className="score"><span className="muted small">Flows classified</span><b>{stats.total}</b></div>
                <div className="score"><span className="muted small">Alerts raised</span><b>{stats.alerts}</b></div>
              </div>
              <ul className="bars">
                {CLASSES.map((c) => (
                  <li key={c}>
                    <span><Verdict name={c} /></span>
                    <span className="bar-track"><span className="bar" style={{ width: `${(stats.by_class[c] / peak) * 100}%` }} /></span>
                    <span className="num">{stats.by_class[c]}</span>
                  </li>
                ))}
              </ul>
              <p className="muted small">XGBoost verdicts across everything in the database.</p>
            </>
          )}
        </div>
        <div className="card">
          <h3>Accuracy on labeled flows so far</h3>
          {!stats ? <p className="muted">Loading</p> : stats.accuracy.xgboost.labeled === 0 ? (
            <p className="muted">Run the simulator or upload a labeled file to fill this in.</p>
          ) : (
            <ul className="bars">
              {MODELS.map((m) => {
                const a = stats.accuracy[m.id];
                const pct = (a.correct / a.labeled) * 100;
                return (
                  <li key={m.id}>
                    <span>{m.label}</span>
                    <span className="bar-track"><span className="bar" style={{ width: `${pct}%` }} /></span>
                    <span className="num">{pct.toFixed(1)}%</span>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
}
