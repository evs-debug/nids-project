import { useEffect, useState } from "react";
import { api, CLASSES, MODELS } from "./api.js";

export function Verdict({ name }) {
  return (
    <span className="verdict">
      <i className={`dot c-${name}`} />
      {name}
    </span>
  );
}

function topReasons(reasons) {
  return reasons
    .slice(0, 2)
    .map((r) => `${r.feature}=${typeof r.value === "number" ? Number(r.value.toPrecision(3)) : r.value}`)
    .join(", ");
}

export function DetectionTable({ rows, selectedId, onSelect, empty }) {
  if (!rows.length) return <p className="muted pad">{empty}</p>;
  const labeled = rows.some((r) => r.truth);
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>#</th>
            {labeled && <th>Actual</th>}
            <th>Rules</th>
            <th>Logistic reg.</th>
            <th>XGBoost</th>
            <th>Conf.</th>
            <th>Main reasons</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr
              key={r.id}
              className={r.id === selectedId ? "selected" : ""}
              onClick={() => onSelect(r.id)}
              tabIndex={0}
              onKeyDown={(e) => e.key === "Enter" && onSelect(r.id)}
            >
              <td className="num">{r.id}</td>
              {labeled && <td>{r.truth ? <Verdict name={r.truth} /> : "-"}</td>}
              {["rules", "logreg", "xgboost"].map((m) => (
                <td key={m} className={r.truth && r[m] !== r.truth ? "miss" : ""}>
                  <Verdict name={r[m]} />
                </td>
              ))}
              <td className="num">{(r.confidence * 100).toFixed(0)}%</td>
              <td className="reasons">{topReasons(r.reasons)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function DetailPanel({ id, onClose }) {
  const [d, setD] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    setD(null);
    setError("");
    api.detection(id).then(setD).catch((e) => setError(e.message));
  }, [id]);

  if (error) return <aside className="card detail"><p className="error-text">{error}</p></aside>;
  if (!d) return <aside className="card detail"><p className="muted">Loading</p></aside>;

  const scale = Math.max(...d.reasons.map((r) => Math.abs(r.shap)), 0.01);
  return (
    <aside className="card detail">
      <div className="detail-head">
        <h3>Flow {d.id}</h3>
        <button className="ghost" onClick={onClose} aria-label="Close details">Close</button>
      </div>
      <div className="kv">
        {d.truth && <><span>Actual</span><Verdict name={d.truth} /></>}
        <span>XGBoost says</span>
        <Verdict name={d.xgboost} />
        <span>Rules say</span>
        <Verdict name={d.rules} />
        <span>Logistic reg. says</span>
        <Verdict name={d.logreg} />
      </div>

      <h4>Why XGBoost said {d.xgboost}</h4>
      <p className="muted small">
        Each bar is how far a feature pushed the verdict toward (right) or away from (left) {d.xgboost}.
      </p>
      <ul className="shap">
        {d.reasons.map((r) => (
          <li key={r.feature}>
            <span className="shap-name">
              {r.feature} <b>{typeof r.value === "number" ? Number(r.value.toPrecision(3)) : r.value}</b>
            </span>
            <span className="shap-track">
              <span
                className={r.shap >= 0 ? "shap-bar pos" : "shap-bar neg"}
                style={{ width: `${(Math.abs(r.shap) / scale) * 50}%` }}
              />
            </span>
            <span className="shap-val num">{r.shap > 0 ? "+" : ""}{r.shap.toFixed(2)}</span>
          </li>
        ))}
      </ul>

      <h4>Class probabilities</h4>
      <ul className="proba">
        {CLASSES.map((c) => (
          <li key={c}>
            <span><Verdict name={c} /></span>
            <span className="proba-track"><span className="proba-bar" style={{ width: `${d.proba[c] * 100}%` }} /></span>
            <span className="num">{(d.proba[c] * 100).toFixed(1)}%</span>
          </li>
        ))}
      </ul>

      <details>
        <summary>All 41 flow features</summary>
        <div className="kv flow">
          {Object.entries(d.flow).map(([k, v]) => (
            <span key={k} className="flow-item"><span>{k}</span><b>{typeof v === "number" ? Number(v.toPrecision(5)) : v}</b></span>
          ))}
        </div>
      </details>
    </aside>
  );
}

export function Scoreboard({ rows }) {
  const labeled = rows.filter((r) => r.truth);
  if (!labeled.length) return null;
  return (
    <div className="score-row">
      {MODELS.map((m) => {
        const hit = labeled.filter((r) => r[m.id] === r.truth).length;
        return (
          <div key={m.id} className={m.id === "xgboost" ? "score main" : "score"}>
            <span className="muted small">{m.label}</span>
            <b>{((hit / labeled.length) * 100).toFixed(1)}%</b>
            <span className="muted small">{hit} of {labeled.length} correct</span>
          </div>
        );
      })}
    </div>
  );
}
