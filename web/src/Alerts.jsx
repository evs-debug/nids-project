import { useCallback, useEffect, useState } from "react";
import { api, CLASSES } from "./api.js";
import { DetailPanel, DetectionTable } from "./components.jsx";

const PAGE = 25;

export default function Alerts() {
  const [rows, setRows] = useState([]);
  const [filter, setFilter] = useState("alerts");
  const [page, setPage] = useState(0);
  const [selected, setSelected] = useState(null);
  const [error, setError] = useState("");

  const load = useCallback(() => {
    const params = { limit: PAGE, offset: page * PAGE };
    if (filter === "alerts") params.alerts_only = "true";
    else if (filter !== "all") params.verdict = filter;
    api.detections(params).then(setRows).catch((e) => setError(e.message));
  }, [filter, page]);

  useEffect(load, [load]);

  async function clearHistory() {
    if (!window.confirm("Delete all stored detections?")) return;
    await api.reset();
    setSelected(null);
    setPage(0);
    load();
  }

  return (
    <div className="split">
      <section className="stack">
        <div className="card">
          <h2>Alert history</h2>
          <p className="muted">Every verdict is stored in Postgres, so history survives a restart.</p>
          <div className="controls">
            <label>
              Show
              <select value={filter} onChange={(e) => { setFilter(e.target.value); setPage(0); }}>
                <option value="alerts">All alerts</option>
                <option value="all">All flows</option>
                {CLASSES.map((c) => <option key={c} value={c}>Only {c}</option>)}
              </select>
            </label>
            <button onClick={load}>Refresh</button>
            <button className="ghost" onClick={clearHistory}>Clear history</button>
          </div>
          {error && <p className="error-text">{error}</p>}
        </div>
        <div className="card">
          <DetectionTable rows={rows} selectedId={selected} onSelect={setSelected} empty="No stored detections match." />
          <div className="pager">
            <button disabled={page === 0} onClick={() => setPage(page - 1)}>Newer</button>
            <span className="muted small">Page {page + 1}</span>
            <button disabled={rows.length < PAGE} onClick={() => setPage(page + 1)}>Older</button>
          </div>
        </div>
      </section>
      {selected && <DetailPanel id={selected} onClose={() => setSelected(null)} />}
    </div>
  );
}
