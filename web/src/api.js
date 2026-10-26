async function request(path, options) {
  const res = await fetch(`/api${path}`, options);
  if (!res.ok) {
    let message = res.statusText;
    try {
      const body = await res.json();
      message = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* body was not JSON */
    }
    throw new Error(message);
  }
  return res.json();
}

const json = (body) => ({
  method: "POST",
  headers: { "content-type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  health: () => request("/health"),
  conditions: () => request("/conditions"),
  simulate: (body) => request("/simulate", json(body)),
  classify: (file) => {
    const form = new FormData();
    form.append("file", file);
    return request("/classify", { method: "POST", body: form });
  },
  detections: (params) => request(`/detections?${new URLSearchParams(params)}`),
  detection: (id) => request(`/detections/${id}`),
  stats: () => request("/stats"),
  reset: () => request("/reset", { method: "POST" }),
  benchmark: (condition) => request(`/benchmark/${condition}`),
};

export const CLASSES = ["normal", "DoS", "Probe", "R2L", "U2R"];
export const MODELS = [
  { id: "rules", label: "Rules" },
  { id: "logreg", label: "Logistic regression" },
  { id: "xgboost", label: "XGBoost" },
];
