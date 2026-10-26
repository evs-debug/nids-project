import io
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy import case, func, select, text
from sqlalchemy.orm import Session as OrmSession

from . import db
from .engine import CLASSES, CONDITIONS, MODEL_NAMES, Engine, read_uploaded

MAX_UPLOAD_ROWS = 1000
state = {}


@asynccontextmanager
async def lifespan(app):
    db.init_db()
    state["engine"] = Engine()
    yield


app = FastAPI(title="NIDS", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_session():
    with db.Session() as session:
        yield session


class SimulateRequest(BaseModel):
    condition: str = "clean"
    n: int = Field(10, ge=1, le=200)
    attack_ratio: Optional[float] = Field(None, ge=0, le=1)


def store(session, batch, rows, flows, truth):
    flow_json = flows.to_dict(orient="records")
    detections = [
        db.Detection(
            batch_id=batch.id,
            position=i,
            truth=truth[i] if truth else None,
            rules=r["rules"],
            logreg=r["logreg"],
            xgboost=r["xgboost"],
            confidence=r["confidence"],
            proba=r["proba"],
            reasons=r["reasons"],
            flow={k: (v if isinstance(v, str) else float(v)) for k, v in flow_json[i].items()},
        )
        for i, r in enumerate(rows)
    ]
    session.add_all(detections)
    session.commit()
    return detections


def summarize(d, detail=False):
    out = {
        "id": d.id,
        "batch_id": d.batch_id,
        "position": d.position,
        "truth": d.truth,
        "rules": d.rules,
        "logreg": d.logreg,
        "xgboost": d.xgboost,
        "confidence": d.confidence,
        "reasons": d.reasons,
        "created_at": d.created_at.isoformat() if d.created_at else None,
    }
    if detail:
        out["proba"] = d.proba
        out["flow"] = d.flow
    return out


@app.get("/api/health")
def health(session: OrmSession = Depends(get_session)):
    session.execute(text("select 1"))
    return {"status": "ok", "classes": CLASSES}


@app.get("/api/conditions")
def conditions():
    return [{"id": k, "label": v[0]} for k, v in CONDITIONS.items()]


@app.post("/api/simulate")
def simulate(req: SimulateRequest, session: OrmSession = Depends(get_session)):
    if req.condition not in CONDITIONS:
        raise HTTPException(422, f"unknown condition '{req.condition}'")
    engine = state["engine"]
    flows, truth = engine.draw(req.n, req.condition, req.attack_ratio, time.time_ns() % 2**32)
    rows = engine.classify(flows)
    batch = db.Batch(source="simulator", condition=req.condition, n_flows=len(rows))
    session.add(batch)
    session.flush()
    saved = store(session, batch, rows, flows, truth)
    return {"batch_id": batch.id, "detections": [summarize(d) for d in saved]}


@app.post("/api/classify")
def classify_upload(file: UploadFile = File(...), session: OrmSession = Depends(get_session)):
    try:
        flows, truth = read_uploaded(io.BytesIO(file.file.read()))
    except ValueError as err:
        raise HTTPException(422, str(err))
    if len(flows) == 0:
        raise HTTPException(422, "the file has no rows")
    if len(flows) > MAX_UPLOAD_ROWS:
        raise HTTPException(413, f"upload at most {MAX_UPLOAD_ROWS} flows at a time")
    rows = state["engine"].classify(flows)
    batch = db.Batch(source="upload", condition="clean", filename=file.filename, n_flows=len(rows))
    session.add(batch)
    session.flush()
    saved = store(session, batch, rows, flows, truth)
    return {"batch_id": batch.id, "detections": [summarize(d) for d in saved]}


@app.get("/api/detections")
def list_detections(
    verdict: Optional[str] = None,
    alerts_only: bool = False,
    batch_id: Optional[int] = None,
    limit: int = Query(50, le=500),
    offset: int = 0,
    session: OrmSession = Depends(get_session),
):
    query = select(db.Detection).order_by(db.Detection.id.desc())
    if verdict:
        query = query.where(db.Detection.xgboost == verdict)
    if alerts_only:
        query = query.where(db.Detection.xgboost != "normal")
    if batch_id:
        query = query.where(db.Detection.batch_id == batch_id)
    items = session.scalars(query.limit(limit).offset(offset)).all()
    return [summarize(d) for d in items]


@app.get("/api/detections/{detection_id}")
def get_detection(detection_id: int, session: OrmSession = Depends(get_session)):
    d = session.get(db.Detection, detection_id)
    if d is None:
        raise HTTPException(404, "no such detection")
    return summarize(d, detail=True)


@app.get("/api/stats")
def stats(session: OrmSession = Depends(get_session)):
    D = db.Detection
    total = session.scalar(select(func.count(D.id))) or 0
    by_class = dict(session.execute(select(D.xgboost, func.count(D.id)).group_by(D.xgboost)).all())
    labeled = D.truth.is_not(None)
    accuracy = {}
    for name in MODEL_NAMES:
        col = getattr(D, name)
        n, hit = session.execute(
            select(func.count(D.id), func.sum(case((col == D.truth, 1), else_=0))).where(labeled)
        ).one()
        accuracy[name] = {"labeled": n or 0, "correct": int(hit or 0)}
    recent = session.execute(
        select(db.Batch.id, db.Batch.source, db.Batch.condition, db.Batch.n_flows, db.Batch.created_at)
        .order_by(db.Batch.id.desc()).limit(8)
    ).all()
    return {
        "total": total,
        "alerts": total - by_class.get("normal", 0),
        "by_class": {c: by_class.get(c, 0) for c in CLASSES},
        "accuracy": accuracy,
        "batches": [
            {"id": b.id, "source": b.source, "condition": b.condition,
             "n_flows": b.n_flows, "created_at": b.created_at.isoformat()}
            for b in recent
        ],
    }


@app.post("/api/reset")
def reset(session: OrmSession = Depends(get_session)):
    session.execute(text("truncate detections, batches restart identity"))
    session.commit()
    return {"status": "cleared"}


@app.get("/api/benchmark/{condition}")
def benchmark(condition: str):
    if condition not in CONDITIONS:
        raise HTTPException(404, "unknown condition")
    return {"condition": condition, "label": CONDITIONS[condition][0],
            "scores": state["engine"].benchmark(condition)}


@app.get("/api/sample.csv")
def sample_csv(n: int = Query(40, ge=5, le=500)):
    frame = state["engine"].sample_frame(n, seed=n)
    return Response(
        frame.to_csv(index=False),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=sample_flows.csv"},
    )
