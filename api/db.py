import os

from sqlalchemy import JSON, text, Column, DateTime, Float, ForeignKey, Index, Integer, String, create_engine, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import make_url
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import DeclarativeBase, sessionmaker

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql+psycopg://localhost/nids"
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
Session = sessionmaker(engine, expire_on_commit=False)
Json = JSON().with_variant(JSONB(), "postgresql")


class Base(DeclarativeBase):
    pass


class Batch(Base):
    __tablename__ = "batches"

    id = Column(Integer, primary_key=True)
    source = Column(String(16), nullable=False)
    condition = Column(String(32), nullable=False)
    filename = Column(String(255))
    n_flows = Column(Integer, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class Detection(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True)
    batch_id = Column(Integer, ForeignKey("batches.id", ondelete="CASCADE"), nullable=False)
    position = Column(Integer, nullable=False)
    truth = Column(String(16))
    rules = Column(String(16), nullable=False)
    logreg = Column(String(16), nullable=False)
    xgboost = Column(String(16), nullable=False)
    confidence = Column(Float, nullable=False)
    proba = Column(Json, nullable=False)
    reasons = Column(Json, nullable=False)
    flow = Column(Json, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        Index("ix_detections_batch", "batch_id"),
        Index("ix_detections_verdict", "xgboost"),
    )


def ensure_database():
    try:
        with engine.connect():
            return
    except OperationalError as err:
        if "does not exist" not in str(err):
            raise
    url = make_url(DATABASE_URL)
    admin = create_engine(url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(text(f'create database "{url.database}"'))
    admin.dispose()


def init_db():
    ensure_database()
    Base.metadata.create_all(engine)
