"""
alert_manager.py
-----------------
AlertManager: turns FuzzyAggregator output into persisted, timestamped
alerts (FR4), maintains the immutable audit trail (FR9), and supports
CSV/PDF export (FR8).
"""

from __future__ import annotations

import csv
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

from config import DB_PATH

logger = logging.getLogger("ai_nids.alert_manager")


@dataclass
class Alert:
    id: Optional[int]
    timestamp: str
    attack_category: str
    confidence: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    status: str  # "alerted" | "review_queue" | "acknowledged"


class AlertManager:
    """Owns the alerts and audit_log tables. See db/schema.sql for DDL."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _ensure_schema(self) -> None:
        import os
        schema_path = os.path.join(os.path.dirname(self.db_path), "schema.sql")
        conn = self._connect()
        try:
            if os.path.exists(schema_path):
                with open(schema_path) as f:
                    conn.executescript(f.read())
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    def raise_alert(
        self, attack_category: str, confidence: float,
        src_ip: str, dst_ip: str, src_port: int = 0, dst_port: int = 0,
        needs_review: bool = False,
    ) -> int:
        ts = datetime.now(timezone.utc).isoformat()
        status = "review_queue" if needs_review else "alerted"

        conn = self._connect()
        try:
            cur = conn.execute(
                "INSERT INTO alerts (timestamp, attack_category, confidence, "
                "src_ip, dst_ip, src_port, dst_port, status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (ts, attack_category, confidence, src_ip, dst_ip, src_port,
                 dst_port, status),
            )
            alert_id = cur.lastrowid
            conn.commit()
        finally:
            conn.close()

        self._audit("ALERT_RAISED", f"alert_id={alert_id} category={attack_category} "
                                     f"confidence={confidence:.3f} status={status}")
        logger.info("Alert #%d raised: %s (confidence=%.3f, status=%s)",
                    alert_id, attack_category, confidence, status)
        return alert_id

    def acknowledge(self, alert_id: int, actor: str) -> None:
        conn = self._connect()
        try:
            conn.execute("UPDATE alerts SET status = 'acknowledged' WHERE id = ?",
                         (alert_id,))
            conn.commit()
        finally:
            conn.close()
        self._audit("ALERT_ACKNOWLEDGED", f"alert_id={alert_id} by={actor}")

    def list_alerts(self, status: Optional[str] = None, limit: int = 500) -> List[Alert]:
        conn = self._connect()
        try:
            if status:
                rows = conn.execute(
                    "SELECT * FROM alerts WHERE status = ? ORDER BY id DESC LIMIT ?",
                    (status, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,)
                ).fetchall()
        finally:
            conn.close()
        return [Alert(**dict(r)) for r in rows]

    # ------------------------------------------------------------------
    # Audit trail (FR9) — immutable: only INSERTs, application layer never
    # issues UPDATE/DELETE against audit_log.
    # ------------------------------------------------------------------
    def _audit(self, event_type: str, details: str, actor: str = "system") -> None:
        ts = datetime.now(timezone.utc).isoformat()
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO audit_log (timestamp, event_type, actor, details) "
                "VALUES (?, ?, ?, ?)",
                (ts, event_type, actor, details),
            )
            conn.commit()
        finally:
            conn.close()

    def record_audit_event(self, event_type: str, details: str, actor: str = "system") -> None:
        """Public entry point for callers outside AlertManager (e.g. the
        Flask dashboard logging threshold changes and retraining events)."""
        self._audit(event_type, details, actor)

    def list_audit_log(self, limit: int = 500) -> List[dict]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        finally:
            conn.close()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Export (FR8)
    # ------------------------------------------------------------------
    def export_alerts_csv(self, out_path: str, status: Optional[str] = None) -> str:
        alerts = self.list_alerts(status=status, limit=100000)
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "timestamp", "attack_category", "confidence",
                              "src_ip", "dst_ip", "src_port", "dst_port", "status"])
            for a in alerts:
                writer.writerow([a.id, a.timestamp, a.attack_category, a.confidence,
                                  a.src_ip, a.dst_ip, a.src_port, a.dst_port, a.status])
        self._audit("EXPORT_CSV", f"path={out_path} status_filter={status}")
        return out_path

    def export_alerts_pdf(self, out_path: str, status: Optional[str] = None) -> str:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import landscape, letter
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle

        alerts = self.list_alerts(status=status, limit=1000)
        data = [["ID", "Timestamp", "Category", "Confidence", "Src IP",
                 "Dst IP", "Src Port", "Dst Port", "Status"]]
        for a in alerts:
            data.append([a.id, a.timestamp, a.attack_category, f"{a.confidence:.3f}",
                         a.src_ip, a.dst_ip, a.src_port, a.dst_port, a.status])

        doc = SimpleDocTemplate(out_path, pagesize=landscape(letter))
        table = Table(data, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f0f5")]),
        ]))
        doc.build([table])
        self._audit("EXPORT_PDF", f"path={out_path} status_filter={status}")
        return out_path
