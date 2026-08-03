"""
web/app.py
----------
DashboardController: Flask-based web dashboard (FR6) with role-based
access control (FR7): 'admin' has full access (threshold changes,
retraining, user management); 'analyst' is read-only (view alerts,
metrics, export).

Run with:
    python web/app.py
Then visit http://127.0.0.1:5000  (default admin/admin — CHANGE THIS,
see README "First run" section).
"""

from __future__ import annotations

import functools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import (Flask, flash, jsonify, redirect, render_template,  # noqa: E402
                    request, send_file, session, url_for)
from werkzeug.security import check_password_hash, generate_password_hash  # noqa: E402

from config import DB_PATH, SECRET_KEY  # noqa: E402
from src.alert_manager import AlertManager  # noqa: E402
from pipeline import NIDSPipeline  # noqa: E402

app = Flask(__name__)
app.secret_key = SECRET_KEY

alert_manager = AlertManager(db_path=DB_PATH)
_pipeline_cache: dict = {}


def get_pipeline() -> NIDSPipeline:
    """Lazily loads the trained pipeline once per process."""
    if "pipeline" not in _pipeline_cache:
        p = NIDSPipeline(alert_manager=alert_manager)
        try:
            p.load()
        except FileNotFoundError:
            pass  # allowed: dashboard can render alert history before a model is trained
        _pipeline_cache["pipeline"] = p
    return _pipeline_cache["pipeline"]


# --------------------------------------------------------------------
# RBAC helpers (FR7)
# --------------------------------------------------------------------
def login_required(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if "username" not in session:
            return redirect(url_for("login", next=request.path))
        return fn(*args, **kwargs)
    return wrapper


def admin_required(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if session.get("role") != "admin":
            flash("This action requires administrator privileges.", "error")
            return redirect(url_for("dashboard"))
        return fn(*args, **kwargs)
    return wrapper


def _get_user(username: str):
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    return dict(row) if row else None


def bootstrap_default_admin():
    """Creates a default admin/admin account on first run if the users
    table is empty, so the dashboard is usable immediately after
    `python web/app.py` without a separate seeding step."""
    import sqlite3
    alert_manager._ensure_schema()
    conn = sqlite3.connect(DB_PATH)
    count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    if count == 0:
        conn.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
            ("admin", generate_password_hash("admin"), "admin"),
        )
        conn.commit()
        print("[!] Created default account admin/admin — change this password immediately.")
    conn.close()


# --------------------------------------------------------------------
# Auth routes
# --------------------------------------------------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        user = _get_user(username)
        if user and check_password_hash(user["password_hash"], password):
            session["username"] = user["username"]
            session["role"] = user["role"]
            alert_manager.record_audit_event("LOGIN", f"user={username}", actor=username)
            return redirect(request.args.get("next") or url_for("dashboard"))
        flash("Invalid username or password.", "error")
    return render_template("login.html")


@app.route("/logout")
def logout():
    username = session.get("username", "unknown")
    session.clear()
    alert_manager.record_audit_event("LOGOUT", f"user={username}", actor=username)
    return redirect(url_for("login"))


# --------------------------------------------------------------------
# Dashboard views (read access for both roles)
# --------------------------------------------------------------------
@app.route("/")
@login_required
def dashboard():
    alerts = alert_manager.list_alerts(limit=50)
    review_queue = alert_manager.list_alerts(status="review_queue", limit=50)
    return render_template(
        "dashboard.html",
        alerts=alerts, review_queue=review_queue,
        role=session.get("role"), username=session.get("username"),
    )


@app.route("/alerts")
@login_required
def alerts_view():
    status = request.args.get("status")
    alerts = alert_manager.list_alerts(status=status, limit=500)
    return render_template("alerts.html", alerts=alerts, status=status,
                            role=session.get("role"))


@app.route("/api/alerts")
@login_required
def api_alerts():
    status = request.args.get("status")
    alerts = alert_manager.list_alerts(status=status, limit=500)
    return jsonify([a.__dict__ for a in alerts])


@app.route("/metrics")
@login_required
def metrics_view():
    import json
    report_path = os.path.join(os.path.dirname(__file__), "..", "logs", "evaluation_report.json")
    report = None
    if os.path.exists(report_path):
        with open(report_path) as f:
            report = json.load(f)
    return render_template("metrics.html", report=report, role=session.get("role"))


# --------------------------------------------------------------------
# Analyst-permitted action: acknowledge (does NOT require admin)
# --------------------------------------------------------------------
@app.route("/alerts/<int:alert_id>/acknowledge", methods=["POST"])
@login_required
def acknowledge_alert(alert_id: int):
    alert_manager.acknowledge(alert_id, actor=session.get("username", "unknown"))
    flash(f"Alert #{alert_id} acknowledged.", "success")
    return redirect(url_for("dashboard"))


# --------------------------------------------------------------------
# Admin-only actions (FR7)
# --------------------------------------------------------------------
@app.route("/admin/retrain", methods=["POST"])
@login_required
@admin_required
def retrain():
    csv_path = request.form.get("csv_path")
    if not csv_path or not os.path.exists(csv_path):
        flash("Valid --data CSV path required for retraining.", "error")
        return redirect(url_for("dashboard"))
    pipeline = get_pipeline()
    result = pipeline.train(csv_path)
    result.pop("test_data", None)
    alert_manager.record_audit_event(
        "MODEL_RETRAINED_VIA_DASHBOARD", str(result), actor=session.get("username"),
    )
    flash("Retraining complete.", "success")
    return redirect(url_for("dashboard"))


@app.route("/admin/threshold", methods=["POST"])
@login_required
@admin_required
def update_threshold():
    try:
        new_delta = float(request.form.get("delta", ""))
    except (TypeError, ValueError):
        flash("Threshold must be a number between 0 and 1.", "error")
        return redirect(url_for("dashboard"))

    if not (0.0 <= new_delta <= 1.0):
        flash("Threshold must be between 0 and 1.", "error")
        return redirect(url_for("dashboard"))

    pipeline = get_pipeline()
    old_delta = pipeline.fuzzy.delta
    pipeline.fuzzy.delta = new_delta
    alert_manager.record_audit_event(
        "THRESHOLD_CHANGED", f"old={old_delta} new={new_delta}",
        actor=session.get("username"),
    )
    flash(f"Ambiguity threshold updated: {old_delta} -> {new_delta}", "success")
    return redirect(url_for("dashboard"))


@app.route("/admin/users", methods=["POST"])
@login_required
@admin_required
def create_user():
    import sqlite3
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")
    role = request.form.get("role", "analyst")
    if role not in ("admin", "analyst") or not username or not password:
        flash("Invalid user details.", "error")
        return redirect(url_for("dashboard"))

    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
            (username, generate_password_hash(password), role),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        flash("Username already exists.", "error")
        return redirect(url_for("dashboard"))
    finally:
        conn.close()

    alert_manager.record_audit_event("USER_CREATED", f"new_user={username} role={role}",
                                      actor=session.get("username"))
    flash(f"User {username} ({role}) created.", "success")
    return redirect(url_for("dashboard"))


@app.route("/admin/audit-log")
@login_required
@admin_required
def audit_log_view():
    events = alert_manager.list_audit_log(limit=500)
    return render_template("audit_log.html", events=events)


# --------------------------------------------------------------------
# Export (FR8) — available to both roles (read-only action)
# --------------------------------------------------------------------
@app.route("/export/<fmt>")
@login_required
def export_alerts(fmt: str):
    status = request.args.get("status")
    out_dir = os.path.join(os.path.dirname(__file__), "..", "logs", "exports")
    os.makedirs(out_dir, exist_ok=True)

    if fmt == "csv":
        path = alert_manager.export_alerts_csv(os.path.join(out_dir, "alerts_export.csv"), status)
    elif fmt == "pdf":
        path = alert_manager.export_alerts_pdf(os.path.join(out_dir, "alerts_export.pdf"), status)
    else:
        flash("Unsupported export format.", "error")
        return redirect(url_for("dashboard"))

    return send_file(path, as_attachment=True)


if __name__ == "__main__":
    bootstrap_default_admin()
    app.run(debug=True, host="127.0.0.1", port=5000)
