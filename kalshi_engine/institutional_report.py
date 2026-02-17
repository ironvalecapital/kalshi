from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict
import csv
import json


def _write_markdown(path: Path, payload: Dict[str, Any]) -> None:
    lines = ["# Institutional Capital Report", "", f"Generated: {datetime.utcnow().isoformat()}Z", ""]
    for k in [
        "strategy_overview",
        "sharpe_history",
        "ev_ratio",
        "drawdown",
        "risk_of_ruin",
        "capacity_limits",
    ]:
        v = payload.get(k)
        lines.append(f"## {k.replace('_', ' ').title()}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(v, indent=2, default=str))
        lines.append("```")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_csv(path: Path, payload: Dict[str, Any]) -> None:
    rows = []
    for k, v in payload.items():
        rows.append({"key": k, "value": json.dumps(v, default=str)})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["key", "value"])
        w.writeheader()
        w.writerows(rows)


def _try_write_pdf(path: Path, payload: Dict[str, Any]) -> bool:
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except Exception:
        return False
    c = canvas.Canvas(str(path), pagesize=letter)
    w, h = letter
    y = h - 40
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, "Institutional Capital Report")
    y -= 24
    c.setFont("Helvetica", 9)
    c.drawString(40, y, f"Generated: {datetime.utcnow().isoformat()}Z")
    y -= 20
    for k, v in payload.items():
        c.setFont("Helvetica-Bold", 10)
        c.drawString(40, y, str(k))
        y -= 14
        c.setFont("Helvetica", 8)
        text = json.dumps(v, default=str)
        for i in range(0, len(text), 110):
            c.drawString(50, y, text[i : i + 110])
            y -= 11
            if y < 60:
                c.showPage()
                y = h - 40
    c.save()
    return True


def export_institutional_report(payload: Dict[str, Any], out_dir: str = "runs/investor_report") -> Dict[str, str]:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    stem = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    md = p / f"investor_report_{stem}.md"
    csvp = p / f"investor_report_{stem}.csv"
    pdf = p / f"investor_report_{stem}.pdf"
    _write_markdown(md, payload)
    _write_csv(csvp, payload)
    pdf_ok = _try_write_pdf(pdf, payload)
    out = {
        "markdown": str(md),
        "csv": str(csvp),
    }
    if pdf_ok:
        out["pdf"] = str(pdf)
    return out
