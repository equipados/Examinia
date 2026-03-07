from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from app.templates import templates
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.database import get_db
from app.db_models import ExamSession, Submission, TokenUsage, User

router = APIRouter(prefix="/reports")

# Precios por millón de tokens (USD) — misma tabla que sessions.py
_TOKEN_PRICES: dict[str, dict[str, float]] = {
    "gemini-2.5-pro":   {"input": 1.25,  "output": 10.0},
    "gemini-2.5-flash": {"input": 0.15,  "output": 0.60},
    "gemini-2.0-flash": {"input": 0.10,  "output": 0.40},
    "gemini-1.5-pro":   {"input": 1.25,  "output": 5.00},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
}

_OP_LABELS = {
    "extract_solutions":   "Extracción IA",
    "extract_teacher_pdf": "PDF profesor",
    "grade_submission":    "Corrección",
}


def _cost(model: str, inp: int, out: int) -> float:
    prices = next(
        (v for k, v in _TOKEN_PRICES.items() if model.startswith(k)),
        {"input": 0.0, "output": 0.0},
    )
    return (inp * prices["input"] + out * prices["output"]) / 1_000_000


@router.get("/usage", response_class=HTMLResponse)
def usage_report(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> HTMLResponse:
    all_rows = db.query(TokenUsage).order_by(TokenUsage.created_at).all()
    sessions = db.query(ExamSession).order_by(ExamSession.created_at.desc()).all()
    session_map = {s.id: s for s in sessions}

    # Exámenes corregidos por sesión
    done_by_session: dict[int, int] = {}
    for s in sessions:
        done_by_session[s.id] = db.query(Submission).filter(
            Submission.session_id == s.id,
            Submission.status == "done",
        ).count()

    # Agrupar por sesión → operación → modelo
    # session_data[session_id] = {
    #   "ops": { op: { model: {input, output, total, calls, cost} } },
    #   "totals": {input, output, total, calls, cost}
    # }
    session_data: dict[int | None, dict] = {}

    for row in all_rows:
        sid = row.session_id
        entry = session_data.setdefault(sid, {"ops": {}, "totals": _zero()})
        op_entry = entry["ops"].setdefault(row.operation, {})
        model_entry = op_entry.setdefault(row.model, _zero())

        c = _cost(row.model, row.input_tokens, row.output_tokens)
        _add(model_entry, row.input_tokens, row.output_tokens, row.total_tokens, row.api_calls, c)
        _add(entry["totals"], row.input_tokens, row.output_tokens, row.total_tokens, row.api_calls, c)

    # Construir lista ordenada de sesiones con datos
    sessions_report = []
    for sid, data in session_data.items():
        session = session_map.get(sid) if sid else None
        done = done_by_session.get(sid, 0) if sid else 0
        cost_per_exam = data["totals"]["cost"] / done if done > 0 else None

        # Aplanar ops: lista de {op_key, op_label, models: [...], subtotal}
        ops_list = []
        for op_key, models_dict in data["ops"].items():
            op_total = _zero()
            models_list = []
            for model_name, mdata in models_dict.items():
                models_list.append({"model": model_name, **mdata})
                _add(op_total, mdata["input"], mdata["output"], mdata["total"], mdata["calls"], mdata["cost"])
            ops_list.append({
                "key": op_key,
                "label": _OP_LABELS.get(op_key, op_key),
                "models": models_list,
                "subtotal": op_total,
            })

        sessions_report.append({
            "session": session,
            "session_id": sid,
            "done_exams": done,
            "cost_per_exam": cost_per_exam,
            "ops": ops_list,
            "totals": data["totals"],
        })

    # Ordenar: sesiones conocidas primero (por fecha desc), sin sesión al final
    sessions_report.sort(
        key=lambda x: (x["session"] is None, -(x["session"].id if x["session"] else 0))
    )

    # Totales globales
    grand = _zero()
    grand_done = 0
    for item in sessions_report:
        _add(grand, item["totals"]["input"], item["totals"]["output"],
             item["totals"]["total"], item["totals"]["calls"], item["totals"]["cost"])
        grand_done += item["done_exams"]
    grand["cost_per_exam"] = grand["cost"] / grand_done if grand_done > 0 else None
    grand["done_exams"] = grand_done

    return templates.TemplateResponse(
        "reports_usage.html",
        {
            "request": request,
            "user": current_user,
            "sessions_report": sessions_report,
            "grand": grand,
            "op_labels": _OP_LABELS,
        },
    )


def _zero() -> dict:
    return {"input": 0, "output": 0, "total": 0, "calls": 0, "cost": 0.0}


def _add(d: dict, inp: int, out: int, total: int, calls: int, cost: float) -> None:
    d["input"] += inp
    d["output"] += out
    d["total"] += total
    d["calls"] += calls
    d["cost"] += cost
