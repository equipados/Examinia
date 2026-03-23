"""Servicio de envío de email para notificaciones de Examinia."""
from __future__ import annotations

import os
import smtplib
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from loguru import logger


def smtp_configured() -> bool:
    """Devuelve True si las variables SMTP mínimas están configuradas."""
    return bool(os.environ.get("SMTP_HOST") and os.environ.get("SMTP_FROM"))


def _smtp_send(msg: MIMEMultipart, to_email: str) -> bool:
    """Envía un mensaje MIME vía SMTP. Nunca lanza excepciones."""
    try:
        host = os.environ.get("SMTP_HOST", "")
        port = int(os.environ.get("SMTP_PORT", "587"))
        user = os.environ.get("SMTP_USER", "")
        password = os.environ.get("SMTP_PASSWORD", "")
        from_email = os.environ.get("SMTP_FROM", "")
        use_tls = os.environ.get("SMTP_USE_TLS", "true").lower() in ("true", "1", "yes")

        if not host or not from_email:
            logger.warning("SMTP no configurado, no se envía email")
            return False

        msg["From"] = from_email
        msg["To"] = to_email

        if port == 465:
            with smtplib.SMTP_SSL(host, port, timeout=60) as server:
                if user and password:
                    server.login(user, password)
                server.sendmail(from_email, [to_email], msg.as_string())
        elif use_tls:
            with smtplib.SMTP(host, port, timeout=60) as server:
                server.starttls()
                if user and password:
                    server.login(user, password)
                server.sendmail(from_email, [to_email], msg.as_string())
        else:
            with smtplib.SMTP(host, port, timeout=60) as server:
                if user and password:
                    server.login(user, password)
                server.sendmail(from_email, [to_email], msg.as_string())

        logger.info(f"Email enviado a {to_email}")
        return True

    except Exception as e:
        logger.error(f"Error enviando email a {to_email}: {e}")
        return False


def send_session_completion_email(
    to_email: str,
    session_name: str,
    results: list[dict],
    teacher_name: str | None = None,
) -> bool:
    """Envía email con resumen de corrección al profesor. Nunca lanza excepciones.

    results: lista de dicts con keys 'student_name', 'total', 'max', 'status'.
    """
    try:
        done = [r for r in results if r.get("status") == "done"]
        error_count = sum(1 for r in results if r.get("status") == "error")
        avg = sum(r["total"] for r in done) / len(done) if done else 0

        rows_html = ""
        for r in sorted(done, key=lambda x: x.get("total", 0), reverse=True):
            pct = (r["total"] / r["max"] * 100) if r["max"] else 0
            color = "#16a34a" if pct >= 70 else "#d97706" if pct >= 50 else "#dc2626"
            rows_html += (
                f'<tr>'
                f'<td style="padding:6px 12px;border-bottom:1px solid #e5e7eb">{r["student_name"]}</td>'
                f'<td style="padding:6px 12px;border-bottom:1px solid #e5e7eb;text-align:right;'
                f'color:{color};font-weight:600">{r["total"]:.2f} / {r["max"]:.2f}</td>'
                f'</tr>'
            )

        greeting = f"<p>Hola {teacher_name},</p>" if teacher_name else ""

        html = f"""
        <div style="font-family:system-ui,sans-serif;max-width:600px;margin:0 auto">
            {greeting}
            <h2 style="color:#1e3a5f">Corrección completada: {session_name}</h2>
            <div style="background:#f0f9ff;border:1px solid #bae6fd;border-radius:8px;padding:16px;margin:16px 0">
                <p style="margin:0"><strong>Alumnos corregidos:</strong> {len(done)}</p>
                <p style="margin:4px 0 0"><strong>Nota media:</strong> {avg:.2f}</p>
                {"<p style='margin:4px 0 0;color:#dc2626'><strong>Errores:</strong> " + str(error_count) + "</p>" if error_count else ""}
            </div>
            <table style="width:100%;border-collapse:collapse;font-size:14px">
                <thead>
                    <tr style="background:#f8fafc">
                        <th style="padding:8px 12px;text-align:left;border-bottom:2px solid #e5e7eb">Alumno</th>
                        <th style="padding:8px 12px;text-align:right;border-bottom:2px solid #e5e7eb">Nota</th>
                    </tr>
                </thead>
                <tbody>{rows_html}</tbody>
            </table>
            <p style="color:#9ca3af;font-size:12px;margin-top:24px">
                Enviado automáticamente por Examinia.
            </p>
        </div>
        """

        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"Examinia — Corrección completada: {session_name}"
        msg.attach(MIMEText(html, "html", "utf-8"))

        return _smtp_send(msg, to_email)

    except Exception as e:
        logger.error(f"Error preparando email para {to_email}: {e}")
        return False


def send_student_report_email(
    to_email: str,
    student_name: str,
    session_name: str,
    combined_pdf: bytes,
    annotated_pdf: bytes | None = None,
) -> bool:
    """Envía al alumno el informe de corrección y opcionalmente el examen corregido anotado."""
    try:
        has_annotated = annotated_pdf is not None
        extra_line = (
            "<p>Incluimos también tu examen con las correcciones anotadas directamente sobre tus respuestas.</p>"
            if has_annotated else ""
        )
        html = f"""
        <div style="font-family:system-ui,sans-serif;max-width:600px;margin:0 auto">
            <p>Hola {student_name},</p>
            <p>Adjuntamos el informe de corrección de la prueba <strong>{session_name}</strong>.</p>
            {extra_line}
            <p style="color:#9ca3af;font-size:12px;margin-top:24px">
                Enviado automáticamente por Examinia.
            </p>
        </div>
        """

        msg = MIMEMultipart("mixed")
        msg["Subject"] = f"Examinia — Informe de corrección: {session_name}"
        msg.attach(MIMEText(html, "html", "utf-8"))

        safe_name = student_name.replace(" ", "_")
        att = MIMEApplication(combined_pdf, _subtype="pdf")
        att.add_header("Content-Disposition", "attachment", filename=f"informe_{safe_name}.pdf")
        msg.attach(att)

        if annotated_pdf:
            att2 = MIMEApplication(annotated_pdf, _subtype="pdf")
            att2.add_header("Content-Disposition", "attachment", filename=f"corregido_{safe_name}.pdf")
            msg.attach(att2)

        return _smtp_send(msg, to_email)

    except Exception as e:
        logger.error(f"Error preparando email de informe para {to_email}: {e}")
        return False
