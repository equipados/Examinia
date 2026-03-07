from __future__ import annotations

import argparse

from loguru import logger

from config import AppConfig, config_from_args
from exam_parser import analyze_pages_with_gemini, build_submission_from_pdf, normalize_submission_structure
from excel_export import export_results_to_excel
from gemini_client import GeminiClient
from grading import grade_exam, load_solution_bank
from image_preprocessing import preprocess_pages
from pdf_processor import convert_pdf_to_images, discover_pdf_files
from reporting import write_exam_report
from utils import ensure_dir, setup_logger


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Corrector automatico de examenes de Matematicas II con Gemini Flash."
    )
    parser.add_argument("--input-dir", default="examenes", help="Carpeta con PDFs escaneados")
    parser.add_argument("--output-dir", default="salidas", help="Carpeta de salida")
    parser.add_argument("--solutions-dir", default="soluciones", help="Carpeta con plantillas de solucion JSON")
    parser.add_argument("--log-level", default="INFO", help="Nivel de log (DEBUG, INFO, WARNING, ERROR)")
    parser.add_argument(
        "--strict-mode",
        action="store_true",
        help="Modo estricto: ante baja confianza prioriza revision manual",
    )
    parser.add_argument(
        "--disable-ai-solver",
        action="store_true",
        help="Desactiva resolucion del enunciado por IA antes de comparar con la respuesta del alumno",
    )
    parser.add_argument(
        "--ai-solver-min-confidence",
        type=float,
        default=0.30,
        help="Confianza minima para aceptar la solucion generada por IA desde enunciado",
    )
    parser.add_argument(
        "--poppler-path",
        default=None,
        help="Ruta a carpeta de binarios de Poppler (donde estan pdftoppm.exe y pdfinfo.exe)",
    )
    parser.add_argument("--dpi", type=int, default=220, help="Resolucion de conversion PDF->imagen")
    parser.add_argument("--no-preprocess", action="store_true", help="Desactiva preprocesado de imagenes")
    parser.add_argument("--page-batch-limit", type=int, default=12, help="Limite de paginas por lote de analisis")
    parser.add_argument("--gemini-model", default="gemini-2.5-flash", help="Modelo Gemini a usar")
    parser.add_argument(
        "--gemini-solver-model",
        default="gemini-2.5-pro",
        help="Modelo Gemini dedicado a resolver enunciados (mas capacidad matematica)",
    )
    parser.add_argument(
        "--low-confidence-threshold",
        type=float,
        default=0.72,
        help="Umbral por debajo del cual se marca baja confianza",
    )
    parser.add_argument(
        "--reanalysis-threshold",
        type=float,
        default=0.55,
        help="Umbral de confianza para reanalizar paginas automaticamente",
    )
    parser.add_argument(
        "--header-confidence-threshold",
        type=float,
        default=0.80,
        help="Umbral para considerar fiable el encabezado de inicio (logo+nombre)",
    )
    parser.add_argument("--max-retries", type=int, default=3, help="Reintentos de llamadas Gemini")
    parser.add_argument(
        "--rate-limit-seconds",
        type=float,
        default=0.0,
        help="Espera minima entre llamadas consecutivas a Gemini",
    )
    parser.add_argument("--request-timeout", type=int, default=120, help="Timeout nominal de peticion (segundos)")
    return parser


def run(config: AppConfig) -> int:
    setup_logger(config.log_level)
    ensure_dir(config.output_dir)
    ensure_dir(config.reports_dir)
    ensure_dir(config.temp_dir)

    logger.info(f"Entrada: {config.input_dir}")
    logger.info(f"Salida: {config.output_dir}")
    logger.info(f"Soluciones: {config.solutions_dir}")
    logger.info(f"Modelo Gemini: {config.gemini_model}")
    logger.info(f"Modelo Gemini solver: {config.gemini_solver_model}")
    logger.info(f"IA resuelve enunciado: {'SI' if config.enable_ai_solver else 'NO'}")
    if config.enable_ai_solver:
        logger.info(f"Umbral confianza solver IA: {config.ai_solver_min_confidence:.2f}")
    logger.info(f"Umbral header inicio examen: {config.header_confidence_threshold:.2f}")
    if config.poppler_path:
        logger.info(f"Poppler path: {config.poppler_path}")
        if not config.poppler_path.exists():
            logger.error(f"La ruta de Poppler no existe: {config.poppler_path}")
            return 1

    try:
        gemini = GeminiClient(
            model=config.gemini_model,
            solver_model=config.gemini_solver_model,
            max_retries=config.max_retries,
            rate_limit_seconds=config.rate_limit_seconds,
            request_timeout_seconds=config.request_timeout_seconds,
        )
    except Exception as exc:
        logger.error(f"No se pudo inicializar Gemini: {exc}")
        return 1

    solution_bank = load_solution_bank(config.solutions_dir)
    pdf_files = discover_pdf_files(config.input_dir)
    if not pdf_files:
        logger.warning("No hay PDFs para procesar.")
        return 0

    all_results = []
    report_names_seen: dict[str, int] = {}

    for pdf_path in pdf_files:
        logger.info(f"Procesando archivo: {pdf_path.name}")
        pdf_temp_dir = ensure_dir(config.temp_dir / pdf_path.stem)
        raw_images_dir = ensure_dir(pdf_temp_dir / "raw")
        preprocessed_dir = ensure_dir(pdf_temp_dir / "preprocessed")

        logger.info(f"  Convirtiendo PDF a imagenes (DPI={config.dpi})...")
        try:
            page_images = convert_pdf_to_images(
                pdf_path=pdf_path,
                output_dir=raw_images_dir,
                dpi=config.dpi,
                poppler_path=config.poppler_path,
            )
        except Exception as exc:
            logger.error(f"Se omite {pdf_path.name} por error de conversion: {exc}")
            continue
        logger.info(f"  PDF convertido: {len(page_images)} pagina(s)")

        if config.preprocess_images:
            logger.info("  Preprocesando imagenes...")
            pages_for_analysis = preprocess_pages(page_images=page_images, output_dir=preprocessed_dir)
        else:
            pages_for_analysis = page_images

        page_extractions = []
        for start in range(0, len(pages_for_analysis), config.page_batch_limit):
            batch = pages_for_analysis[start : start + config.page_batch_limit]
            logger.info(
                f"  Extrayendo paginas {batch[0].page_number}-{batch[-1].page_number} con Gemini..."
            )
            batch_extractions = analyze_pages_with_gemini(
                page_images=batch,
                gemini_client=gemini,
                low_confidence_threshold=config.low_confidence_threshold,
                reanalysis_threshold=config.reanalysis_threshold,
            )
            page_extractions.extend(batch_extractions)

        total_questions = sum(len(e.questions) for e in page_extractions)
        logger.info(f"  Extraccion completada: {len(page_extractions)} pagina(s), {total_questions} ejercicio(s) detectado(s)")

        submission = build_submission_from_pdf(page_extractions, pdf_path.name)
        logger.info(f"  Alumno detectado: {submission.student_name or 'desconocido'} | Modelo: {submission.exam_model or 'no detectado'}")

        normalized_exam = normalize_submission_structure(submission)
        total_parts = sum(len(q.parts) for q in normalized_exam.questions)
        logger.info(f"  Estructura normalizada: {len(normalized_exam.questions)} ejercicio(s), {total_parts} apartado(s)")

        logger.info(f"  Iniciando correccion con Gemini...")
        graded = grade_exam(
            submission=normalized_exam,
            solution_bank=solution_bank,
            gemini_client=gemini,
            low_confidence_threshold=config.low_confidence_threshold,
            strict_mode=config.strict_mode,
            allow_ai_solver=config.enable_ai_solver,
            ai_solver_min_confidence=config.ai_solver_min_confidence,
        )
        logger.info(
            f"  Correccion finalizada: {graded.total_points:.2f}/{graded.max_total_points:.2f} puntos"
        )

        report_path = write_exam_report(
            result=graded,
            reports_dir=config.reports_dir,
            seen_filenames=report_names_seen,
        )
        graded.report_path = str(report_path)
        logger.info(f"  Informe generado: {report_path.name}")
        all_results.append(graded)

    excel_path = config.output_dir / "resultados.xlsx"
    export_results_to_excel(results=all_results, output_file=excel_path)
    logger.info(f"Excel generado: {excel_path}")
    logger.info(f"Examenes corregidos: {len(all_results)}")
    return 0


def main() -> int:
    parser = build_cli()
    args = parser.parse_args()
    config = config_from_args(args)
    return run(config=config)


if __name__ == "__main__":
    raise SystemExit(main())
