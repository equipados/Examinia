from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class AppConfig:
    input_dir: Path = Path("examenes")
    output_dir: Path = Path("salidas")
    solutions_dir: Path = Path("soluciones")
    reports_dir_name: str = "informes"
    temp_dir_name: str = ".tmp"
    log_level: str = "INFO"
    strict_mode: bool = False
    enable_ai_solver: bool = True
    ai_solver_min_confidence: float = 0.30
    poppler_path: Path | None = None
    dpi: int = 220
    preprocess_images: bool = True
    page_batch_limit: int = 12
    gemini_model: str = "gemini-2.5-flash"
    gemini_solver_model: str = "gemini-2.5-pro"
    low_confidence_threshold: float = 0.72
    reanalysis_threshold: float = 0.55
    header_confidence_threshold: float = 0.80
    max_retries: int = 3
    rate_limit_seconds: float = 0.0
    request_timeout_seconds: int = 120
    # Web server
    web_upload_dir: Path = Path("web_uploads")
    web_db_path: Path = Path("corrector.db")
    web_secret_key: str = "change-me-in-production"

    @property
    def reports_dir(self) -> Path:
        return self.output_dir / self.reports_dir_name

    @property
    def temp_dir(self) -> Path:
        return self.output_dir / self.temp_dir_name


def config_from_args(args: Namespace) -> AppConfig:
    return AppConfig(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        solutions_dir=Path(args.solutions_dir),
        log_level=args.log_level,
        strict_mode=args.strict_mode,
        enable_ai_solver=not args.disable_ai_solver,
        ai_solver_min_confidence=args.ai_solver_min_confidence,
        poppler_path=Path(args.poppler_path) if args.poppler_path else None,
        dpi=args.dpi,
        preprocess_images=not args.no_preprocess,
        page_batch_limit=args.page_batch_limit,
        gemini_model=args.gemini_model,
        gemini_solver_model=args.gemini_solver_model,
        low_confidence_threshold=args.low_confidence_threshold,
        reanalysis_threshold=args.reanalysis_threshold,
        header_confidence_threshold=args.header_confidence_threshold,
        max_retries=args.max_retries,
        rate_limit_seconds=args.rate_limit_seconds,
        request_timeout_seconds=args.request_timeout,
    )
