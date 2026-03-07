# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run all tests
pytest -q

# Run a single test file
pytest tests/test_grading.py -q

# Run a single test by name
pytest tests/test_grading.py::test_function_name -q

# Install dependencies
pip install -r requirements.txt

# Run the corrector (basic)
python main.py

# Run with custom options
python main.py \
  --input-dir examenes \
  --solutions-dir soluciones \
  --output-dir salidas \
  --gemini-model gemini-2.5-flash \
  --gemini-solver-model gemini-2.5-pro \
  --poppler-path "C:\\tools\\poppler\\Library\\bin" \
  --log-level DEBUG
```

**Environment:** Requires `GEMINI_API_KEY` in `.env` or environment. Requires `poppler` installed (for `pdf2image`). Use `--poppler-path` if Poppler is not in `PATH`.

## Architecture

The pipeline runs in `main.py::run()` in this order:

1. **PDF -> Images** (`pdf_processor.py`): `discover_pdf_files` finds PDFs in `examenes/`; `convert_pdf_to_images` renders each page to PNG via Poppler.
2. **Preprocessing** (`image_preprocessing.py`): Optional image cleanup (binarization, deskew) before sending to Gemini.
3. **Page extraction** (`exam_parser.py`): `analyze_pages_with_gemini` sends page images to Gemini in batches. Returns `PageExtraction` per page, re-analyzing pages below `reanalysis_threshold` confidence.
4. **Segmentation** (`exam_segmenter.py`): `segment_exams` splits a multi-student PDF into individual `ExamSubmission` objects. The strong segmentation rule is detecting the printed header (logo + student name). Fallbacks: LLM boundary hint, student name change.
5. **Structure normalization** (`exam_parser.py`): `normalize_submission_structure` distributes points evenly across parts when only a question total is known.
6. **Grading** (`grading.py`): `grade_exam` processes each part:
   - Optionally calls `gemini_client.solve_math_question` (using `gemini_solver_model`) to resolve from the statement when no template exists or to enrich an existing one.
   - Calls `gemini_client.assess_math_answer` for AI-assisted classification.
   - Calls `decide_part_grade` (pure code logic) applying answer equivalence checks (`rapidfuzz` + `sympy`), partial credit rules, and confidence thresholds.
   - Generates student-facing feedback via `gemini_client.generate_feedback_explanation`.
7. **Reporting** (`reporting.py`): Writes per-student Markdown reports to `salidas/informes/`.
8. **Excel export** (`excel_export.py`): Writes `salidas/resultados.xlsx` with dynamic columns per part (`1.a`, `1.b`, etc.).

## Key modules

| File | Responsibility |
|---|---|
| `config.py` | `AppConfig` dataclass + `config_from_args` |
| `models.py` | All Pydantic models (data contracts between pipeline stages) |
| `gemini_client.py` | Wraps `google-genai`; handles retries, native schema fallback, JSON coercion |
| `grading.py` | Grading logic: `SolutionBank`, `decide_part_grade`, `grade_exam` |
| `utils.py` | `normalize_identifier`, `parse_point_string`, `round_points`, `part_column_id` |

## Data models flow

```
PDF -> PageImage -> PageExtraction -> ExamSubmission -> ExamGradeResult
                                          ^
                              SolutionTemplate (from soluciones/*.json)
```

All Pydantic models use `extra="forbid"`. `normalize_identifier` and `parse_point_string` are called in field validators to canonicalize IDs and point strings before comparison.

## Solution templates

JSON files in `soluciones/` are loaded into `SolutionBank`. Each file can be a single object, a list, or `{"solutions": [...]}`. `SolutionBank.find(exercise, part, exam_model)` resolves with exam-model specificity first, then part match, then generic (no part).

## GeminiClient behavior

- Uses `gemini-2.5-flash` for page extraction/assessment and `gemini-2.5-pro` for solving statements (configurable).
- Attempts native `response_schema` first; on `additional_properties` rejection, falls back to plain JSON + Pydantic validation.
- On invalid JSON from Gemini, retries with a stricter prompt up to `max_retries` times (tenacity, exponential backoff).
- Three coercion helpers (`_coerce_page_extraction_payload`, `_coerce_assessment_payload`, `_coerce_solved_exercise_payload`) normalize Gemini's flexible field naming to the canonical Pydantic schema.

## Grading logic (code-side, not AI)

`decide_part_grade` in `grading.py` applies deterministic rules in this priority:
1. No answer -> 0 pts, `incorrecto`
2. `revision_manual` or `illegible` -> 0 pts, `revision_manual`
3. Low confidence + strict mode -> `revision_manual`; without strict mode -> 25% conservative
4. Exact/equivalent final answer + good procedure -> full points
5. Exact answer + partial procedure -> 75%
6. Partial credit rules from template (keyword matching on condition string)
7. Arithmetic/sign error + good procedure -> 50%
8. Partial procedure -> 35%
9. Otherwise -> 0 pts, `incorrecto`
