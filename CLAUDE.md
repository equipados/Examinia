# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run all tests
pytest -q

# Run a single test file
pytest tests/test_grading.py -q

# Run the web app (primary interface)
python web.py   # listens on 0.0.0.0:8000

# Run the corrector (CLI, legacy)
python main.py \
  --input-dir examenes \
  --solutions-dir soluciones \
  --output-dir salidas \
  --gemini-model gemini-2.5-flash \
  --gemini-solver-model gemini-2.5-pro \
  --poppler-path "C:\\tools\\poppler\\Library\\bin" \
  --log-level DEBUG
```

**Python:** Always use `C:\Users\glyco\AppData\Local\Python\pythoncore-3.14-64\python.exe` — never `py` or `python`.

**Environment:** `.env` file requires:
- `GEMINI_API_KEY` — required always (page extraction + assessment)
- `OPENAI_API_KEY` — optional, for OpenAI solver
- `DEEPSEEK_API_KEY` — optional, for DeepSeek solver
- `POPPLER_PATH` — path to Poppler bin

**Version:** `APP_VERSION` in `web.py`. Current: `1.2.0`. Bump on every change.

## Web app architecture

Primary interface is FastAPI (`web.py`). Admin user: `admin` / `admin123`.

```
web.py                  FastAPI entry, lifespan, APP_VERSION
app/database.py         SQLAlchemy + WAL mode
app/db_models.py        ORM: User, ExamSession, Submission, QuestionResult, PartResult,
                              SessionSolution, TokenUsage, SessionHistory
app/auth.py             bcrypt (direct, not passlib), JWT in HttpOnly cookie
app/scheduler.py        ThreadPoolExecutor(max_workers=2), background pipeline
app/templates.py        Jinja2 shared instance, filters: from_json, basename
app/pricing.py          Centralized token prices + cost_eur() + provider_of()
app/routers/            auth, dashboard, sessions, submissions, reports
openai_solver.py        OpenAI-compatible solver (OpenAI + DeepSeek via base_url)
```

## Solver architecture (multi-provider)

`_make_solver(solver_provider, cfg)` in `scheduler.py` creates the right solver:

| `solver_provider` value | Model | API key needed |
|---|---|---|
| `gemini-pro` | gemini-2.5-pro | GEMINI_API_KEY |
| `gemini-flash` | gemini-2.5-flash | GEMINI_API_KEY |
| `openai-gpt4o` | gpt-4o | OPENAI_API_KEY |
| `openai-o4mini` | o4-mini | OPENAI_API_KEY |
| `deepseek-v3` | deepseek-chat | DEEPSEEK_API_KEY |
| `deepseek-r1` | deepseek-reasoner | DEEPSEEK_API_KEY |

`OpenAISolver` (in `openai_solver.py`) handles OpenAI and DeepSeek (same API, different `base_url`).
DeepSeek base URL: `https://api.deepseek.com`.

Gemini is always used for page extraction/OCR (vision required). The solver is only for `solve_math_question` (text-only).

`_CachingGeminiWrapper` wraps `GeminiClient` and intercepts `solve_math_question`:
- Uses `self._solver` (set per-correction) for the actual solving call
- Caches results in DB (`session_solutions`) to avoid repeated API calls
- All other methods (`assess_math_answer`, `generate_feedback_explanation`) delegate to `self._client` (always GeminiClient)

## Web correction flow

1. Create session (`max_total_points` optional, `solver_provider` selectable)
2. Upload student PDFs
3. Solutions page: choose **IA mode** (AI solves) or **Profesor mode** (upload PDF with answers)
4. **Teacher PDF extraction** (`_run_extract_teacher_solutions`):
   - Extracts answers, `question_max_points` (from "(3p)" notation), evaluation criteria
   - Prorates `question_max_points` across parts → saves in `SessionSolution.max_points`
   - Saves `evaluation_criteria` in each `SessionSolution` row
5. Validate solutions individually or with "Validate all"
6. Start grading → `_run_pipeline` per submission:
   - Extracts student answers from PDF
   - **Point priority**: (1) teacher PDF max_points → (2) student PDF extracted points → (3) equitative
   - Passes `evaluation_criteria` to `assess_math_answer` prompt
   - Log shows: `Puntos extraídos del PDF: Ej1=3.0p (1 apt) | Ej2=5.0p (5 apt) | ...`

## Point extraction logic

The exam cover page prompt (`extract_exam_questions`) explicitly asks Gemini to extract
`question_max_points` from bold notation like "(3p)", "(5p)", "(2,5 pts)".
`normalize_submission_structure` distributes question-level points to parts equitably.
`_apply_session_max_points` additionally handles scaling if total differs from `max_total_points`.

## DB schema notes (migrations applied manually via ALTER TABLE)

`session_solutions` extra columns (added post-creation):
- `max_points REAL` — points per part (prorated from question_max_points)
- `evaluation_criteria TEXT` — grading criteria from exam footer

`exam_sessions` extra columns:
- `current_step TEXT`, `session_log TEXT`, `solution_mode TEXT DEFAULT 'ai'`
- `max_total_points REAL DEFAULT NULL`, `solver_provider TEXT`

`submissions` extra columns:
- `processing_log TEXT`

Table `token_usage` — created with CREATE TABLE IF NOT EXISTS.

## Token usage tracking

`gemini_client.py` accumulates tokens in `_usage` dict per model; `get_usage()` / `reset_usage()`.
`_save_token_usage(db, client, operation, session_id, submission_id)` saves to `token_usage` table.
Operations: `extract_solutions`, `extract_teacher_pdf`, `grade_submission`.
`_build_token_summary` groups by `(operation, model)` — shows model column in cost table.
View at `/reports/usage` (navbar "Costes IA"). Also shown in `session_detail` collapsible block.

Token prices in `app/pricing.py`:
- Gemini Flash: $0.15/$0.60 per 1M tokens
- Gemini Pro: $1.25/$10.00
- GPT-4o: $2.50/$10.00
- o4-mini: $1.10/$4.40
- DeepSeek V3: $0.27/$1.10
- DeepSeek R1: $0.55/$2.19

## Pipeline architecture (CLI, legacy)

`main.py::run()` order:
1. **PDF → Images** (`pdf_processor.py`)
2. **Preprocessing** (`image_preprocessing.py`)
3. **Page extraction** (`exam_parser.py`): `analyze_pages_with_gemini` → `PageExtraction` per page
4. **Segmentation** (`exam_segmenter.py`): splits multi-student PDF → `ExamSubmission` objects
5. **Structure normalization** (`exam_parser.py`): `normalize_submission_structure`
6. **Grading** (`grading.py`): `grade_exam` → `decide_part_grade`
7. **Reporting** (`reporting.py`): Markdown reports
8. **Excel export** (`excel_export.py`): `resultados.xlsx`

## Key modules

| File | Responsibility |
|---|---|
| `config.py` | `AppConfig` dataclass + `config_from_args` |
| `models.py` | All Pydantic models. `TeacherSolutionItem` has `question_max_points`. `TeacherSolutionsExtraction` has `evaluation_criteria`. |
| `gemini_client.py` | Wraps `google-genai`; retries, schema fallback, JSON coercion. All prompts in Spanish. |
| `grading.py` | `SolutionBank`, `decide_part_grade`, `grade_exam` (accepts `evaluation_criteria`) |
| `utils.py` | `normalize_identifier`, `parse_point_string`, `round_points`, `part_column_id` |
| `openai_solver.py` | `OpenAISolver`: OpenAI + DeepSeek (via `base_url` + `api_key_env` params) |
| `app/pricing.py` | `TOKEN_PRICES`, `cost_eur()`, `provider_of()` — Gemini/OpenAI/DeepSeek |

## Data models flow

```
PDF -> PageImage -> PageExtraction -> ExamSubmission -> ExamGradeResult
                                          ^
                        SolutionTemplate (from SessionSolution or soluciones/*.json)
```

All Pydantic models use `extra="forbid"`.

## GeminiClient prompts

- `extract_exam_questions`: cover page — extracts structure, enunciados, and **`question_max_points`** from "(3p)" notation. Always returns `page_role="cover"`.
- `extract_student_answers`: subsequent pages — OCR of handwritten answers.
- `extract_teacher_solutions_from_pages`: teacher PDF — extracts answers + `question_max_points` + `evaluation_criteria`. Ignores cover/carátula page.
- `assess_math_answer`: accepts optional `evaluation_criteria` param, included in prompt.
- `solve_math_question`: two variants (from image / from text). Both instruct Spanish output.
- All prompts include "IMPORTANTE: Responde SIEMPRE en español."

## Grading logic (code-side)

`decide_part_grade` priority:
1. No answer → 0 pts, `incorrecto`
2. `revision_manual` or `illegible` → 0 pts, `revision_manual`
3. Low confidence + strict mode → `revision_manual`; else 25% conservative
4. Exact/equivalent answer + good procedure → full points
5. Exact answer + partial procedure → 75%
6. Partial credit rules from template
7. Arithmetic/sign error + good procedure → 50%
8. Partial procedure → 35%
9. Otherwise → 0 pts, `incorrecto`
