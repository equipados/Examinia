from __future__ import annotations

import pytest
from pydantic import ValidationError

from models import LLMPageExtraction, SolutionTemplate


def test_solution_template_parses_spanish_point_text() -> None:
    template = SolutionTemplate(
        exercise="1",
        part="a",
        expected_final_answer="2",
        max_points="1,5 puntos",
    )
    assert template.max_points == 1.5


def test_llm_page_extraction_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        LLMPageExtraction.model_validate(
            {
                "student_name": "A",
                "unknown_field": "x",
            }
        )
