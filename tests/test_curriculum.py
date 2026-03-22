"""Tests para el módulo de datos curriculares LOMLOE."""
from app.curriculum import (
    get_criterios,
    get_competencias_clave,
    get_competencias_especificas,
    get_cc_codes_from_criterio,
    get_criterios_for_cc,
    has_curriculum,
)


def test_has_curriculum_mat2() -> None:
    assert has_curriculum("2o_bachillerato", "Matemáticas II")
    assert not has_curriculum("2o_bachillerato", "Historia")
    assert not has_curriculum(None, None)


def test_criterios_count() -> None:
    criterios = get_criterios("2o_bachillerato", "Matemáticas II")
    assert len(criterios) == 17
    codes = [c.code for c in criterios]
    assert "1.1" in codes
    assert "9.3" in codes


def test_competencias_clave_count() -> None:
    cc = get_competencias_clave("2o_bachillerato", "Matemáticas II")
    assert len(cc) == 8
    codes = [c.code for c in cc]
    assert "STEM" in codes
    assert "CCL" in codes


def test_competencias_especificas_count() -> None:
    ce = get_competencias_especificas("2o_bachillerato", "Matemáticas II")
    assert len(ce) == 9


def test_cc_from_criterio() -> None:
    cc = get_cc_codes_from_criterio("1.1", "2o_bachillerato", "Matemáticas II")
    assert "STEM" in cc
    assert "CCL" in cc
    assert "CD" in cc


def test_criterios_for_cc_stem() -> None:
    crits = get_criterios_for_cc("STEM", "2o_bachillerato", "Matemáticas II")
    assert len(crits) > 10  # STEM aparece en casi todos los criterios
    assert "1.1" in crits


def test_criterios_for_cc_ccec() -> None:
    crits = get_criterios_for_cc("CCEC", "2o_bachillerato", "Matemáticas II")
    assert "6.2" in crits
    assert "7.1" in crits


def test_empty_for_unknown_course() -> None:
    assert get_criterios("3o_eso", "Ciencias") == []
    assert get_competencias_clave(None, None) == []
    assert get_cc_codes_from_criterio("1.1", None, None) == []
