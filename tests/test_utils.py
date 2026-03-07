from __future__ import annotations

from utils import deduplicate_name, safe_filename


def test_safe_filename_removes_unsafe_chars() -> None:
    assert safe_filename("Apellido, Nombre?.pdf") == "apellido_nombrepdf"


def test_deduplicate_name_adds_suffix() -> None:
    seen: dict[str, int] = {}
    first = deduplicate_name("juan_perez", seen)
    second = deduplicate_name("juan_perez", seen)
    assert first == "juan_perez"
    assert second == "juan_perez_2"
