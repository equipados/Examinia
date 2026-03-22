"""Datos curriculares LOMLOE (Decreto 30/2023, Canarias) para Matemáticas de Bachillerato.

Contiene criterios de evaluación, competencias específicas, competencias clave,
descriptores operativos y saberes básicos. Usado para mapear preguntas de examen
a criterios y generar vistas de progreso por competencias.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CriterioEvaluacion:
    code: str                         # "1.1", "2.1", etc.
    ce_code: str                      # Competencia específica: "C1", "C2"
    description: str                  # Descripción corta del criterio
    descriptores: list[str] = field(default_factory=list)  # ["CCL2", "STEM1", ...]
    saberes: list[str] = field(default_factory=list)       # Bloques de saberes básicos


@dataclass
class CompetenciaEspecifica:
    code: str       # "C1", "C2", ...
    name: str       # Nombre corto
    description: str


@dataclass
class CompetenciaClave:
    code: str   # "CCL", "STEM", "CD", etc.
    name: str   # Nombre completo


# ---------------------------------------------------------------------------
# Competencias clave (comunes a todo Bachillerato)
# ---------------------------------------------------------------------------
COMPETENCIAS_CLAVE = [
    CompetenciaClave("CCL", "Competencia en comunicación lingüística"),
    CompetenciaClave("CP", "Competencia plurilingüe"),
    CompetenciaClave("STEM", "Competencia matemática y en ciencia, tecnología e ingeniería"),
    CompetenciaClave("CD", "Competencia digital"),
    CompetenciaClave("CPSAA", "Competencia personal, social y de aprender a aprender"),
    CompetenciaClave("CC", "Competencia ciudadana"),
    CompetenciaClave("CE", "Competencia emprendedora"),
    CompetenciaClave("CCEC", "Competencia en conciencia y expresión culturales"),
]

# ---------------------------------------------------------------------------
# Matemáticas II — 2º Bachillerato
# ---------------------------------------------------------------------------

_CE_MAT2 = [
    CompetenciaEspecifica("C1", "Modelizar y resolver problemas",
        "Modelizar y resolver problemas de la vida cotidiana y de la ciencia y la tecnología "
        "aplicando diferentes estrategias y formas de razonamiento."),
    CompetenciaEspecifica("C2", "Verificar soluciones",
        "Verificar la validez de las posibles soluciones de un problema empleando el razonamiento "
        "y la argumentación."),
    CompetenciaEspecifica("C3", "Formular conjeturas",
        "Formular o investigar conjeturas o problemas, utilizando el razonamiento, la argumentación, "
        "la creatividad y el uso de herramientas tecnológicas."),
    CompetenciaEspecifica("C4", "Pensamiento computacional",
        "Utilizar el pensamiento computacional de forma eficaz, modificando, creando y generalizando "
        "algoritmos que resuelvan problemas."),
    CompetenciaEspecifica("C5", "Conexiones matemáticas",
        "Establecer, investigar y utilizar conexiones entre las diferentes ideas matemáticas."),
    CompetenciaEspecifica("C6", "Vínculos interdisciplinares",
        "Descubrir los vínculos de las matemáticas con otras áreas de conocimiento y profundizar "
        "en sus conexiones."),
    CompetenciaEspecifica("C7", "Representar conceptos",
        "Representar conceptos, procedimientos e información matemática seleccionando diferentes "
        "tecnologías."),
    CompetenciaEspecifica("C8", "Comunicar ideas",
        "Comunicar las ideas matemáticas empleando el soporte, la terminología y el rigor apropiados."),
    CompetenciaEspecifica("C9", "Destrezas personales y sociales",
        "Utilizar destrezas personales y sociales, identificando y gestionando las propias emociones."),
]

_CRITERIOS_MAT2: list[CriterioEvaluacion] = [
    # --- C1: Modelizar y resolver problemas ---
    CriterioEvaluacion(
        code="1.1", ce_code="C1",
        description="Manejar diferentes estrategias y herramientas para describir, analizar y modelizar problemas.",
        descriptores=["CCL2", "STEM1", "STEM2", "STEM3", "CD2", "CD3", "CD5", "CPSAA4", "CPSAA5", "CE3"],
        saberes=["Sentido numérico", "Sentido algebraico", "Sentido de la medida"],
    ),
    CriterioEvaluacion(
        code="1.2", ce_code="C1",
        description="Obtener todas las posibles soluciones matemáticas de problemas con autonomía.",
        descriptores=["CCL1", "STEM1", "STEM3", "CD3", "CD5", "CE3"],
        saberes=["Sentido numérico", "Sentido algebraico"],
    ),

    # --- C2: Verificar soluciones ---
    CriterioEvaluacion(
        code="2.1", ce_code="C2",
        description="Demostrar la validez matemática de las posibles soluciones de un problema.",
        descriptores=["CCL1", "CCL2", "STEM1", "STEM2", "CD2", "CD3", "CPSAA4"],
        saberes=["Sentido algebraico", "Sentido numérico"],
    ),
    CriterioEvaluacion(
        code="2.2", ce_code="C2",
        description="Seleccionar la solución más adecuada de un problema en función del contexto.",
        descriptores=["CCL1", "STEM1", "CD2", "CD3", "CC3", "CE3"],
        saberes=["Sentido algebraico"],
    ),

    # --- C3: Formular conjeturas ---
    CriterioEvaluacion(
        code="3.1", ce_code="C3",
        description="Formular, investigar y justificar conjeturas y problemas con creatividad.",
        descriptores=["CCL1", "STEM1", "STEM2", "STEM4", "CD1", "CD2", "CD3", "CD5", "CE3"],
        saberes=["Sentido algebraico", "Sentido espacial"],
    ),

    # --- C4: Pensamiento computacional ---
    CriterioEvaluacion(
        code="4.1", ce_code="C4",
        description="Modificar, crear y generalizar algoritmos para resolver situaciones problemáticas.",
        descriptores=["CCL2", "STEM1", "STEM2", "STEM3", "CD2", "CD3", "CD5", "CE3"],
        saberes=["Sentido algebraico", "Sentido numérico"],
    ),

    # --- C5: Conexiones matemáticas ---
    CriterioEvaluacion(
        code="5.1", ce_code="C5",
        description="Conectar las diferentes ideas matemáticas identificando los vínculos existentes.",
        descriptores=["CCL3", "STEM1", "STEM2", "STEM3", "CD1", "CD2", "CD3"],
        saberes=["Sentido algebraico", "Sentido de la medida", "Sentido numérico"],
    ),
    CriterioEvaluacion(
        code="5.2", ce_code="C5",
        description="Resolver problemas estableciendo y aplicando conexiones entre ideas matemáticas.",
        descriptores=["STEM1", "STEM2", "CD3", "CPSAA5"],
        saberes=["Sentido algebraico", "Sentido de la medida"],
    ),

    # --- C6: Vínculos interdisciplinares ---
    CriterioEvaluacion(
        code="6.1", ce_code="C6",
        description="Establecer conexiones entre ideas matemáticas y otras áreas de conocimiento.",
        descriptores=["CCL2", "STEM1", "STEM2", "CD1", "CD3", "CPSAA5", "CC4", "CE3"],
        saberes=["Sentido de la medida", "Sentido estocástico"],
    ),
    CriterioEvaluacion(
        code="6.2", ce_code="C6",
        description="Analizar la aportación de las matemáticas al progreso de la humanidad.",
        descriptores=["CCL2", "STEM2", "CD1", "CD2", "CPSAA5", "CC4", "CCEC1"],
        saberes=["Sentido socioafectivo"],
    ),

    # --- C7: Representar conceptos ---
    CriterioEvaluacion(
        code="7.1", ce_code="C7",
        description="Representar conceptos e información matemática para visualizar ideas.",
        descriptores=["CCL2", "STEM2", "STEM4", "CD1", "CD2", "CD3", "CD5", "CE3", "CCEC4.1", "CCEC4.2"],
        saberes=["Sentido espacial", "Sentido algebraico"],
    ),
    CriterioEvaluacion(
        code="7.2", ce_code="C7",
        description="Seleccionar y combinar diversas formas de representación matemática.",
        descriptores=["CCL1", "CCL3", "STEM1", "STEM3", "STEM4", "CD2", "CD3", "CE3"],
        saberes=["Sentido espacial", "Sentido algebraico"],
    ),

    # --- C8: Comunicar ideas ---
    CriterioEvaluacion(
        code="8.1", ce_code="C8",
        description="Comunicar hechos, ideas y procedimientos complejos con la terminología apropiada.",
        descriptores=["CCL1", "CCL3", "STEM2", "STEM4", "CD2", "CD3", "CCEC3.2"],
        saberes=["Sentido algebraico", "Sentido espacial"],
    ),
    CriterioEvaluacion(
        code="8.2", ce_code="C8",
        description="Reconocer y emplear el lenguaje matemático en diferentes contextos.",
        descriptores=["CCL1", "CCL2", "CCL3", "STEM2", "STEM4", "CD3"],
        saberes=["Sentido algebraico"],
    ),

    # --- C9: Destrezas personales y sociales ---
    CriterioEvaluacion(
        code="9.1", ce_code="C9",
        description="Perseverar en la consecución de objetivos ante situaciones de incertidumbre.",
        descriptores=["STEM5", "CPSAA1.1", "CPSAA1.2", "CC1", "CE2"],
        saberes=["Sentido socioafectivo"],
    ),
    CriterioEvaluacion(
        code="9.2", ce_code="C9",
        description="Aceptar y aprender de la crítica razonada con actitud positiva y cooperativa.",
        descriptores=["CCL1", "CPSAA1.1", "CPSAA1.2", "CPSAA3.1", "CC1", "CC3", "CE2"],
        saberes=["Sentido socioafectivo"],
    ),
    CriterioEvaluacion(
        code="9.3", ce_code="C9",
        description="Trabajar en tareas matemáticas en equipos heterogéneos respetando la diversidad.",
        descriptores=["CCL1", "CPSAA1.1", "CPSAA1.2", "CPSAA3.1", "CPSAA3.2", "CC1", "CC2", "CC3", "CE2"],
        saberes=["Sentido socioafectivo"],
    ),
]

_SABERES_MAT2 = [
    "Sentido numérico",
    "Sentido de la medida",
    "Sentido espacial",
    "Sentido algebraico",
    "Sentido estocástico",
    "Sentido socioafectivo",
]

# ---------------------------------------------------------------------------
# Registro de currículos
# ---------------------------------------------------------------------------

CURRICULA: dict[tuple[str, str], dict] = {
    ("2o_bachillerato", "Matemáticas II"): {
        "criterios": _CRITERIOS_MAT2,
        "competencias_especificas": _CE_MAT2,
        "competencias_clave": COMPETENCIAS_CLAVE,
        "saberes_basicos": _SABERES_MAT2,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_subject(subject: str) -> str:
    """Normaliza variantes comunes de nombres de asignatura."""
    s = subject.strip()
    _ALIASES: dict[str, str] = {
        "Matemáticas": "Matemáticas II",
        "Mates II": "Matemáticas II",
        "Mat II": "Matemáticas II",
        "Matemáticas 2": "Matemáticas II",
        "Mates I": "Matemáticas I",
        "Mat I": "Matemáticas I",
        "Matemáticas 1": "Matemáticas I",
    }
    return _ALIASES.get(s, s)


def get_curriculum(course_level: str | None, subject: str | None) -> dict | None:
    """Devuelve el currículo para un curso y asignatura, o None si no existe."""
    if not course_level or not subject:
        return None
    normalized = _normalize_subject(subject)
    return CURRICULA.get((course_level, normalized))


def get_criterios(course_level: str | None, subject: str | None) -> list[CriterioEvaluacion]:
    """Devuelve los criterios de evaluación para un curso y asignatura."""
    cur = get_curriculum(course_level, subject)
    return cur["criterios"] if cur else []


def get_competencias_clave(course_level: str | None, subject: str | None) -> list[CompetenciaClave]:
    """Devuelve las competencias clave disponibles."""
    cur = get_curriculum(course_level, subject)
    return cur["competencias_clave"] if cur else []


def get_competencias_especificas(course_level: str | None, subject: str | None) -> list[CompetenciaEspecifica]:
    """Devuelve las competencias específicas."""
    cur = get_curriculum(course_level, subject)
    return cur["competencias_especificas"] if cur else []


def get_cc_codes_from_criterio(code: str, course_level: str | None, subject: str | None) -> list[str]:
    """Devuelve los códigos de competencias clave (sin número) vinculados a un criterio.

    Ejemplo: criterio "1.1" tiene descriptores ["CCL2", "STEM1", "STEM2", ...]
    → devuelve ["CCL", "STEM", ...] (sin duplicados, ordenados).
    """
    criterios = get_criterios(course_level, subject)
    for crit in criterios:
        if crit.code == code:
            cc_set: set[str] = set()
            for desc in crit.descriptores:
                # Extraer código base: "STEM1" → "STEM", "CPSAA1.1" → "CPSAA", "CCEC4.2" → "CCEC"
                base = ""
                for ch in desc:
                    if ch.isalpha():
                        base += ch
                    else:
                        break
                if base:
                    cc_set.add(base)
            return sorted(cc_set)
    return []


def get_criterios_for_cc(cc_code: str, course_level: str | None, subject: str | None) -> list[str]:
    """Devuelve los códigos de criterios vinculados a una competencia clave.

    Ejemplo: cc_code="STEM" → ["1.1", "1.2", "2.1", "2.2", ...] (todos los que tienen algún descriptor STEM*).
    """
    criterios = get_criterios(course_level, subject)
    result = []
    for crit in criterios:
        for desc in crit.descriptores:
            base = ""
            for ch in desc:
                if ch.isalpha():
                    base += ch
                else:
                    break
            if base == cc_code:
                result.append(crit.code)
                break
    return result


def has_curriculum(course_level: str | None, subject: str | None) -> bool:
    """Comprueba si hay datos curriculares para un curso y asignatura."""
    return get_curriculum(course_level, subject) is not None
