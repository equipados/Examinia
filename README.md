# CorrectorExamenes (Matemáticas II)

Aplicación en Python para corregir exámenes escaneados manuscritos de Matemáticas II (2º Bachillerato) usando **Google Gemini Flash** para extracción/interpretación, con lógica final de puntuación implementada en código.

## Características

- Procesa todos los PDFs de `examenes/`.
- Convierte PDF a imágenes y aplica preprocesado opcional.
- Usa Gemini Flash para:
  - lectura OCR de manuscrito,
  - interpretación de notación matemática,
  - estructuración por ejercicios/apartados,
  - resolución del enunciado (cuando sea legible, recomendado con modelo Pro),
  - evaluación asistida del procedimiento.
- Segmenta exámenes cuando hay varios en un mismo PDF.
  - Regla fuerte: inicio de examen detectado por encabezado (logo izquierda + nombre).
- Aplica reglas de negocio:
  - reparto equitativo de puntos por apartados cuando solo hay puntuación total,
  - IA puede resolver el ejercicio desde el enunciado para construir/ayudar la referencia de corrección,
  - puntuación parcial basada en procedimiento y errores típicos,
  - marca `revision_manual` ante baja confianza o ambigüedad.
- Genera:
  - `salidas/resultados.xlsx` (resumen global),
  - `salidas/informes/*.md` (informe individual por alumno).

## Requisitos

- Python 3.11+
- Dependencias de `requirements.txt`
- `GEMINI_API_KEY` válida
- `poppler` instalado (necesario para `pdf2image`)

## Instalación

1. Crear y activar entorno virtual.
2. Instalar dependencias:

```bash
pip install -r requirements.txt
```

3. Configurar API key en entorno o `.env`:

```bash
GEMINI_API_KEY=tu_clave
```

## Estructura del proyecto

```text
main.py
config.py
models.py
pdf_processor.py
image_preprocessing.py
gemini_client.py
exam_segmenter.py
exam_parser.py
grading.py
reporting.py
excel_export.py
utils.py

tests/
examenes/
soluciones/
salidas/
salidas/informes/
```

## Formato esperado de `soluciones/`

Puedes tener uno o varios `.json`. Cada archivo admite:

- un único objeto de solución,
- una lista de objetos,
- o `{ "solutions": [ ... ] }`.

Ejemplo de item:

```json
{
  "exercise": "1",
  "part": "a",
  "topic": "sistemas de ecuaciones",
  "expected_final_answer": "x=2, y=-1",
  "accepted_equivalents": ["y=-1, x=2", "(2,-1)"],
  "expected_steps": [
    "plantea el sistema correctamente",
    "elimina o despeja una variable",
    "obtiene la solución"
  ],
  "common_errors": ["error de signo", "error aritmético final"],
  "max_points": 0.5,
  "partial_credit_rules": [
    {
      "condition": "procedimiento correcto pero error aritmético final",
      "points": 0.25,
      "explanation": "Planteamiento correcto con error de cálculo en el paso final."
    }
  ]
}
```

## Ejecución

Ejecución básica:

```bash
python main.py
```

Ejemplo con parámetros:

```bash
python main.py \
  --input-dir examenes \
  --solutions-dir soluciones \
  --output-dir salidas \
  --gemini-model gemini-2.5-flash \
  --gemini-solver-model gemini-2.5-pro \
  --poppler-path "C:\\tools\\poppler\\Library\\bin" \
  --ai-solver-min-confidence 0.75 \
  --header-confidence-threshold 0.80 \
  --log-level INFO \
  --strict-mode \
  --page-batch-limit 10 \
  --dpi 240
```

Si no tienes Poppler en `PATH`, usa siempre `--poppler-path` apuntando a la carpeta con `pdftoppm.exe` y `pdfinfo.exe`.
Si quieres desactivar la resolución previa del enunciado por IA, usa `--disable-ai-solver`.

## Salidas

- Excel global: `salidas/resultados.xlsx`
- Informes: `salidas/informes/*.md`

El Excel incluye columnas dinámicas por apartado (`1.a`, `1.b`, etc.), total, incidencias e informe.

## Decisiones de robustez (conservadoras)

- Toda salida estructurada de Gemini se valida con Pydantic.
- Si Gemini devuelve JSON inválido, se reintenta automáticamente con prompt más restrictivo.
- Si la confianza es baja o hay ambigüedad, se marca `revision_manual`.
- No se otorgan puntos arbitrarios cuando falta plantilla de solución.

## Casos que requieren revisión manual

- OCR manuscrito ilegible.
- Nombre de alumno con baja confianza.
- Segmentación dudosa de varios exámenes en un PDF.
- Puntuación máxima no deducible.
- Falta de plantilla para un apartado.

## Tests

```bash
pytest -q
```

Incluyen:

- reparto equitativo de puntos,
- cálculo de totales,
- validación de modelos,
- generación de informe Markdown,
- exportación Excel,
- saneado de nombres de archivo,
- reglas de puntuación parcial.

## Limitaciones actuales

- La calidad OCR depende del escaneo y la caligrafía.
- La equivalencia simbólica cubre casos simples; no todos los formatos algebraicos complejos.
- La segmentación de múltiples exámenes en un mismo PDF es heurística asistida por Gemini.

## Mejoras futuras

- Cache de respuestas Gemini para re-ejecuciones.
- UI de revisión manual.
- Ajuste de rúbricas por centro/profesor.
- Exportación adicional a PDF de informes.
