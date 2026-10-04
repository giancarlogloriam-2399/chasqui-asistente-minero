# Chasqui — Fase 2: Motor RAG (prototipo)

Esta carpeta contiene el motor de preguntas y respuestas de Chasqui: lee los
documentos de `01_Legal_y_Normativo`, `02_Tecnico_Procesos` y
`03_ Formalizacion minera`, y permite hacerles preguntas en lenguaje natural.

## Requisitos previos

1. **Python 3.10 o superior** instalado en tu computadora.
2. **Una API key de Google Gemini** (gratis para probar):
   - Entra a https://aistudio.google.com/apikey
   - Crea una API key.
   - **Importante (control de gasto)**: en Google AI Studio, configura un
     límite de gasto (billing cap) antes de usar la app en producción, tal
     como lo conversamos — esto evita sorpresas si algo falla y la API se
     usa de más.

## Instalación (una sola vez)

Abre una terminal (CMD o PowerShell) dentro de esta carpeta `App_Chasqui` y
ejecuta:

```
pip install -r requirements.txt
```

## Configurar tu API key

**Windows (CMD):**
```
set GEMINI_API_KEY=tu_clave_aqui
```

**Windows (PowerShell):**
```
$env:GEMINI_API_KEY="tu_clave_aqui"
```

Esto solo dura mientras esa ventana de terminal esté abierta. Si cierras la
terminal, hay que volver a escribirlo (más adelante, cuando pasemos a
producción, esto se reemplaza por un archivo de configuración seguro).

## Paso 1 — Construir la base de conocimiento (ingest.py)

Esto lee todos los PDF, los convierte en fragmentos, genera sus embeddings
y arma la base de datos vectorial. **Se corre una sola vez**, y de nuevo
cada vez que agregues o cambies documentos.

```
python ingest.py
```

Vas a ver en pantalla el progreso archivo por archivo. Al final te dice
cuántos fragmentos se guardaron. Esto puede tardar varios minutos
dependiendo de cuántos PDF tengas (recuerda que hay una pequeña pausa entre
cada llamada a la API para no exceder la cuota gratuita).

Si un PDF está escaneado (como `casos de formalizacion minera.pdf`, que no
tiene texto seleccionable), el script lo va a saltar automáticamente y te
avisa — ese archivo necesitaría pasar primero por OCR, que no está incluido
en este prototipo.

## Paso 2 — Levantar la aplicación (app.py)

```
streamlit run app.py
```

Esto abre una ventana en tu navegador (normalmente en `http://localhost:8501`)
donde puedes escribir preguntas y ver las respuestas, junto con los
fragmentos de documento que se usaron para generarlas.

## Cómo funciona por dentro (resumen)

```
Tu pregunta
    │
    ▼
Se convierte en un "embedding" (un vector numérico que representa su significado)
    │
    ▼
Se busca en ChromaDB los 5 fragmentos de documentos más parecidos a esa pregunta
    │
    ▼
Esos fragmentos + tu pregunta se le pasan a Gemini como "contexto"
    │
    ▼
Gemini responde SOLO con base en esos fragmentos (no inventa, no usa internet)
    │
    ▼
Se te muestra la respuesta + de qué documento salió cada dato
```

## Qué falta para pasar de prototipo a producción (próximos pasos, Fase 3-5)

- [ ] Modo 2: que el minero pueda subir su propio documento para consultarlo
      (sesión temporal, sin mezclarlo con la base curada).
- [ ] Guardar el feedback de los usuarios en Supabase.
- [ ] Desplegar la app en Streamlit Community Cloud para compartir el link.
- [ ] Mover la API key a un archivo de configuración seguro (no en texto
      plano en la terminal).
- [ ] Resolver el OCR para los PDF escaneados (como el de "casos de
      formalización minera").
- [ ] Ampliar la base de conocimiento cuando haya financiamiento — el
      proceso es el mismo: agregar PDFs a las carpetas y volver a correr
      `ingest.py`.

## Archivos de esta carpeta

| Archivo | Qué hace |
|---|---|
| `ingest.py` | Lee los documentos y construye la base de datos vectorial (`chroma_db/`) |
| `app.py` | La aplicación web de preguntas y respuestas |
| `requirements.txt` | Lista de librerías necesarias |
| `chroma_db/` | Se crea automáticamente al correr `ingest.py` — es la base de datos, no se edita a mano |
| `contador_uso.json` | Se crea automáticamente — lleva la cuenta de preguntas hechas por día (control de gasto) |
