# -*- coding: utf-8 -*-
"""
APP.PY - Fase 2 de Chasqui
-------------------------------------------------------------
Esta es la aplicación web (Streamlit) donde un minero chatea con
Chasqui y recibe respuestas basadas en los documentos que cargamos
con ingest.py (no en lo que el modelo "ya sabe" de internet) - salvo
que explícitamente le avisemos que la respuesta vino de una búsqueda
en internet porque no había nada en los documentos.

Flujo de cada pregunta:
  1. El minero escribe su pregunta (o toca una pregunta frecuente).
  2. Si ya venía conversando, reformulamos la pregunta para que sea
     independiente del historial (para que la búsqueda entienda
     referencias tipo "¿y cuánto tarda eso?").
  3. Se buscan en ChromaDB los fragmentos de documentos más parecidos.
  4. Si no hay fragmentos relevantes, o el modelo dice que no tiene
     información suficiente, se hace una segunda consulta a Gemini
     con búsqueda en Google activada, y se etiqueta como tal.
  5. Se muestra la respuesta + de qué documento (y página) salió cada
     dato, con un enlace para abrir el PDF original en esa página.

Cómo correrlo (después de haber corrido ingest.py al menos una vez):
    streamlit run app.py
"""

import os
import json
import datetime
from pathlib import Path

import streamlit as st
import chromadb
from google import genai
from google.genai import types

# ------------------------------------------------------------------
# CONFIGURACIÓN
# ------------------------------------------------------------------

CARPETA_DE_ESTE_SCRIPT = os.path.dirname(os.path.abspath(__file__))
CARPETA_BASE_DATOS = os.path.join(CARPETA_DE_ESTE_SCRIPT, "chroma_db")
NOMBRE_COLECCION = "chasqui_conocimiento"

MODELO_EMBEDDING = "gemini-embedding-001"
MODELO_RESPUESTA = "gemini-3.1-flash-lite"  # verifica el nombre vigente en ai.google.dev

CANTIDAD_FRAGMENTOS_A_BUSCAR = 5

# --- Control de gasto ---
# Límite global (para todos los que usen la app) además del límite de
# gasto que configures en Google AI Studio (billing cap), y un límite
# por sesión para que una sola persona no agote la cuota del día.
LIMITE_PREGUNTAS_POR_DIA = 200
LIMITE_PREGUNTAS_POR_SESION = 30
ARCHIVO_CONTADOR = os.path.join(CARPETA_DE_ESTE_SCRIPT, "contador_uso.json")
ARCHIVO_RETROALIMENTACION = os.path.join(CARPETA_DE_ESTE_SCRIPT, "retroalimentacion.jsonl")

FRASE_SIN_INFO = "no tengo información suficiente"

CATEGORIAS_POR_CARPETA = [
    ("01_Legal_y_Normativo", "Legal"),
    ("02_Tecnico_Procesos", "Técnico"),
    ("03_ Formalizacion minera", "Formalización"),
]

PREGUNTAS_FRECUENTES = [
    ("📝 REINFO", "¿Cómo me inscribo en el REINFO?"),
    ("⚖️ Formalización", "¿Qué requisitos necesito para la formalización minera?"),
    ("⚗️ Procesos", "¿Qué es la flotación en el proceso minero?"),
    ("☣️ Seguridad", "¿Cómo debo manejar el cianuro de forma segura?"),
]

INSTRUCCION_DEL_SISTEMA = f"""
Eres Chasqui, un asistente que ayuda a pequeños mineros y mineros
artesanales del Perú con preguntas legales (formalización, REINFO) y
técnicas (procesos metalúrgicos, seguridad, medio ambiente).

Reglas que debes seguir siempre:
1. Responde ÚNICAMENTE con información que aparezca en el CONTEXTO que
   se te entrega abajo. No uses conocimiento externo ni inventes datos.
2. Si el contexto no tiene información suficiente para responder,
   dilo EXACTAMENTE así: "{FRASE_SIN_INFO.capitalize()} sobre esto en
   mis documentos" - no inventes una respuesta.
3. Usa un lenguaje simple y directo, pensado para alguien sin formación
   técnica avanzada.
4. Al final de tu respuesta, menciona de qué documento(s) sacaste la
   información (te lo indico en el contexto).
5. Si la pregunta es sobre un trámite legal específico (ej. REINFO),
   aclara siempre que esto es información general y que debe
   confirmarse con la DREM/GREM de su región o con el MINEM, porque
   las normas cambian con el tiempo.
6. Si hay HISTORIAL DE LA CONVERSACIÓN, úsalo para entender a qué se
   refiere el minero si su pregunta depende de algo que preguntó antes.
"""


# ------------------------------------------------------------------
# CONTROL DE GASTO - contadores de preguntas
# ------------------------------------------------------------------

def leer_contador_de_hoy():
    hoy = str(datetime.date.today())
    if os.path.exists(ARCHIVO_CONTADOR):
        with open(ARCHIVO_CONTADOR, "r") as archivo:
            datos = json.load(archivo)
    else:
        datos = {}
    return datos.get(hoy, 0), hoy


def sumar_una_pregunta(hoy):
    if os.path.exists(ARCHIVO_CONTADOR):
        with open(ARCHIVO_CONTADOR, "r") as archivo:
            datos = json.load(archivo)
    else:
        datos = {}
    datos[hoy] = datos.get(hoy, 0) + 1
    with open(ARCHIVO_CONTADOR, "w") as archivo:
        json.dump(datos, archivo)


def registrar_feedback(pregunta, respuesta, valor):
    entrada = {
        "fecha": str(datetime.datetime.now()),
        "pregunta": pregunta,
        "respuesta": respuesta[:500],
        "feedback": valor,
    }
    try:
        with open(ARCHIVO_RETROALIMENTACION, "a", encoding="utf-8") as archivo:
            archivo.write(json.dumps(entrada, ensure_ascii=False) + "\n")
    except Exception:
        pass


# ------------------------------------------------------------------
# BÚSQUEDA Y RESPUESTA
# ------------------------------------------------------------------

def categoria_de_ruta(ruta):
    ruta_norm = ruta.replace("\\", "/")
    for carpeta, categoria in CATEGORIAS_POR_CARPETA:
        if carpeta in ruta_norm:
            return categoria
    return "Otro"


def enlace_pdf_local(ruta, pagina):
    """Enlace file:// a la página exacta del PDF local (abre con el lector por defecto)."""
    try:
        return f"{Path(ruta).resolve().as_uri()}#page={pagina}"
    except Exception:
        return None


def texto_de_historial(mensajes, limite_intercambios=6):
    recientes = mensajes[-limite_intercambios:]
    etiqueta = {"user": "Minero", "assistant": "Chasqui"}
    return "\n".join(f"{etiqueta.get(m['role'], m['role'])}: {m['content']}" for m in recientes)


def reformular_pregunta_independiente(cliente_gemini, historial_previo, pregunta_nueva):
    """Convierte una pregunta de seguimiento (ej. '¿y cuánto tarda?') en una
    pregunta completa e independiente, para que la búsqueda vectorial la
    entienda sin depender del historial."""
    if not historial_previo:
        return pregunta_nueva

    prompt = (
        f"Historial de conversación:\n{texto_de_historial(historial_previo)}\n\n"
        f"Nueva pregunta del usuario: {pregunta_nueva}\n\n"
        "Reescribe la nueva pregunta como una pregunta independiente y completa, "
        "que se entienda sin necesidad de leer el historial. Si ya es independiente, "
        "repítela igual. Responde ÚNICAMENTE con la pregunta reescrita, sin comillas "
        "ni explicaciones."
    )
    try:
        resultado = cliente_gemini.models.generate_content(
            model=MODELO_RESPUESTA,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0),
        )
        return (resultado.text or "").strip() or pregunta_nueva
    except Exception:
        return pregunta_nueva


def buscar_fragmentos_relevantes(cliente_gemini, coleccion, pregunta_busqueda, categorias_filtro=None):
    resultado_embedding = cliente_gemini.models.embed_content(
        model=MODELO_EMBEDDING,
        contents=pregunta_busqueda,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    embedding_pregunta = resultado_embedding.embeddings[0].values

    # el filtro de categoría se calcula a partir de la carpeta del archivo,
    # que no está guardada como campo de metadato propio en ChromaDB, así
    # que pedimos de más y filtramos en Python antes de recortar al tamaño final
    hay_filtro = categorias_filtro is not None and len(categorias_filtro) < len(CATEGORIAS_POR_CARPETA)
    n_a_pedir = CANTIDAD_FRAGMENTOS_A_BUSCAR * 4 if hay_filtro else CANTIDAD_FRAGMENTOS_A_BUSCAR

    resultados = coleccion.query(
        query_embeddings=[embedding_pregunta],
        n_results=n_a_pedir,
    )
    documentos = resultados["documents"][0]
    metadatos = resultados["metadatas"][0]

    if hay_filtro:
        pares_filtrados = [
            (doc, meta) for doc, meta in zip(documentos, metadatos)
            if categoria_de_ruta(meta["ruta"]) in categorias_filtro
        ][:CANTIDAD_FRAGMENTOS_A_BUSCAR]
        documentos = [doc for doc, _ in pares_filtrados]
        metadatos = [meta for _, meta in pares_filtrados]

    return documentos, metadatos


def armar_contexto(documentos, metadatos):
    partes_de_contexto = []
    for texto_fragmento, metadato in zip(documentos, metadatos):
        encabezado = f"[Fuente: {metadato['archivo']}, página {metadato['pagina']}]"
        partes_de_contexto.append(f"{encabezado}\n{texto_fragmento}")
    return "\n\n---\n\n".join(partes_de_contexto)


def preguntar_a_gemini(cliente_gemini, historial_previo, pregunta, contexto):
    historial_texto = texto_de_historial(historial_previo)
    mensaje_completo = (
        (f"HISTORIAL DE LA CONVERSACIÓN:\n{historial_texto}\n\n" if historial_texto else "")
        + f"CONTEXTO:\n{contexto}\n\n"
        + f"PREGUNTA ACTUAL DEL MINERO:\n{pregunta}"
    )
    respuesta = cliente_gemini.models.generate_content(
        model=MODELO_RESPUESTA,
        contents=mensaje_completo,
        config=types.GenerateContentConfig(
            system_instruction=INSTRUCCION_DEL_SISTEMA,
            temperature=0.2,
        ),
    )
    return respuesta.text


def buscar_en_internet(cliente_gemini, pregunta):
    """Respaldo cuando los documentos no tienen la respuesta: busca en
    internet con Gemini y deja muy claro que no viene de los documentos
    oficiales verificados."""
    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    prompt = (
        "Eres un asistente que ayuda a pequeños mineros y mineros artesanales "
        "del Perú. No encontré información sobre esto en los documentos internos "
        "verificados. Busca en internet y responde de forma breve, simple y en "
        "español. Al final, recuerda que esto es información general de internet "
        "(no verificada contra la normativa peruana oficial) y que debe "
        "confirmarse con el MINEM o la DREM/GREM de su región.\n\n"
        f"Pregunta: {pregunta}"
    )
    try:
        respuesta = cliente_gemini.models.generate_content(
            model=MODELO_RESPUESTA,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[grounding_tool],
                temperature=0.2,
            ),
        )
        return respuesta.text
    except Exception as error:
        return f"No pude buscar en internet en este momento ({error}). Intenta de nuevo más tarde."


def generar_respuesta(cliente_gemini, coleccion, historial_previo, pregunta, categorias_filtro=None):
    """Devuelve (texto_respuesta, fuentes_o_None, desde_internet)."""
    try:
        pregunta_busqueda = reformular_pregunta_independiente(cliente_gemini, historial_previo, pregunta)
        documentos, metadatos = buscar_fragmentos_relevantes(
            cliente_gemini, coleccion, pregunta_busqueda, categorias_filtro
        )
    except Exception:
        return (
            "No pude conectarme con el servicio de búsqueda en este momento. "
            "Intenta de nuevo en unos segundos.",
            None,
            False,
        )

    if len(documentos) == 0:
        return buscar_en_internet(cliente_gemini, pregunta), None, True

    contexto = armar_contexto(documentos, metadatos)
    try:
        respuesta = preguntar_a_gemini(cliente_gemini, historial_previo, pregunta, contexto)
    except Exception:
        return (
            "No pude generar una respuesta en este momento. Intenta de nuevo en unos segundos.",
            None,
            False,
        )

    if FRASE_SIN_INFO in respuesta.lower():
        return buscar_en_internet(cliente_gemini, pregunta), None, True

    return respuesta, list(zip(documentos, metadatos)), False


# ------------------------------------------------------------------
# INTERFAZ (Streamlit)
# ------------------------------------------------------------------

def render_encabezado():
    st.markdown(
        """
        <style>
        .chasqui-banner {
            background: linear-gradient(135deg, #8B5A2B 0%, #C1662F 55%, #D4942C 100%);
            padding: 1.4rem 1.6rem;
            border-radius: 14px;
            color: #FFF8EC;
            margin-bottom: 1.3rem;
        }
        .chasqui-banner h1 { margin: 0; font-size: 1.7rem; }
        .chasqui-banner p { margin: 0.35rem 0 0 0; opacity: 0.95; font-size: 0.95rem; }
        </style>
        <div class="chasqui-banner">
            <h1>⛏️ Chasqui — Asistente para el Pequeño Minero</h1>
            <p>Consulta legal (REINFO, formalización) y técnica, basada en documentos oficiales verificados.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def mostrar_fuentes(fuentes):
    st.markdown("**📚 Fuentes:**")
    for _, metadato in fuentes:
        enlace = enlace_pdf_local(metadato["ruta"], metadato["pagina"])
        etiqueta = f"{metadato['archivo']} — página {metadato['pagina']}"
        if enlace:
            st.markdown(f"- 📄 [{etiqueta}]({enlace})")
        else:
            st.markdown(f"- 📄 {etiqueta}")


def construir_texto_exportable(indice):
    mensajes = st.session_state.messages
    mensaje = mensajes[indice]
    pregunta = mensajes[indice - 1]["content"] if indice > 0 else ""
    lineas = [f"Pregunta: {pregunta}", "", f"Respuesta: {mensaje['content']}"]
    if mensaje.get("desde_internet"):
        lineas.append("\n(Respuesta de internet, no de los documentos oficiales)")
    if mensaje.get("fuentes"):
        lineas.append("\nFuentes:")
        for _, metadato in mensaje["fuentes"]:
            lineas.append(f"- {metadato['archivo']}, página {metadato['pagina']}")
    return "\n".join(lineas)


def render_acciones_mensaje(indice):
    mensaje = st.session_state.messages[indice]
    col1, col2, col3 = st.columns([1, 1, 4])
    ya_opino = mensaje.get("feedback") is not None
    with col1:
        if st.button("👍", key=f"up_{indice}", disabled=ya_opino):
            mensaje["feedback"] = "👍"
            registrar_feedback(
                st.session_state.messages[indice - 1]["content"] if indice > 0 else "",
                mensaje["content"],
                "👍",
            )
    with col2:
        if st.button("👎", key=f"down_{indice}", disabled=ya_opino):
            mensaje["feedback"] = "👎"
            registrar_feedback(
                st.session_state.messages[indice - 1]["content"] if indice > 0 else "",
                mensaje["content"],
                "👎",
            )
    with col3:
        st.download_button(
            "📋 Descargar esta respuesta",
            data=construir_texto_exportable(indice),
            file_name=f"chasqui_respuesta_{indice}.txt",
            mime="text/plain",
            key=f"export_{indice}",
        )
    if ya_opino:
        st.caption(f"Gracias por tu opinión: {mensaje['feedback']}")


def main():
    st.set_page_config(page_title="Chasqui - Asistente Minero", page_icon="⛏️")
    render_encabezado()

    api_key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        st.error("Falta configurar la API key de Gemini (variable de entorno GEMINI_API_KEY o secreto de Streamlit).")
        st.stop()

    if not os.path.isdir(CARPETA_BASE_DATOS):
        st.error("Todavía no existe la base de conocimiento. Corre primero: python ingest.py")
        st.stop()

    cliente_gemini = genai.Client(api_key=api_key)
    cliente_chroma = chromadb.PersistentClient(path=CARPETA_BASE_DATOS)
    try:
        coleccion = cliente_chroma.get_collection(NOMBRE_COLECCION)
    except Exception:
        st.error("No pude abrir la base de conocimiento. Corre de nuevo: python ingest.py")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "contador_sesion" not in st.session_state:
        st.session_state.contador_sesion = 0

    preguntas_hechas_hoy, fecha_hoy = leer_contador_de_hoy()
    with st.sidebar:
        if st.button("🔄 Nueva conversación"):
            st.session_state.messages = []
            st.rerun()

        st.markdown("**Buscar en:**")
        categorias_filtro = [
            categoria for _, categoria in CATEGORIAS_POR_CARPETA
            if st.checkbox(categoria, value=True, key=f"cat_{categoria}")
        ]
        if not categorias_filtro:
            st.caption("Selecciona al menos una categoría — buscando en todas por ahora.")
            categorias_filtro = [categoria for _, categoria in CATEGORIAS_POR_CARPETA]

        st.divider()
        st.metric("Preguntas usadas hoy (todos)", f"{preguntas_hechas_hoy} / {LIMITE_PREGUNTAS_POR_DIA}")
        st.metric("Tus preguntas en esta sesión", f"{st.session_state.contador_sesion} / {LIMITE_PREGUNTAS_POR_SESION}")

    limite_alcanzado = (
        preguntas_hechas_hoy >= LIMITE_PREGUNTAS_POR_DIA
        or st.session_state.contador_sesion >= LIMITE_PREGUNTAS_POR_SESION
    )
    if limite_alcanzado:
        st.warning("Se alcanzó el límite de preguntas. Intenta de nuevo más tarde.")

    for indice, mensaje in enumerate(st.session_state.messages):
        with st.chat_message(mensaje["role"]):
            if mensaje["role"] == "assistant" and mensaje.get("desde_internet"):
                st.info("🌐 No encontré esto en mis documentos oficiales. Es información general de internet — verifícala con tu DREM/GREM o el MINEM.")
            st.write(mensaje["content"])
            if mensaje.get("fuentes"):
                mostrar_fuentes(mensaje["fuentes"])
            if mensaje["role"] == "assistant":
                render_acciones_mensaje(indice)

    pregunta_faq = None
    if not limite_alcanzado:
        st.caption("Preguntas frecuentes:")
        columnas = st.columns(len(PREGUNTAS_FRECUENTES))
        for columna, (etiqueta, texto_pregunta) in zip(columnas, PREGUNTAS_FRECUENTES):
            if columna.button(etiqueta, key=f"faq_{etiqueta}"):
                pregunta_faq = texto_pregunta

    pregunta_chat = st.chat_input("Escribe tu pregunta (ej: ¿Cómo me inscribo en el REINFO?)", disabled=limite_alcanzado)

    pregunta_nueva = pregunta_faq or pregunta_chat
    if pregunta_nueva:
        historial_previo = list(st.session_state.messages)
        st.session_state.messages.append({"role": "user", "content": pregunta_nueva})

        with st.spinner("Chasqui está revisando los documentos..."):
            respuesta, fuentes, desde_internet = generar_respuesta(
                cliente_gemini, coleccion, historial_previo, pregunta_nueva, categorias_filtro
            )

        st.session_state.messages.append({
            "role": "assistant",
            "content": respuesta,
            "fuentes": fuentes,
            "desde_internet": desde_internet,
            "feedback": None,
        })
        sumar_una_pregunta(fecha_hoy)
        st.session_state.contador_sesion += 1
        st.rerun()


if __name__ == "__main__":
    main()
