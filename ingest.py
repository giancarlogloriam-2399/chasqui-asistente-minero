# -*- coding: utf-8 -*-
"""
INGEST.PY - Fase 2 de Chasqui
-------------------------------------------------------------
Este script hace 4 cosas, en orden:
  1. Recorre las carpetas de documentos (legal y técnico).
  2. Extrae el texto de cada PDF (y de cada .md de resumen).
  3. Parte ese texto en fragmentos pequeños ("chunks").
  4. Genera un embedding de cada fragmento con la API de Gemini
     y lo guarda en una base de datos vectorial local (ChromaDB).

Se corre UNA VEZ cada vez que agregues o cambies documentos.
La app (app.py) después solo LEE esta base, no la vuelve a construir.

Cómo correrlo:
    1. Instala las librerías:  pip install -r requirements.txt
    2. Define tu API key:      set GEMINI_API_KEY=tu_clave_aqui   (Windows)
                                export GEMINI_API_KEY=tu_clave_aqui (Mac/Linux)
    3. Ejecuta:                python ingest.py
"""

import os
import time
import pypdf
import chromadb
from google import genai
from google.genai import types

# ------------------------------------------------------------------
# CONFIGURACIÓN - ajusta estos valores a tu gusto
# ------------------------------------------------------------------

# Carpeta donde vive este script (para que funcione sin importar desde
# dónde lo ejecutes - no depende del directorio de trabajo actual)
CARPETA_DE_ESTE_SCRIPT = os.path.dirname(os.path.abspath(__file__))

# Carpetas donde están los documentos, relativas a la carpeta del proyecto
# (un nivel arriba de App_Chasqui). Ajusta los nombres si cambias la
# estructura de carpetas.
CARPETAS_DOCUMENTOS = [
    os.path.join(CARPETA_DE_ESTE_SCRIPT, "..", "01_Legal_y_Normativo"),
    os.path.join(CARPETA_DE_ESTE_SCRIPT, "..", "02_Tecnico_Procesos"),
    os.path.join(CARPETA_DE_ESTE_SCRIPT, "..", "03_ Formalizacion minera"),
]

# Dónde se guarda la base de datos vectorial (se crea sola)
CARPETA_BASE_DATOS = os.path.join(CARPETA_DE_ESTE_SCRIPT, "chroma_db")
NOMBRE_COLECCION = "chasqui_conocimiento"

# Tamaño de cada fragmento de texto, en palabras (no en caracteres)
PALABRAS_POR_FRAGMENTO = 350
PALABRAS_DE_SOLAPE = 50  # para no cortar una idea justo en la mitad

# Modelo de embeddings de Gemini (verifica el nombre vigente en
# https://ai.google.dev/gemini-api/docs/embeddings antes de correr,
# Google a veces cambia estos nombres)
MODELO_EMBEDDING = "gemini-embedding-001"

# Pausa entre llamadas a la API para no pasarte de la cuota gratuita
PAUSA_ENTRE_LLAMADAS_SEGUNDOS = 1.0


# ------------------------------------------------------------------
# PASO 1: Encontrar todos los archivos a procesar
# ------------------------------------------------------------------

def encontrar_archivos():
    """Busca todos los PDF y MD dentro de las carpetas configuradas."""
    archivos_encontrados = []
    for carpeta in CARPETAS_DOCUMENTOS:
        if not os.path.isdir(carpeta):
            print(f"Aviso: no existe la carpeta {carpeta}, se salta.")
            continue
        for carpeta_actual, subcarpetas, archivos in os.walk(carpeta):
            for nombre_archivo in archivos:
                extension = nombre_archivo.lower().split(".")[-1]
                if extension in ("pdf", "md"):
                    ruta_completa = os.path.join(carpeta_actual, nombre_archivo)
                    archivos_encontrados.append(ruta_completa)
    return archivos_encontrados


# ------------------------------------------------------------------
# PASO 2: Extraer el texto de cada archivo
# ------------------------------------------------------------------

def extraer_texto_de_pdf(ruta_pdf):
    """Devuelve una lista de (numero_de_pagina, texto_de_esa_pagina)."""
    paginas_con_texto = []
    try:
        lector = pypdf.PdfReader(ruta_pdf)
        for numero_pagina, pagina in enumerate(lector.pages, start=1):
            texto = pagina.extract_text() or ""
            texto = texto.strip()
            if len(texto) > 20:  # ignora páginas casi vacías
                paginas_con_texto.append((numero_pagina, texto))
    except Exception as error:
        print(f"  No se pudo leer {ruta_pdf}: {error}")
    return paginas_con_texto


def extraer_texto_de_md(ruta_md):
    """Devuelve el texto completo de un archivo markdown como una sola 'página'."""
    with open(ruta_md, "r", encoding="utf-8") as archivo:
        texto = archivo.read()
    return [(1, texto)]


# ------------------------------------------------------------------
# PASO 3: Partir el texto en fragmentos (chunking)
# ------------------------------------------------------------------

def partir_en_fragmentos(texto):
    """Divide un texto largo en fragmentos de PALABRAS_POR_FRAGMENTO palabras,
    con un pequeño solape entre fragmentos para no perder contexto."""
    palabras = texto.split()
    fragmentos = []
    inicio = 0
    while inicio < len(palabras):
        fin = inicio + PALABRAS_POR_FRAGMENTO
        fragmento = " ".join(palabras[inicio:fin])
        fragmentos.append(fragmento)
        inicio = fin - PALABRAS_DE_SOLAPE
        if inicio < 0:
            inicio = 0
        if fin >= len(palabras):
            break
    return fragmentos


# ------------------------------------------------------------------
# PASO 4: Generar el embedding de un fragmento con Gemini
# ------------------------------------------------------------------

def generar_embedding(cliente_gemini, texto_fragmento):
    resultado = cliente_gemini.models.embed_content(
        model=MODELO_EMBEDDING,
        contents=texto_fragmento,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )
    return resultado.embeddings[0].values


# ------------------------------------------------------------------
# PROGRAMA PRINCIPAL
# ------------------------------------------------------------------

class CuotaAgotada(Exception):
    """Se lanza cuando la API de Gemini devuelve que se acabó la cuota del día."""
    pass


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: no encontré la variable de entorno GEMINI_API_KEY.")
        print("Defínela antes de correr este script (ver instrucciones arriba).")
        return
    cliente_gemini = genai.Client(api_key=api_key)

    cliente_chroma = chromadb.PersistentClient(path=CARPETA_BASE_DATOS)
    # reutilizamos la colección si ya existe, para poder reanudar sin repetir
    # fragmentos que ya se guardaron en una corrida anterior (p.ej. si se
    # cortó por la cuota diaria de la API)
    coleccion = cliente_chroma.get_or_create_collection(NOMBRE_COLECCION)
    ids_ya_guardados = set(coleccion.get(include=[])["ids"])
    if ids_ya_guardados:
        print(f"Encontré {len(ids_ya_guardados)} fragmentos ya guardados en una corrida anterior; se omiten.\n")

    archivos = encontrar_archivos()
    print(f"Encontré {len(archivos)} archivos para procesar.\n")

    total_fragmentos_guardados = 0
    total_fragmentos_omitidos = 0

    try:
        for ruta_archivo in archivos:
            nombre_archivo = os.path.basename(ruta_archivo)
            print(f"Procesando: {nombre_archivo}")

            if ruta_archivo.lower().endswith(".pdf"):
                paginas = extraer_texto_de_pdf(ruta_archivo)
            else:
                paginas = extraer_texto_de_md(ruta_archivo)

            if len(paginas) == 0:
                print("  (sin texto extraíble - probablemente es un PDF escaneado, se omite)")
                continue

            for numero_pagina, texto_pagina in paginas:
                fragmentos = partir_en_fragmentos(texto_pagina)
                for indice_fragmento, texto_fragmento in enumerate(fragmentos):
                    id_unico = f"{nombre_archivo}__p{numero_pagina}__f{indice_fragmento}"

                    if id_unico in ids_ya_guardados:
                        total_fragmentos_omitidos += 1
                        continue

                    try:
                        embedding = generar_embedding(cliente_gemini, texto_fragmento)
                    except Exception as error:
                        if "RESOURCE_EXHAUSTED" in str(error):
                            raise CuotaAgotada(str(error))
                        print(f"  Error generando embedding, se salta un fragmento: {error}")
                        continue

                    coleccion.add(
                        ids=[id_unico],
                        embeddings=[embedding],
                        documents=[texto_fragmento],
                        metadatas=[{
                            "archivo": nombre_archivo,
                            "ruta": ruta_archivo,
                            "pagina": numero_pagina,
                        }],
                    )
                    total_fragmentos_guardados += 1
                    time.sleep(PAUSA_ENTRE_LLAMADAS_SEGUNDOS)
    except CuotaAgotada as error:
        print(f"\nSe acabó la cuota diaria gratuita de la API de Gemini: {error}")
        print("Lo que ya se guardó queda en la base de datos. Vuelve a correr este")
        print("mismo script más tarde (cuando se reinicie la cuota) y va a continuar")
        print("donde se quedó, sin repetir los fragmentos ya guardados.")

    print(f"\nSe guardaron {total_fragmentos_guardados} fragmentos nuevos en esta corrida.")
    if total_fragmentos_omitidos:
        print(f"Se omitieron {total_fragmentos_omitidos} fragmentos que ya estaban guardados de antes.")
    print(f"La base de datos quedó en: {os.path.abspath(CARPETA_BASE_DATOS)}")


if __name__ == "__main__":
    main()
