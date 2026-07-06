"""
Plataforma Inteligente de Salud Predictiva basada en Big Data
Guia Practica Unidad 2 - TEC.AZUAY
Autor: Santiago Molina

Fuente de datos: API pública disease.sh (COVID-19 / datos de salud global)
Documentación: https://disease.sh/docs/
No requiere API key.
"""

import io
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import streamlit.components.v1 as components

# --------------------------------------------------------------------------------
# CONFIGURACIÓN GENERAL DE LA PÁGINA
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="Salud Predictiva - Big Data",
    page_icon="🏥",
    layout="wide",
)

DB_PATH = "salud_predictiva.db"
TABLE_NAME = "salud_paises"
API_URL = "https://disease.sh/v3/covid-19/countries"
API_HISTORICAL_URL = "https://disease.sh/v3/covid-19/historical/all?lastdays=60"


# --------------------------------------------------------------------------------
# 1. CARGA DE DATOS (desde API + guardado en SQLite)
# --------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cargar_datos_desde_api() -> pd.DataFrame:
    """Consume la API pública disease.sh y devuelve un DataFrame."""
    respuesta = requests.get(API_URL, timeout=15)
    respuesta.raise_for_status()
    datos = respuesta.json()
    df = pd.json_normalize(datos)
    return df


def guardar_en_sqlite(df: pd.DataFrame, db_path: str = DB_PATH, tabla: str = TABLE_NAME):
    """Guarda el DataFrame crudo de la API dentro de una base SQLite."""
    conn = sqlite3.connect(db_path)
    df.to_sql(tabla, conn, if_exists="replace", index=False)
    conn.close()


def leer_desde_sqlite(db_path: str = DB_PATH, tabla: str = TABLE_NAME) -> pd.DataFrame:
    """Lee los datos ya almacenados en SQLite usando pd.read_sql()."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(f"SELECT * FROM {tabla}", conn)
    conn.close()
    return df


@st.cache_data(show_spinner=False)
def cargar_datos_historicos() -> pd.DataFrame:
    """Serie histórica global (para el gráfico de línea)."""
    respuesta = requests.get(API_HISTORICAL_URL, timeout=15)
    respuesta.raise_for_status()
    datos = respuesta.json()
    df_casos = pd.DataFrame(list(datos["cases"].items()), columns=["fecha", "casos"])
    df_muertes = pd.DataFrame(list(datos["deaths"].items()), columns=["fecha", "muertes"])
    df_hist = df_casos.merge(df_muertes, on="fecha")
    df_hist["fecha"] = pd.to_datetime(df_hist["fecha"], format="%m/%d/%y")
    return df_hist


# --------------------------------------------------------------------------------
# 3. LIMPIEZA DE DATOS
# --------------------------------------------------------------------------------
def limpiar_datos(df_raw: pd.DataFrame, quitar_outliers: bool = True) -> pd.DataFrame:
    df = df_raw.copy()

    # Nos quedamos con las columnas relevantes para el análisis de salud
    columnas_utiles = [
        "country", "continent", "population", "cases", "todayCases",
        "deaths", "todayDeaths", "recovered", "active", "critical",
        "casesPerOneMillion", "deathsPerOneMillion", "tests",
        "testsPerOneMillion", "updated", "countryInfo.iso3",
    ]
    columnas_existentes = [c for c in columnas_utiles if c in df.columns]
    df = df[columnas_existentes]

    # Técnica 1: renombrar columnas a español para claridad del dashboard
    df = df.rename(columns={
        "country": "pais",
        "continent": "continente",
        "population": "poblacion",
        "cases": "casos_totales",
        "todayCases": "casos_hoy",
        "deaths": "muertes_totales",
        "todayDeaths": "muertes_hoy",
        "recovered": "recuperados",
        "active": "activos",
        "critical": "criticos",
        "casesPerOneMillion": "casos_por_millon",
        "deathsPerOneMillion": "muertes_por_millon",
        "tests": "pruebas_totales",
        "testsPerOneMillion": "pruebas_por_millon",
        "updated": "actualizado_unix",
        "countryInfo.iso3": "iso3",
    })

    # Técnica 2: corregir tipo de dato -> convertir timestamp unix a datetime real
    if "actualizado_unix" in df.columns:
        df["fecha_actualizacion"] = pd.to_datetime(df["actualizado_unix"], unit="ms")

    # Técnica 3: reemplazar nulos en columnas numéricas por 0 (continente vacío -> "Desconocido")
    columnas_numericas = df.select_dtypes(include=[np.number]).columns
    df[columnas_numericas] = df[columnas_numericas].fillna(0)
    if "continente" in df.columns:
        df["continente"] = df["continente"].fillna("Desconocido")

    # Técnica 4: eliminar registros duplicados por país
    df = df.drop_duplicates(subset=["pais"])

    # Técnica 5: eliminación de outliers de "casos_totales" usando rango intercuartílico (IQR)
    # (se puede desactivar para conservar países con valores extremos, ej. en el mapa mundial)
    if quitar_outliers and "casos_totales" in df.columns:
        q1 = df["casos_totales"].quantile(0.25)
        q3 = df["casos_totales"].quantile(0.75)
        iqr = q3 - q1
        limite_inferior = q1 - 1.5 * iqr
        limite_superior = q3 + 1.5 * iqr
        df = df[(df["casos_totales"] >= limite_inferior) & (df["casos_totales"] <= limite_superior)]

    # Técnica 6: asegurar tipos de datos correctos (enteros para conteos)
    for col in ["casos_totales", "muertes_totales", "recuperados", "activos", "poblacion"]:
        if col in df.columns:
            df[col] = df[col].astype("int64")

    df = df.reset_index(drop=True)
    return df


# --------------------------------------------------------------------------------
# 4. TRANSFORMACIÓN DE DATOS
# --------------------------------------------------------------------------------
def transformar_datos(df_limpio: pd.DataFrame) -> pd.DataFrame:
    df = df_limpio.copy()

    # Transformación 1: nueva columna -> tasa de mortalidad (%)
    df["tasa_mortalidad_pct"] = np.where(
        df["casos_totales"] > 0,
        (df["muertes_totales"] / df["casos_totales"]) * 100,
        0,
    ).round(2)

    # Transformación 2: nueva columna categórica -> nivel de riesgo (apply + función)
    def clasificar_riesgo(tasa):
        if tasa >= 3:
            return "Alto"
        elif tasa >= 1:
            return "Medio"
        else:
            return "Bajo"

    df["nivel_riesgo"] = df["tasa_mortalidad_pct"].apply(clasificar_riesgo)

    # Transformación 3: extraer año y mes de la fecha de actualización
    if "fecha_actualizacion" in df.columns:
        df["anio_actualizacion"] = df["fecha_actualizacion"].dt.year
        df["mes_actualizacion"] = df["fecha_actualizacion"].dt.month

    # Transformación 4: ordenar registros por casos totales descendente
    df = df.sort_values(by="casos_totales", ascending=False)

    return df


def resumen_por_continente(df: pd.DataFrame) -> pd.DataFrame:
    """Agrupación (groupby) de apoyo para gráficos de continente."""
    resumen = (
        df.groupby("continente")[["casos_totales", "muertes_totales", "poblacion"]]
        .sum()
        .reset_index()
        .sort_values(by="casos_totales", ascending=False)
    )
    return resumen


# --------------------------------------------------------------------------------
# INTERFAZ STREAMLIT
# --------------------------------------------------------------------------------
def main():
    st.sidebar.title("🏥 Salud Predictiva Big Data")
    st.sidebar.markdown("**Guía Práctica Unidad 2**")
    seccion = st.sidebar.radio(
        "Navegación",
        [
            "🏠 Inicio",
            "1️⃣ Carga de Datos",
            "2️⃣ Exploración",
            "3️⃣ Limpieza",
            "4️⃣ Transformación",
            "5️⃣ Visualización",
            "6️⃣ Exportación",
        ],
    )

    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Actualizar datos"):
        st.cache_data.clear()
        st.rerun()

    st.sidebar.caption("Fuente de datos: API pública disease.sh")
    st.sidebar.caption("Autor: Santiago Molina · TEC.AZUAY")

    # -------- Carga base para todas las secciones (con cache) --------
    try:
        with st.spinner("Consultando la API de salud..."):
            df_raw = cargar_datos_desde_api()
            guardar_en_sqlite(df_raw)
            df_desde_sqlite = leer_desde_sqlite()
    except Exception as e:
        st.error(
            "⚠️ La API de salud (disease.sh) no está disponible en este momento. "
            "Por favor intenta de nuevo en unos minutos."
        )
        st.caption(f"Detalle técnico: {e}")
        st.stop()

    df_limpio = limpiar_datos(df_desde_sqlite)
    df_final = transformar_datos(df_limpio)

    # Versión completa (todos los países, sin remover outliers) — solo para el mapa mundial,
    # ya que remover los países con más casos les quitaría todo el color al mapa.
    df_mapa_completo = transformar_datos(limpiar_datos(df_desde_sqlite, quitar_outliers=False))

    # ================= INICIO =================
    if seccion == "🏠 Inicio":
        col_izq, col_centro, col_der = st.columns([1, 2, 1])
        with col_centro:
            st.image("logo_tecazuay.png", use_column_width=True)

        st.title("Plataforma Inteligente de Salud Predictiva basada en Big Data")
        st.markdown(
            """
            <div style="text-align: center; margin-top: -10px; margin-bottom: 20px;">
                <p style="margin:0;">Instituto Superior Tecnológico del Azuay</p>
                <p style="margin:0;">Carrera de Tecnología Superior en Big Data</p>
                <p style="margin:0;">Guía Práctica Unidad 2</p>
            </div>
            <p style="text-align:center; font-style: italic; color: #B5B5B5; margin-top: -10px;">
                Caso de estudio: datos epidemiológicos globales de COVID-19
            </p>
            <hr>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            Esta aplicación analiza datos globales de salud pública (casos, muertes,
            pruebas y tasas de mortalidad por país) consumidos en tiempo real desde una
            **API externa**, almacenados en **SQLite**, y procesados con **Pandas** para
            aplicar técnicas de limpieza, transformación y visualización de datos.
            """
        )
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Países analizados", df_final["pais"].nunique())
        col2.metric("Casos totales", f'{int(df_final["casos_totales"].sum()):,}')
        col3.metric("Muertes totales", f'{int(df_final["muertes_totales"].sum()):,}')
        col4.metric(
            "Tasa mortalidad promedio",
            f'{df_final["tasa_mortalidad_pct"].mean():.2f}%',
        )
        st.info("Usa el menú lateral para recorrer cada sección del análisis.")

    # ================= 1. CARGA DE DATOS =================
    elif seccion == "1️⃣ Carga de Datos":
        st.header("1️⃣ Carga de Datos")
        st.markdown(f"**Fuente:** API pública `{API_URL}`")
        st.markdown(f"**Persistencia:** guardado en base de datos SQLite (`{DB_PATH}`, tabla `{TABLE_NAME}`)")

        st.subheader("Vista previa del DataFrame (datos crudos desde SQLite)")
        st.dataframe(df_desde_sqlite.head(10), use_container_width=True)

        col1, col2 = st.columns(2)
        col1.metric("Número de registros", df_desde_sqlite.shape[0])
        col2.metric("Número de columnas", df_desde_sqlite.shape[1])

    # ================= 2. EXPLORACIÓN =================
    elif seccion == "2️⃣ Exploración":
        st.header("2️⃣ Exploración de Datos")

        st.subheader("Primeros registros — df.head()")
        st.dataframe(df_desde_sqlite.head(), use_container_width=True)

        st.subheader("Información general — df.info()")
        buffer = io.StringIO()
        df_desde_sqlite.info(buf=buffer)
        st.text(buffer.getvalue())

        st.subheader("Tipos de datos — df.dtypes")
        st.dataframe(df_desde_sqlite.dtypes.astype(str).rename("tipo_dato"))

        st.subheader("Estadísticas descriptivas — df.describe()")
        st.dataframe(df_desde_sqlite.describe(), use_container_width=True)

        st.subheader("Valores únicos por columna — df.nunique()")
        st.dataframe(df_desde_sqlite.nunique().rename("valores_unicos"))

        st.subheader("Valores nulos por columna — df.isnull().sum()")
        st.dataframe(df_desde_sqlite.isnull().sum().rename("nulos"))

    # ================= 3. LIMPIEZA =================
    elif seccion == "3️⃣ Limpieza":
        st.header("3️⃣ Limpieza de Datos")
        st.markdown(
            """
            Técnicas aplicadas:
            1. **Renombrado de columnas** (`rename`) a español.
            2. **Corrección de tipo de dato**: timestamp Unix → `datetime` (`pd.to_datetime`).
            3. **Reemplazo de nulos** (`fillna`) en columnas numéricas y categóricas.
            4. **Eliminación de duplicados** por país (`drop_duplicates`).
            5. **Eliminación de outliers** en casos totales usando el método IQR.
            6. **Conversión de tipos** (`astype`) a enteros en columnas de conteo.
            """
        )
        st.subheader("Resultado tras la limpieza")
        st.dataframe(df_limpio.head(15), use_container_width=True)
        col1, col2 = st.columns(2)
        col1.metric("Registros antes", df_desde_sqlite.shape[0])
        col2.metric("Registros después", df_limpio.shape[0])
        st.caption("La reducción de registros corresponde a duplicados y outliers eliminados.")

    # ================= 4. TRANSFORMACIÓN =================
    elif seccion == "4️⃣ Transformación":
        st.header("4️⃣ Transformación de Datos")
        st.markdown(
            """
            Transformaciones aplicadas:
            1. **Nueva columna calculada**: tasa de mortalidad (%).
            2. **Nueva columna categórica** con `apply()`: nivel de riesgo (Alto/Medio/Bajo).
            3. **Extracción de fecha** (`dt.year`, `dt.month`) desde la fecha de actualización.
            4. **Ordenamiento** de registros por casos totales (`sort_values`).
            """
        )
        st.subheader("DataFrame transformado")
        st.dataframe(df_final.head(15), use_container_width=True)

        st.subheader("Filtrado interactivo (`loc[]`) por nivel de riesgo")
        riesgo_sel = st.selectbox("Selecciona nivel de riesgo", df_final["nivel_riesgo"].unique())
        st.dataframe(df_final.loc[df_final["nivel_riesgo"] == riesgo_sel], use_container_width=True)

        st.subheader("Agrupación por continente (`groupby`)")
        st.dataframe(resumen_por_continente(df_final), use_container_width=True)

    # ================= 5. VISUALIZACIÓN =================
    elif seccion == "5️⃣ Visualización":
        st.header("5️⃣ Visualización de Datos")

        top_n = st.slider("Número de países a mostrar (Top N)", 5, 30, 10)
        df_top = df_final.head(top_n)

        # Gráfico 1: Barras
        st.subheader("📊 Casos totales por país (Top N)")
        fig_barras = px.bar(
            df_top, x="pais", y="casos_totales", color="continente",
            title="Casos totales por país",
        )
        st.plotly_chart(fig_barras, use_container_width=True)

        # Gráfico 2: Pastel
        st.subheader("🥧 Distribución de casos por continente")
        resumen_continente = resumen_por_continente(df_final)
        fig_pastel = px.pie(
            resumen_continente, names="continente", values="casos_totales",
            title="Participación de casos por continente",
        )
        st.plotly_chart(fig_pastel, use_container_width=True)

        # Gráfico 3: Línea (serie histórica global)
        st.subheader("📈 Evolución histórica global de casos (últimos 60 días)")
        try:
            df_hist = cargar_datos_historicos()
            fig_linea = px.line(
                df_hist, x="fecha", y=["casos", "muertes"],
                title="Evolución global de casos y muertes",
            )
            st.plotly_chart(fig_linea, use_container_width=True)
        except Exception as e:
            st.warning(f"No se pudo cargar la serie histórica en este momento: {e}")

        # Gráfico 4: Histograma
        st.subheader("📉 Distribución de la tasa de mortalidad")
        fig_hist = px.histogram(
            df_final, x="tasa_mortalidad_pct", nbins=30,
            title="Histograma de tasa de mortalidad (%)",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # Gráfico 5: Dispersión
        st.subheader("🔵 Relación entre casos y muertes totales")
        fig_scatter = px.scatter(
            df_final, x="casos_totales", y="muertes_totales",
            color="nivel_riesgo", hover_name="pais", size="poblacion",
            title="Casos vs. Muertes por país",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Gráfico 6: Mapa mundial (choropleth)
        st.subheader("🗺️ Mapa mundial de casos totales")
        if "iso3" in df_mapa_completo.columns:
            df_mapa = df_mapa_completo.dropna(subset=["iso3"]).copy()
            df_mapa = df_mapa[df_mapa["iso3"].astype(str).str.len() == 3]
            df_mapa = df_mapa[df_mapa["casos_totales"] > 0]

            # Escala logarítmica: los casos varían en varios órdenes de magnitud
            # entre países (cientos vs. millones), así que log10 muestra mejor el contraste.
            df_mapa["log_casos"] = np.log10(df_mapa["casos_totales"])

            fig_mapa = px.choropleth(
                df_mapa,
                locations="iso3",
                locationmode="ISO-3",
                color="log_casos",
                hover_name="pais",
                hover_data={
                    "iso3": False,
                    "log_casos": False,
                    "casos_totales": ":,",
                    "muertes_totales": ":,",
                    "nivel_riesgo": True,
                },
                color_continuous_scale="Reds",
                title="Casos totales de COVID-19 por país (escala logarítmica)",
            )
            fig_mapa.update_geos(
                showcoastlines=True,
                showcountries=True,
                showocean=True,
                oceancolor="#0E1117",
                landcolor="#1B2530",
                bgcolor="rgba(0,0,0,0)",
                projection_type="natural earth",
            )
            fig_mapa.update_layout(
                template="plotly_dark",
                margin=dict(l=0, r=0, t=40, b=0),
                paper_bgcolor="#0E1117",
                plot_bgcolor="#0E1117",
                geo=dict(bgcolor="rgba(0,0,0,0)"),
                coloraxis_colorbar=dict(title="Casos (log10)"),
                title_font=dict(color="white"),
                font=dict(color="white"),
            )

            # --- FIX: renderizar como HTML puro para evitar que Streamlit
            # re-pinte el choropleth y se quede sin colorear (theme=None no lo resolvía) ---
            html_mapa = fig_mapa.to_html(
                include_plotlyjs="cdn",
                full_html=False,
                config={"displayModeBar": True, "responsive": True, "scrollZoom": True},
            )
            # Envolvemos en un contenedor con fondo oscuro para que el iframe
            # combine con el resto del dashboard (por defecto components.html es blanco)
            html_mapa_dark = f"""
            <div style="background-color:#0E1117; margin:0; padding:0;">
                {html_mapa}
            </div>
            <style>
                body {{ background-color:#0E1117; margin:0; padding:0; }}
            </style>
            """
            components.html(html_mapa_dark, height=550, scrolling=False)

            st.caption(
                f"Se muestran {df_mapa.shape[0]} países con datos válidos de casos totales "
                "(incluye países con valores extremos, ya que aquí no se aplica el filtro de outliers). "
                "Pasa el mouse sobre cada país para ver el detalle. Puedes hacer zoom con la rueda o arrastrar para mover el mapa."
            )

            with st.expander("🔍 Diagnóstico técnico del mapa (verificar si hay problema de datos)"):
                st.write("Rango de casos_totales:", df_mapa["casos_totales"].min(), "→", df_mapa["casos_totales"].max())
                st.write("Rango de log_casos (lo que colorea el mapa):", df_mapa["log_casos"].min(), "→", df_mapa["log_casos"].max())
                st.write("Valores únicos de log_casos:", df_mapa["log_casos"].nunique())
                st.write("Muestra de datos que se están graficando:")
                st.dataframe(df_mapa[["pais", "iso3", "casos_totales", "log_casos"]].sample(min(15, len(df_mapa))))
        else:
            st.warning("No se encontró el código ISO3 de los países para construir el mapa.")

    # ================= 6. EXPORTACIÓN =================
    elif seccion == "6️⃣ Exportación":
        st.header("6️⃣ Exportación de Datos")
        st.markdown("Descarga el conjunto de datos limpio y transformado, listo para el proyecto final.")

        df_final.to_csv("datos_limpios.csv", index=False)
        df_final.to_excel("datos_limpios.xlsx", index=False)

        st.dataframe(df_final.head(10), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "⬇️ Descargar CSV",
                data=df_final.to_csv(index=False).encode("utf-8"),
                file_name="datos_limpios.csv",
                mime="text/csv",
            )
        with col2:
            buffer_excel = io.BytesIO()
            df_final.to_excel(buffer_excel, index=False)
            st.download_button(
                "⬇️ Descargar Excel",
                data=buffer_excel.getvalue(),
                file_name="datos_limpios.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        st.success("Archivos generados: datos_limpios.csv y datos_limpios.xlsx")


if __name__ == "__main__":
    main()