import streamlit as st
import pandas as pd
import plotly.express as px
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# --- CONFIGURACIÓN DE NLTK Y STOPWORDS ---
import nltk
from nltk.corpus import stopwords
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Definir lista de exclusión para la Nube de Palabras
stop_es = set(stopwords.words('spanish'))
stop_dominio = {
    "seguros", "aseguradora", "aseguradoras", "póliza", "poliza", "asegurado", "asegurada",
    "cliente", "clientes", "grupo", "fundacion", "colombia", "colombiano", "empresa", "nacional",
    "gestión", "riesgo", "integral", "fiduciaria", "previsora", "axa", "colpatria", "servicio", "producto",
    "ser", "haber", "hacer", "año", "anos", "mes", "dia", "dias", "mil", "millon", "millones", "pesos",
    "https", "http", "co", "com", "www", "status", "web", "twitter", "x", "tweet", 
    "si", "va", "tan", "mas", "ahora", "hoy", "asi", "dice", "hace", "solo", "sus", "por", "para"
}
STOPWORDS_CLOUD = stop_es | stop_dominio

# ==========================================
# 1. CONFIGURACIÓN DE PÁGINA
# ==========================================
st.set_page_config(page_title="Monitor de Aseguradoras", page_icon="🛡️", layout="wide")
st.title("🛡️ Monitor de Reputación en Seguros")
st.markdown("Análisis de tópicos y sentimiento en redes sociales (X/Twitter).")

# ==========================================
# 2. CARGA DE DATOS ROBUSTA
# ==========================================
# ==========================================
# 2. CARGA DE DATOS (MODO CONFIANZA)
# ==========================================
@st.cache_data
def load_data():
    # 1. RUTA ABSOLUTA (La copiaste del Notebook, pégala aquí si cambia)
    ruta = r"c:\Users\santi\Downloads\Learning\Maestria\Programacion Lenguaje Natural\Proyecto\Entrega 3\Resultados_Fase3\DATASET_FINAL_MAESTRO.csv"
    
    if not os.path.exists(ruta):
        return None

    # 2. Leer archivo
    df = pd.read_csv(ruta, sep=";")
    
    # 3. Normalizar Fechas
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')

    # 4. LIMPIEZA CRÍTICA DE SENTIMIENTO
    # El Notebook ya exporta "Positivo", "Negativo". Solo aseguramos que sean strings.
    # Si hay espacios extra "Positivo ", los quitamos.
    if "sentimiento" in df.columns:
        df["sentimiento_final"] = df["sentimiento"].astype(str).str.strip()
    else:
        # Fallback de emergencia si la columna se llama distinto
        df["sentimiento_final"] = "Neutro"

    # 5. Ordenar para que los gráficos salgan Verde-Gris-Rojo
    orden = ["Positivo", "Neutro", "Negativo"]
    df["sentimiento_final"] = pd.Categorical(
        df["sentimiento_final"], 
        categories=orden, 
        ordered=True
    )
    
    return df

df = load_data()

if df is None:
    st.error("❌ No se encuentra el archivo 'DATASET_FINAL_MAESTRO.csv'.")
    st.stop()

# ==========================================
# 3. BARRA LATERAL (FILTROS CON "SELECCIONAR TODO")
# ==========================================
st.sidebar.header("Filtros")

# --- A. Marcas ---
st.sidebar.subheader("Marcas")
marcas = sorted(df["brand_primary"].dropna().unique().astype(str))
chk_todas_marcas = st.sidebar.checkbox("✅ Todas las Marcas", value=False)

if chk_todas_marcas:
    sel_marcas = st.sidebar.multiselect("Selección:", marcas, default=marcas)
else:
    sel_marcas = st.sidebar.multiselect("Selección:", marcas, default=marcas[:3])

# --- B. Temas ---
st.sidebar.subheader("Temas")
temas = sorted(df["topic_name"].dropna().unique().astype(str))
# Excluir ruido por defecto si no se selecciona "Todos"
temas_limpios = [t for t in temas if "Excluido" not in t and "Ruido" not in t]

chk_todos_temas = st.sidebar.checkbox("✅ Todos los Temas", value=False)

if chk_todos_temas:
    sel_temas = st.sidebar.multiselect("Selección:", temas, default=temas)
else:
    sel_temas = st.sidebar.multiselect("Selección:", temas, default=temas_limpios)

# --- C. Fechas ---
if "Date" in df.columns:
    st.sidebar.markdown("---")
    min_date = df["Date"].min()
    max_date = df["Date"].max()
    date_range = st.sidebar.date_input("Rango Fechas:", [min_date, max_date])

# --- APLICAR FILTROS ---
if not sel_marcas or not sel_temas:
    st.warning("⚠️ Selecciona al menos una marca y un tema.")
    st.stop()

df_filtered = df[
    (df["brand_primary"].isin(sel_marcas)) & 
    (df["topic_name"].isin(sel_temas))
]

if "Date" in df.columns and len(date_range) == 2:
    df_filtered = df_filtered[
        (df_filtered["Date"] >= pd.to_datetime(date_range[0])) & 
        (df_filtered["Date"] <= pd.to_datetime(date_range[1]))
    ]

st.sidebar.markdown("---")
st.sidebar.metric("Tweets Filtrados", f"{len(df_filtered):,}")

# ==========================================
# 4. VISUALIZACIÓN
# ==========================================
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🆚 Comparativo", "🔎 Explorador"])

# --- TAB 1: DASHBOARD ---
with tab1:
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("Distribución de Sentimiento")
        # Colores fijos: Verde, Gris, Rojo
        colores = {'Positivo':'#2ecc71', 'Neutro':'#95a5a6', 'Negativo':'#e74c3c'}
        
        fig_pie = px.pie(
            df_filtered, names='sentimiento_final', 
            color='sentimiento_final',
            color_discrete_map=colores,
            hole=0.4
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with c2:
        st.subheader("Métricas")
        total = len(df_filtered)
        pos = len(df_filtered[df_filtered['sentimiento_final']=='Positivo'])
        neg = len(df_filtered[df_filtered['sentimiento_final']=='Negativo'])
        neu = len(df_filtered[df_filtered['sentimiento_final']=='Neutro'])
        
        st.metric("Total Tweets", f"{total:,}")
        st.metric("Positivos", f"{pos:,} ({pos/total*100:.1f}%)")
        st.metric("Negativos", f"{neg:,} ({neg/total*100:.1f}%)", delta_color="inverse")

    st.markdown("---")
    st.subheader("Temas de Conversación")
    
    counts = df_filtered['topic_name'].value_counts().reset_index()
    counts.columns = ['Tema', 'Volumen']
    
    fig_bar = px.bar(
        counts, x='Volumen', y='Tema', orientation='h', 
        color='Volumen', color_continuous_scale='Blues',
        text='Volumen'
    )
    fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_bar, use_container_width=True)

# --- TAB 2: COMPARATIVO ---
with tab2:
    st.subheader("Sentimiento por Marca (Normalizado 100%)")
    
    comp_data = df_filtered.groupby(['brand_primary', 'sentimiento_final'], observed=False).size().reset_index(name='count')
    comp_data['percentage'] = comp_data.groupby('brand_primary')['count'].transform(lambda x: x / x.sum() * 100)
    
    fig_comp = px.bar(
        comp_data, y="brand_primary", x="percentage", color="sentimiento_final",
        orientation='h',
        color_discrete_map=colores,
        text=comp_data['percentage'].apply(lambda x: '{0:1.0f}%'.format(x) if x > 5 else ''),
        title="Comparativa de Reputación"
    )
    st.plotly_chart(fig_comp, use_container_width=True)

# --- TAB 3: EXPLORADOR ---
with tab3:
    c1, c2 = st.columns([1, 3])
    with c1:
        st.markdown("#### Filtros de Texto")
        f_sent = st.multiselect("Sentimiento:", ["Positivo", "Negativo", "Neutro"], default=["Negativo", "Positivo"])
        search = st.text_input("Buscar palabra clave:")
    
    with c2:
        # Aplicar filtros locales
        df_zoom = df_filtered[df_filtered['sentimiento_final'].isin(f_sent)]
        if search:
            df_zoom = df_zoom[df_zoom['text_clean'].astype(str).str.contains(search.lower(), na=False)]
        
        st.dataframe(
            df_zoom[["brand_primary", "sentimiento_final", "topic_name", "text_raw"]].sample(min(len(df_zoom), 100), random_state=42),
            use_container_width=True,
            hide_index=True
        )
        st.caption(f"Mostrando {min(len(df_zoom), 100)} registros de {len(df_zoom)} encontrados.")

    st.markdown("---")
    if st.checkbox("☁️ Generar Nube de Palabras (Sin basura)"):
        st.info("Generando visualización...")
        # Usar 'text_clean' (con tildes) para mejor lectura
        text_cloud = " ".join(df_zoom["text_clean"].astype(str))
        
        if len(text_cloud) > 50:
            wc = WordCloud(
                width=900, height=400, 
                background_color="white", 
                colormap="viridis",
                stopwords=STOPWORDS_CLOUD, # <--- AQUÍ ACTÚA EL FILTRO
                max_words=80
            ).generate(text_cloud)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.warning("No hay suficiente texto para generar la nube.")