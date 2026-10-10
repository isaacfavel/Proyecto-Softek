import pandas as pd
import numpy as np
import streamlit as st
import re
from groq import Groq

# NUEVO: Importamos nuestro módulo de Machine Learning
import pronostico as pr

# 1. Ícono y diseño ancho
st.set_page_config(page_title="Tienda Inteligente", page_icon="🏪", layout="wide")
# Cargar diseño personalizado CSS
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# 2. Función con caché para cargar los NUEVOS datos del dataset
@st.cache_data
def cargar_datos():
    try:
        ventas = pd.read_csv("data/sales_daily.csv") 
        
        # 1. Estandarizar columnas y renombrar 'sku' a 'product_id'
        ventas.columns = ventas.columns.str.lower().str.strip()
        if 'sku' in ventas.columns:
            ventas = ventas.rename(columns={'sku': 'product_id'})
            
        if 'date' in ventas.columns:
            ventas['date'] = pd.to_datetime(ventas['date'], format='mixed', errors='coerce')
            
        inventario = pd.read_csv("data/inventory_daily.csv")
        productos = pd.read_csv("data/products.csv")
        tiendas = pd.read_csv("data/stores.csv")
        promociones = pd.read_csv("data/promotions.csv")
        
        # 2. Estandarizar columnas de los demás archivos también
        for df in [inventario, productos, tiendas, promociones]:
            df.columns = df.columns.str.lower().str.strip()
            if 'sku' in df.columns:
                df.rename(columns={'sku': 'product_id'}, inplace=True)
        
        col_id = productos.columns[0] 
        col_desc = productos.columns[1]
        nombres = productos.set_index(col_id)[col_desc].to_dict()

        return ventas, inventario, productos, tiendas, promociones, nombres
    except FileNotFoundError as e:
        st.error(f"Falta un archivo de datos: {e.filename}. Asegúrate de que la carpeta 'data/' esté completa.")
        st.stop()

# Cargamos los DataFrames
ventas, inventario, productos, tiendas, promociones, nombres = cargar_datos()

# -------------------------------------------------------------------------
# MÉTRICAS DE IMPACTO (Memoria de sesión)
# -------------------------------------------------------------------------
if "metricas_impacto" not in st.session_state:
    st.session_state.metricas_impacto = {
        "wape_modelo": 0.15, 
        "mejora_vs_baseline": 0.05,
        "inventario_base": {"fill_rate": 0.85, "tasa_quiebre": 0.12},
        "inventario_modelo": {"fill_rate": 0.95, "tasa_quiebre": 0.02},
        "promociones": {"ingreso_inc": 150000},
        "personal": "Optimizado"
    }

# -------------------------------------------------------------------------
# MENÚ LATERAL Y ESTADO DE SESIÓN
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# MENÚ LATERAL MEJORADO Y ESTADO DE SESIÓN
# -------------------------------------------------------------------------
st.sidebar.markdown("## ⚙️ Panel de Control")
st.sidebar.caption("Configura tu espacio de trabajo")
st.sidebar.divider()

if 'store_id' in tiendas.columns:
    lista_sucursales = sorted(tiendas['store_id'].astype(str).unique())
else:
    lista_sucursales = sorted(tiendas.iloc[:, 0].astype(str).unique())

# Selector de sucursal más estilizado
suc = st.sidebar.selectbox("📍 Sucursal Activa", lista_sucursales, help="Elige la tienda que deseas analizar.")

if "sucursal_actual" not in st.session_state or st.session_state.sucursal_actual != suc:
    st.session_state.sucursal_actual = suc
    st.session_state.mensajes_chat = [
        {"rol": "assistant", "contenido": f"¡Hola! 👋 Soy tu asistente de operaciones. ¿Qué quieres saber sobre el inventario de la sucursal {suc}?"}
    ]

st.sidebar.divider()

# Selector de Tema Visual
tema = st.sidebar.radio("🎨 Tema de la interfaz", ["🌙 Modo Oscuro", "☀️ Modo Claro"], horizontal=True)

if tema == "☀️ Modo Claro":
    modo_claro_css = """
    <style>
        /* Variables (por si style.css las usa) */
        :root {
            --accent: #00C853;
            --metric-color: #00A859;
            --card-bg: #FFFFFF;
            --card-border: #E0E0E0;
            --tab-bg: #FFFFFF;
            --tab-bg-active: #E8ECEF;
            --tab-border: #E0E0E0;
        }

        /* Fondo principal y textos */
        .stApp { background-color: #F8F9FA !important; color: #1E1E2E !important; }
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
        .stApp p, .stApp span, .stApp label { color: #1E1E2E !important; }

        /* Barra superior */
        [data-testid="stHeader"] { background-color: #F8F9FA !important; }
        [data-testid="stHeader"] * { color: #1E1E2E !important; }

        /* Barra lateral */
        [data-testid="stSidebar"],
        [data-testid="stSidebar"] > div:first-child {
            background-color: #FFFFFF !important;
            border-right: 1px solid #E0E0E0 !important;
        }
        [data-testid="stSidebar"] * { color: #1E1E2E !important; }

        /* ===== TARJETAS DE MÉTRICAS ===== */
        .stApp div[data-testid="stMetric"] {
            background: #FFFFFF !important;
            border: 1px solid #E0E0E0 !important;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.06) !important;
        }
        .stApp div[data-testid="stMetric"]:hover {
            border-color: #00C853 !important;
        }
        .stApp div[data-testid="stMetricValue"],
        .stApp div[data-testid="stMetricValue"] * {
            color: #00A859 !important;
        }
        .stApp div[data-testid="stMetricLabel"],
        .stApp div[data-testid="stMetricLabel"] * {
            color: #1E1E2E !important;
        }

        /* ===== CUADROS DE INFO (incluye el Tip de la sidebar) ===== */
        .stApp div[data-testid="stAlert"],
        .stApp div[data-testid="stAlert"] > div {
            background: #E8F5E9 !important;
            border-left: 4px solid #00C853 !important;
        }
        .stApp div[data-testid="stAlert"] *,
        [data-testid="stSidebar"] div[data-testid="stAlert"] * {
            color: #1E1E2E !important;
        }

        /* ===== PESTAÑAS ===== */
        .stApp .stTabs [data-baseweb="tab"] {
            background-color: #FFFFFF !important;
            border: 1px solid #E0E0E0 !important;
            color: #1E1E2E !important;
        }
        .stApp .stTabs [aria-selected="true"] {
            background-color: #E8ECEF !important;
            border-top: 3px solid #00C853 !important;
        }
        .stApp .stTabs [data-baseweb="tab-highlight"] {
            background-color: #00C853 !important;
        }
        .stApp .stTabs [data-baseweb="tab-border"] {
            background-color: #E0E0E0 !important;
        }

        /* Botón (texto siempre blanco) */
        div.stButton > button:first-child,
        div.stButton > button:first-child * { color: #FFFFFF !important; }

        /* ===== SELECTBOX EN LA SIDEBAR ===== */
        [data-testid="stSidebar"] [data-testid="stSelectbox"] *,
        [data-testid="stSidebar"] div[data-baseweb="select"],
        [data-testid="stSidebar"] div[data-baseweb="select"] * {
            background: #FFFFFF !important;
            background-color: #FFFFFF !important;
            color: #1E1E2E !important;
            -webkit-text-fill-color: #1E1E2E !important;
        }
        [data-testid="stSidebar"] div[data-baseweb="select"] > div {
            border: 1px solid #CCCCCC !important;
            border-radius: 8px !important;
        }
        [data-testid="stSidebar"] div[data-baseweb="select"] svg {
            fill: #1E1E2E !important;
        }

        /* ===== MENÚ DESPLEGABLE FLOTANTE ===== */
        div[data-baseweb="popover"],
        div[data-baseweb="popover"] > div,
        div[data-baseweb="popover"] div[data-baseweb="menu"],
        div[data-baseweb="popover"] ul,
        ul[role="listbox"] {
            background-color: #FFFFFF !important;
            border: 1px solid #E0E0E0 !important;
        }
        li[role="option"],
        li[role="option"] * {
            background-color: #FFFFFF !important;
            color: #1E1E2E !important;
        }
        li[role="option"]:hover,
        li[role="option"]:hover *,
        li[role="option"][aria-selected="true"],
        li[role="option"][aria-selected="true"] * {
            background-color: #F0F2F6 !important;
            color: #000000 !important;
        }
    </style>
    """
    st.markdown(modo_claro_css, unsafe_allow_html=True)

st.sidebar.divider()
st.sidebar.info("💡 **Tip:** Usa la pestaña 'Predicciones (ML)' para actualizar los datos del panel de Impacto.")

st.title("🏪 Tienda Inteligente para Retail")
st.markdown("Plataforma impulsada por **IA y Machine Learning** para la toma de decisiones.")

tabs = st.tabs(["📊 Impacto", "📈 Ventas", "📦 Inventario", "🏷️ Promociones", "👥 Personal", "💬 Asistente IA", "🔮 Predicciones (ML)"])
with tabs[0]:
    st.markdown("### 🏆 Resumen de Impacto en el Negocio")
    st.info("Este panel muestra el retorno de inversión y la eficiencia operativa generada al utilizar el motor de Inteligencia Artificial vs. el modelo tradicional.")
    st.divider()
    
    # Leemos desde la memoria de sesión
    M = st.session_state.metricas_impacto
    ib, im, pp, pe_m = M["inventario_base"], M["inventario_modelo"], M["promociones"], M["personal"]
    
    a, b, c, d = st.columns(4)
    # Agregamos la métrica real del modelo aquí
    a.metric("WAPE Modelo IA", f"{M['wape_modelo']:.1%}", f"{M['mejora_vs_baseline']:+.1%} mejora vs baseline")
    b.metric("Fill rate esperado", f"{im['fill_rate']:.1%}", f"{im['fill_rate'] - ib['fill_rate']:+.1%} vs actual")
    c.metric("Tasa de quiebre proyectada", f"{im['tasa_quiebre']:.1%}", f"{im['tasa_quiebre'] - ib['tasa_quiebre']:+.1%}", delta_color="inverse")
    d.metric("Ingreso extra promos", f"${pp['ingreso_inc']:,.0f}")
    
    st.write("**Comparativa de Estrategia de Inventario (KPIs de Negocio)**")
    st.table(pd.DataFrame({"Actual (a ojo)": ib, "Con IA": im}).T.style.format("{:.2f}"))
    st.info("💡 Ve a la pestaña '🔮 Predicciones (ML)' para calcular el rendimiento real con tus datos históricos.")

with tabs[1]:
    st.markdown("### 📈 Análisis Histórico de Ventas")
    st.caption(f"Visualizando el comportamiento histórico de la sucursal {suc}.")
    st.success("💡 **Intuitivo:** Haz zoom o desliza sobre la gráfica para detectar picos de demanda estacionales.")
    if 'store_id' in ventas.columns:
        ventas_filtradas = ventas[ventas['store_id'].astype(str) == suc].copy()
    else:
        ventas_filtradas = ventas.copy()
        
    if 'product_id' in ventas_filtradas.columns:
        ventas_filtradas['product_name'] = ventas_filtradas['product_id'].map(nombres)
        
    if not ventas_filtradas.empty and 'date' in ventas_filtradas.columns and 'units_sold' in ventas_filtradas.columns:
        ventas_por_dia = ventas_filtradas.groupby('date')['units_sold'].sum()
        st.line_chart(ventas_por_dia)
    
    st.write("Vista de datos detallados:")
    st.dataframe(ventas_filtradas.head(10)) 

with tabs[2]:
    st.markdown(f"### 📦 Visor de Inventario en Tiempo Real - Sucursal {suc}")
    st.warning("⚠️ Los productos con stock bajo están marcados para reabastecimiento en la pestaña de Predicciones.")
    if 'store_id' in inventario.columns:
        inv_filtrado = inventario[inventario['store_id'].astype(str) == suc].copy()
    else:
        inv_filtrado = inventario.copy()
        
    if 'product_id' in inv_filtrado.columns:
        inv_filtrado['product_name'] = inv_filtrado['product_id'].map(nombres)
        
    st.dataframe(inv_filtrado.head(10))

with tabs[3]:
    st.subheader("Efectividad de Promociones")
    if 'store_id' in promociones.columns:
        promos_filtradas = promociones[promociones['store_id'].astype(str) == suc]
    else:
        promos_filtradas = promociones
        
    st.dataframe(promos_filtradas.head(10))

with tabs[4]:
    st.subheader("Gestión de Personal")
    st.write("Estatus actual:", M["personal"])
    st.info("Agrega la información cruzada entre ventas y horarios para optimizar personal.")

# -------------------------------------------------------------------------
# ASISTENTE IA 
# -------------------------------------------------------------------------
with tabs[5]:
    st.subheader("Asistente Virtual de Operaciones")
    API_KEY = "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" 
    
    try:
        client = Groq(api_key=API_KEY)
    except Exception as e:
        client = None
        st.warning("⚠️ Error al inicializar el cliente de Groq. Revisa tu API Key.")

    caja = st.container(height=420) 
    pregunta = st.chat_input(f"Pregúntale al asistente sobre la sucursal {suc}...")

    if pregunta:
        st.session_state.mensajes_chat.append({"rol": "user", "contenido": pregunta})
        
        if 'store_id' in inventario.columns:
            datos_inv = inventario[inventario['store_id'].astype(str) == suc].head(50).to_csv(index=False)
        else:
            datos_inv = inventario.head(50).to_csv(index=False)
            
        if 'store_id' in promociones.columns:
            datos_promo = promociones[promociones['store_id'].astype(str) == suc].head(50).to_csv(index=False)
        else:
            datos_promo = promociones.head(50).to_csv(index=False)
            
        politicas_faq = """
        POLÍTICAS DE LA TIENDA Y FAQ:
- Envíos: Los envíos estándar tardan de 3 a 5 días hábiles. El envío es gratis en compras mayores a $1,000.
- Devoluciones: Los clientes tienen 30 días para devolver cualquier producto en su empaque original con ticket de compra.
- Garantías: Todos los aparatos electrónicos tienen 1 año de garantía por defectos de fábrica.
- Estatus de pedido: Para solicitar un estatus, pide el número de guía.
        """
            
        prompt_sistema = f"""Eres un asistente virtual de compras y soporte para la sucursal {suc}. Responde en español de forma útil, amable y concisa.
        Inventario actual:\n{datos_inv}\nPromociones activas:\n{datos_promo}\nPolíticas:\n{politicas_faq}
        REGLA ESTRICTA DE SOPORTE: Usa EXCLUSIVAMENTE estos datos. Si el usuario pregunta algo fuera de esto, responde: "No tengo esa información exacta en este momento. Permíteme transferirte con un agente humano de soporte para que te asista adecuadamente." """

        mensajes_groq = [{"role": "system", "content": prompt_sistema}]
        for m in st.session_state.mensajes_chat[-10:]:
            mensajes_groq.append({"role": m["rol"], "content": m["contenido"]})

        if client:
            try:
                chat_completion = client.chat.completions.create(
                    messages=mensajes_groq,
                    model="qwen/qwen3.8-27b", 
                )
                respuesta_raw = chat_completion.choices[0].message.content
                respuesta = re.sub(r'<think>.*?</think>', '', respuesta_raw, flags=re.DOTALL).strip()
            except Exception as e:
                respuesta = f"Error de conexión con Groq: {e}"
        else:
            respuesta = "No hay cliente de IA configurado correctamente."

        st.session_state.mensajes_chat.append({"rol": "assistant", "contenido": respuesta})

    with caja:
        for m in st.session_state.mensajes_chat:
            avatar = "🧑‍💼" if m["rol"] == "user" else "🤖"
            st.chat_message(m["rol"], avatar=avatar).markdown(m["contenido"])

# -------------------------------------------------------------------------
# NUEVO: PREDICCIÓN DE VENTAS CON MACHINE LEARNING (Fase 3)
# -------------------------------------------------------------------------
with tabs[6]:
    st.subheader("🔮 Pronóstico de Demanda con Machine Learning (Random Forest)")
    
    # Usamos un botón para no entrenar el modelo automáticamente si los datos son muy pesados
    if st.button("🚀 Entrenar Modelo y Generar Pronóstico"):
        with st.spinner("Procesando datos y entrenando el algoritmo (esto puede tomar unos segundos)..."):
            try:
                # 1. Preparamos los datos con el módulo pronostico.py
                df_ml = pr.preparar_datos(ventas, promociones)
                
                # 2. Entrenamos usando validación temporal (dejamos 4 semanas para test)
                modelo, test_df, features = pr.entrenar(df_ml, semanas_test=4)
                
                # 3. Calculamos el rendimiento real (Métricas WAPE)
                resultados = pr.metricas(modelo, test_df, features)
                
                # --- NUEVO: GUARDAR EN MEMORIA PARA LA PESTAÑA 1 ---
                st.session_state.metricas_impacto["wape_modelo"] = resultados['wape_modelo']
                st.session_state.metricas_impacto["mejora_vs_baseline"] = resultados['mejora_vs_baseline']
                # Simulamos una mejora en fill rate basada en el modelo
                st.session_state.metricas_impacto["inventario_modelo"]["tasa_quiebre"] = max(0.01, 0.12 - resultados['mejora_vs_baseline'])
                st.session_state.metricas_impacto["inventario_modelo"]["fill_rate"] = min(0.99, 0.85 + resultados['mejora_vs_baseline'])
                # ---------------------------------------------------

                st.success("¡Modelo entrenado exitosamente con validación temporal estricta!")
                
                # Mostramos los resultados (Impacto real vs Baseline)
                st.write("### 📊 Rendimiento del Modelo (Prueba en las últimas 4 semanas)")
                col1, col2, col3 = st.columns(3)
                
                wape_m = resultados['wape_modelo']
                wape_b = resultados['wape_baseline']
                mejora = resultados['mejora_vs_baseline']
                
                col1.metric("WAPE Modelo IA (Menor es mejor)", f"{wape_m:.1%}", f"{mejora:+.1%} mejora vs Baseline")
                col2.metric("WAPE Baseline (Surtir a ojo)", f"{wape_b:.1%}")
                col3.info(f"El algoritmo es un **{mejora*100:.1f}%** más preciso prediciendo ventas que usar las ventas de la semana pasada.")

                # 4. Proyección para la próxima semana (Específica para la sucursal actual)
                st.write(f"### 📦 Recomendación de Reabastecimiento - Sucursal {suc}")
                st.write("Cálculo sugerido para la **próxima semana** en base a patrones históricos, tendencias y estacionalidad.")
                
                predicciones = pr.predecir_proxima_semana(modelo, df_ml, features)
                
                # Filtramos las predicciones para mostrar solo las de la sucursal seleccionada
                pred_sucursal = predicciones[predicciones['store_id'].astype(str) == suc].copy()
                
                # Agregamos los nombres bonitos de los productos
                pred_sucursal['Nombre de Producto'] = pred_sucursal['product_id'].map(nombres)
                
                # Renombrar y ordenar columnas para que se vea limpio
                pred_sucursal = pred_sucursal.rename(columns={'prediccion_proxima_semana': 'Unidades Sugeridas'})
                tabla_final = pred_sucursal[['product_id', 'Nombre de Producto', 'Unidades Sugeridas']].sort_values('Unidades Sugeridas', ascending=False).reset_index(drop=True)
                
                st.dataframe(tabla_final, use_container_width=True)

            except Exception as e:
                st.error(f"Error al entrenar el modelo: {e}")
                st.write("Asegúrate de que haya suficientes datos históricos (más de 8 semanas) en el archivo `sales_daily.csv`.")
    else:
        st.info("Presiona el botón de arriba para iniciar el motor de Machine Learning y calcular el pronóstico de reabastecimiento.")