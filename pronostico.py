import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def preparar_datos(ventas, promociones=None):
    """
    Agrupa ventas por semana, producto y sucursal, y crea variables predictivas (lags, promedios).
    """
    # Asegurar formato de fecha
    ventas['date'] = pd.to_datetime(ventas['date'], format='mixed', errors='coerce')
    
    # 1. Agregar ventas por semana (lunes a domingo), sucursal y producto
    # Verificamos que existan las columnas base en ventas
    cols_agrupacion = []
    if 'store_id' in ventas.columns: cols_agrupacion.append('store_id')
    if 'product_id' in ventas.columns: cols_agrupacion.append('product_id')
    
    if not cols_agrupacion:
        raise ValueError("El archivo de ventas no contiene 'store_id' ni 'product_id'.")
        
    df = ventas.groupby(
        cols_agrupacion + [pd.Grouper(key='date', freq='W-Mon')]
    )['units_sold'].sum().reset_index()
    
    # Ordenar cronológicamente para que los 'shifts' (rezagos) funcionen bien
    df = df.sort_values(by=cols_agrupacion + ['date'])
    
    # 2. Crear variables (Features)
    df['semana_anio'] = df['date'].dt.isocalendar().week.astype(int)
    
    # Variables de rezago (Lag) y promedios móviles
    df['ventas_lag_1'] = df.groupby(cols_agrupacion)['units_sold'].shift(1)
    df['ventas_lag_2'] = df.groupby(cols_agrupacion)['units_sold'].shift(2)
    df['ventas_lag_4'] = df.groupby(cols_agrupacion)['units_sold'].shift(4)
    
    # Promedio móvil de las últimas 4 semanas
    df['promedio_movil_4w'] = df.groupby(cols_agrupacion)['ventas_lag_1'].rolling(window=4, min_periods=1).mean().reset_index(level=list(range(len(cols_agrupacion))), drop=True)
    
    # 3. Integrar Promociones de forma segura
    df['en_promo'] = 0
    if promociones is not None and not promociones.empty:
        # Solo intentamos cruzar si ambas columnas existen en promociones
        if 'store_id' in promociones.columns and 'product_id' in promociones.columns:
            promos_activas = promociones[['store_id', 'product_id']].drop_duplicates()
            promos_activas['en_promo_flag'] = 1
            df = df.merge(promos_activas, on=['store_id', 'product_id'], how='left')
            df['en_promo'] = df['en_promo_flag'].fillna(0)
            df = df.drop(columns=['en_promo_flag'])
        elif 'product_id' in promociones.columns:
            # Si solo hay product_id
            promos_activas = promociones[['product_id']].drop_duplicates()
            promos_activas['en_promo_flag'] = 1
            df = df.merge(promos_activas, on=['product_id'], how='left')
            df['en_promo'] = df['en_promo_flag'].fillna(0)
            df = df.drop(columns=['en_promo_flag'])

    # Eliminar NaNs generados por los shifts (perdemos las primeras 4 semanas de historia)
    df = df.dropna()
    return df

def calcular_wape(y_true, y_pred):
    """Calcula el Weighted Absolute Percentage Error"""
    suma_errores = np.sum(np.abs(y_true - y_pred))
    suma_real = np.sum(y_true)
    if suma_real == 0:
        return 0
    return suma_errores / suma_real

def entrenar(df, semanas_test=4):
    """
    Entrena el modelo usando validación por tiempo.
    """
    features = ['semana_anio', 'ventas_lag_1', 'ventas_lag_2', 'ventas_lag_4', 'promedio_movil_4w', 'en_promo']
    
    fechas_unicas = sorted(df['date'].unique())
    
    if len(fechas_unicas) <= semanas_test:
        raise ValueError(f"Solo hay {len(fechas_unicas)} semanas de datos. Se necesitan más para entrenar y evaluar el modelo.")
        
    fecha_corte = fechas_unicas[-semanas_test]
    
    train_df = df[df['date'] < fecha_corte]
    test_df = df[df['date'] >= fecha_corte]
    
    X_train = train_df[features]
    y_train = train_df['units_sold']
    
    X_test = test_df[features]
    y_test = test_df['units_sold']
    
    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)
    
    return modelo, test_df, features

def metricas(modelo, test_df, features):
    """
    Calcula el WAPE del modelo vs el baseline (ventas de la semana pasada)
    """
    test_df_copy = test_df.copy()
    test_df_copy['prediccion'] = modelo.predict(test_df_copy[features])
    test_df_copy['baseline'] = test_df_copy['ventas_lag_1']
    
    wape_modelo = calcular_wape(test_df_copy['units_sold'], test_df_copy['prediccion'])
    wape_baseline = calcular_wape(test_df_copy['units_sold'], test_df_copy['baseline'])
    
    mejora = wape_baseline - wape_modelo
    
    return {
        "wape_modelo": wape_modelo,
        "wape_baseline": wape_baseline,
        "mejora_vs_baseline": mejora,
        "detalle_test": test_df_copy
    }

def predecir_proxima_semana(modelo, df, features):
    """
    Proyecta la venta de la semana que sigue para cada producto en base a los últimos datos conocidos.
    """
    cols_agrupacion = []
    if 'store_id' in df.columns: cols_agrupacion.append('store_id')
    if 'product_id' in df.columns: cols_agrupacion.append('product_id')
        
    ultimos_datos = df.loc[df.groupby(cols_agrupacion)['date'].idxmax()].copy()
    
    ultimos_datos['ventas_lag_4'] = ultimos_datos['ventas_lag_2'] 
    ultimos_datos['ventas_lag_2'] = ultimos_datos['ventas_lag_1']
    ultimos_datos['ventas_lag_1'] = ultimos_datos['units_sold']
    ultimos_datos['semana_anio'] = (ultimos_datos['semana_anio'] + 1) % 52
    if ultimos_datos['semana_anio'].eq(0).any(): 
        ultimos_datos['semana_anio'] = ultimos_datos['semana_anio'].replace(0, 52)
        
    ultimos_datos['prediccion_proxima_semana'] = modelo.predict(ultimos_datos[features])
    
    columnas_resultado = cols_agrupacion + ['prediccion_proxima_semana']
    resultados = ultimos_datos[columnas_resultado].copy()
    resultados['prediccion_proxima_semana'] = np.ceil(resultados['prediccion_proxima_semana']).astype(int)
    
    return resultados