import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.cm as cm


# Cargar datos
df_items = pd.read_csv("./Olist_Data/olist_order_items_dataset.csv")
df_reviews = pd.read_csv("./Olist_Data/olist_order_reviews_dataset.csv")
df_orders = pd.read_csv("./Olist_Data/olist_orders_dataset.csv")
df_products = pd.read_csv("./Olist_Data/olist_products_dataset.csv")
df_sellers = pd.read_csv("./Olist_Data/olist_sellers_dataset.csv")
df_payments = pd.read_csv("./Olist_Data/olist_order_payments_dataset.csv")
df_customers = pd.read_csv("./Olist_Data/olist_customers_dataset.csv")
df_category = pd.read_csv("./Olist_Data/product_category_name_translation.csv")
estado_nombres = {
    'AC': 'Acre', 'AL': 'Alagoas', 'AM': 'Amazonas', 'AP': 'Amapá', 'BA': 'Bahia', 'CE': 'Ceará',
    'DF': 'Distrito Federal', 'ES': 'Espírito Santo', 'GO': 'Goiás', 'MA': 'Maranhão', 'MG': 'Minas Gerais',
    'MS': 'Mato Grosso do Sul', 'MT': 'Mato Grosso', 'PA': 'Pará', 'PB': 'Paraíba', 'PE': 'Pernambuco',
    'PI': 'Piauí', 'PR': 'Paraná', 'RJ': 'Rio de Janeiro', 'RN': 'Rio Grande do Norte',
    'RO': 'Rondônia', 'RR': 'Roraima', 'RS': 'Rio Grande do Sul', 'SC': 'Santa Catarina',
    'SE': 'Sergipe', 'SP': 'São Paulo', 'TO': 'Tocantins'
}
df_customers['customer_state'] = df_customers['customer_state'].map(estado_nombres)
# Merge
df = df_orders.merge(df_items, on='order_id', how='left')
df = df.merge(df_payments, on='order_id', how='left', validate='m:m')
df = df.merge(df_reviews, on='order_id', how='left')
df = df.merge(df_products, on='product_id', how='left')
df = df.merge(df_customers, on='customer_id', how='left')
df = df.merge(df_sellers, on='seller_id', how='left')
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])
df['order_estimated_delivery_date'] = pd.to_datetime(df['order_estimated_delivery_date'])

# Sidebar - Navegación y Filtros
st.sidebar.title(" Navegación")
seccion = st.sidebar.radio("Selecciona una sección", [
    "Resumen General",
    "Top Ciudades",
    "Retrasos en Entregas",
    "Reviews",
    "Productos Más Vendidos"
])

min_fecha = df['order_purchase_timestamp'].min().date()
max_fecha = df['order_purchase_timestamp'].max().date()

fecha_inicio, fecha_fin = st.sidebar.slider(
    "Selecciona un rango de fechas:",
    min_value=min_fecha,
    max_value=max_fecha,
    value=(min_fecha, max_fecha),
    format="YYYY-MM-DD"
)

min_clientes = st.sidebar.slider("Número mínimo de clientes por ciudad:", min_value=0, max_value=500, value=50)
estados_disponibles = sorted(df['customer_state'].dropna().unique())
estado_seleccionado = st.sidebar.multiselect("Filtrar por estado(s):", estados_disponibles, default=estados_disponibles)

# Aplicar filtros
df_filtrado = df[(df['order_purchase_timestamp'] >= pd.to_datetime(fecha_inicio)) & 
                 (df['order_purchase_timestamp'] <= pd.to_datetime(fecha_fin))]
df_filtrado = df_filtrado[df_filtrado['customer_state'].isin(estado_seleccionado)]

# Agrupaciones
clientes_agrupados = df_filtrado.drop_duplicates(subset='customer_unique_id').groupby(['customer_state', 'customer_city']).agg(num_clientes=('customer_unique_id', 'count')).reset_index()
clientes_agrupados = clientes_agrupados[clientes_agrupados['num_clientes'] >= min_clientes]
pedidos_agrupados = df_filtrado.groupby(['customer_state', 'customer_city']).agg(num_pedidos=('order_id', 'nunique')).reset_index()
tabla_completa = pd.merge(clientes_agrupados, pedidos_agrupados, on=['customer_state', 'customer_city'])
total_pedidos = tabla_completa['num_pedidos'].sum()
tabla_completa['porc_pedidos'] = (tabla_completa['num_pedidos'] / total_pedidos * 100).round(2)
tabla_completa['pedidos_por_cliente'] = (tabla_completa['num_pedidos'] / tabla_completa['num_clientes']).round(2)
top_ciudades = tabla_completa.sort_values('num_clientes', ascending=False).head(20)
top_ciudades1 = clientes_agrupados.sort_values('num_clientes', ascending=False).head(20)

# Gráfico comparativo
fig, ax1 = plt.subplots(figsize=(20, 10))
sns.barplot(data=top_ciudades, x='customer_city', y='num_clientes', color='green', label='Clientes', ax=ax1)
sns.barplot(data=top_ciudades, x='customer_city', y='num_pedidos', color='steelblue', alpha=0.7, label='Pedidos', ax=ax1)
ax2 = ax1.twinx()
sns.lineplot(data=top_ciudades, x='customer_city', y='pedidos_por_cliente', color='red', marker='o', label='Pedidos/Cliente', ax=ax2)
ax1.set_title("Clientes, Pedidos y Ratio de Pedidos por Cliente (Top 20 Ciudades)")
ax1.set_xticks(range(len(top_ciudades))) 
ax1.set_xticklabels(top_ciudades['customer_city'], rotation=45, ha='right')
ax1.set_xlabel("Ciudad")
ax1.set_ylabel("Número de clientes / pedidos")
ax2.set_ylabel("Pedidos por cliente")
fig.legend(loc='upper left')
plt.tight_layout()

# Gráfico por estado
clientes_por_estado = (clientes_agrupados.groupby('customer_state')['num_clientes'].sum().sort_values(ascending=False).head(5).index.tolist())
top_3_ciudades_por_estado = (
    clientes_agrupados[clientes_agrupados['customer_state'].isin(clientes_por_estado)]
    .sort_values(['customer_state', 'num_clientes'], ascending=[True, False])
    .groupby('customer_state').head(3)
)
plt_estado = plt.figure(figsize=(12, 8))
sns.barplot(data=top_3_ciudades_por_estado, x='num_clientes', y='customer_city', hue='customer_state', dodge=False)
plt.title('Top 3 ciudades con más clientes en los 5 estados con más clientes totales')
plt.xlabel('Número de clientes')
plt.ylabel('Ciudad')
plt.tight_layout()

# Gráfico retrasos
entregados = df_filtrado[df_filtrado['order_delivered_customer_date'].notna()].copy()
entregados['dias_retraso'] = (entregados['order_delivered_customer_date'] - entregados['order_estimated_delivery_date']).dt.days
entregados['llegó_tarde'] = entregados['dias_retraso'] > 0
retrasos_por_ciudad = (
    entregados.groupby('customer_city')
    .agg(pedidos_totales=('order_id', 'count'), pedidos_tarde=('llegó_tarde', 'sum'),
         retraso_medio_dias=('dias_retraso', lambda x: x[x > 0].mean()))
)
retrasos_por_ciudad['porcentaje_tarde'] = (retrasos_por_ciudad['pedidos_tarde'] / retrasos_por_ciudad['pedidos_totales']) * 100
estado_por_ciudad = df_filtrado.groupby(['customer_city', 'order_status']).size().unstack(fill_value=0)
retrasos_por_ciudad = retrasos_por_ciudad.join(estado_por_ciudad, how='left')

def diagnostico_avanzado(row):
    if 'canceled' in row and row['canceled'] > 0.2 * row['pedidos_totales']:
        return "Alta tasa de cancelaciones"
    elif 'unavailable' in row and row['unavailable'] > 0.1 * row['pedidos_totales']:
        return "Problemas de disponibilidad"
    elif row['porcentaje_tarde'] > 50 and row['retraso_medio_dias'] > 5:
        return "Problemas logísticos"
    elif row['porcentaje_tarde'] > 30:
        return "Zona con retrasos frecuentes"
    elif row['retraso_medio_dias'] > 10:
        return "Revisar excepciones"
    elif row['porcentaje_tarde'] > 10:
        return "Retrasos puntuales"
    else:
        return "Sin problemas destacados"

retrasos_por_ciudad['diagnostico'] = retrasos_por_ciudad.apply(diagnostico_avanzado, axis=1)
retrasos_filtrados = retrasos_por_ciudad[retrasos_por_ciudad['pedidos_totales'] >= 100]
top_cities = retrasos_filtrados.sort_values(by='pedidos_totales', ascending=False).head(10)

norm = mcolors.Normalize(vmin=top_cities['porcentaje_tarde'].min(), vmax=top_cities['porcentaje_tarde'].max())
cmap = cm.get_cmap('RdYlGn_r')
colors = [cmap(norm(val)) for val in top_cities['porcentaje_tarde']]
fig_retrasos, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(top_cities.index, top_cities['porcentaje_tarde'], color=colors)
ax.set_xlabel('% de pedidos entregados tarde')
ax.set_title('Top 10 ciudades con más pedidos y % de retrasos')
ax.invert_yaxis()
for i, (valor, ciudad) in enumerate(zip(top_cities['porcentaje_tarde'], top_cities.index)):
    ax.text(valor + 0.5, i, f"{top_cities.loc[ciudad, 'pedidos_totales']} pedidos", va='center')
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('% de pedidos entregados tarde')
plt.tight_layout()

# Mostrar según sección
if seccion == "Resumen General":
    st.subheader("📍 Top 20 Ciudades con Más Clientes")
    st.dataframe(top_ciudades1)
    st.subheader("📈 Clientes por Estado")
    st.pyplot(plt_estado)

elif seccion == "Top Ciudades":
    st.subheader("📋 Tabla Completa de Ciudades, Clientes y Pedidos")
    st.dataframe(top_ciudades)
    st.subheader("📈 Gráfico Comparativo")
    st.pyplot(fig)

elif seccion == "Retrasos en Entregas":
    st.subheader("📋 Tabla de retrasos")
    st.dataframe(top_cities)
    st.subheader("📈 Gráfico de Retrasos")
    st.pyplot(fig_retrasos)


elif seccion == "Reviews":
    st.subheader(" Reviews")

    # Contar valores faltantes en 'review_comment_message'
    valores_nulos_reviws = df_filtrado['review_comment_message'].isna().sum()
    # Eliminar duplicados en 'review_comment_message', manteniendo el primero
    df_reviws = df_filtrado.drop_duplicates(subset=['review_comment_message'], keep='first').reset_index(drop=True)
    

    entregados2 = df_reviws[df_reviws['order_delivered_customer_date'].notna()].copy()

    # Calcular días de retraso
    entregados2['dias_retraso'] = (entregados2['order_delivered_customer_date'] - entregados2['order_estimated_delivery_date']).dt.days
    entregados2['llegó_tarde'] = entregados2['dias_retraso'] > 0
    # Solo reviews completas y entregadas a tiempo
    df_reviews_validas = entregados2[(entregados2['order_delivered_customer_date'].notna()) &(entregados2['dias_retraso'] <= 0) &  (entregados2['customer_state'].notna())]
    # Agrupamos por estado
    reviews_por_estado = df_reviews_validas.groupby('customer_state').agg(n_reviews=('review_id', 'count'),score_medio=('review_score', 'mean')).sort_values(by='n_reviews', ascending=False)


    fig, ax1 = plt.subplots(figsize=(14, 6))

    #número de reviews
    color1 = 'skyblue'
    ax1.bar(reviews_por_estado.index, reviews_por_estado['n_reviews'], color=color1, label='Número de reviews')
    ax1.set_ylabel('Número de reviews', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    # score medio
    ax2 = ax1.twinx()
    color2 = 'seagreen'
    ax2.plot(reviews_por_estado.index, reviews_por_estado['score_medio'], color=color2, marker='o', label='Score medio')
    ax2.set_ylabel('Score medio', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(3.5, 5)

    # general
    plt.title('Número de Reviews y Score Medio por Estado (Sin pedidos entregados tarde)')
    ax1.set_xticklabels(reviews_por_estado.index, rotation=45)
    fig.tight_layout()
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    st.pyplot(fig)
    st.subheader("📋 Tabla Reviews y Score medio")
    st.dataframe(reviews_por_estado)

elif seccion == "Productos Más Vendidos":
    st.subheader("📋 Tabla de Productos Más Vendidos")
    
    productos_mas_vendidos = df_filtrado.groupby('product_category_name')['order_item_id'].count().reset_index()
    productos_mas_vendidos.columns = ['Categoría', 'Unidades Vendidas']
    productos_mas_vendidos = productos_mas_vendidos.sort_values(by='Unidades Vendidas', ascending=False)

    ingresos = df_filtrado.groupby('product_category_name')['price'].sum().reset_index()
    ingresos.columns = ['Categoría', 'Ingresos Totales']
    productos_mas_vendidos = productos_mas_vendidos.merge(ingresos, on='Categoría', how='left')

    precio_prom = df_filtrado.groupby('product_category_name')['price'].mean().reset_index()
    precio_prom.columns = ['Categoría', 'Precio Promedio']
    productos_mas_vendidos = productos_mas_vendidos.merge(precio_prom, on='Categoría', how='left')

    vendedores = df_filtrado.groupby('product_category_name')['seller_id'].nunique().reset_index()
    vendedores.columns = ['Categoría', 'N° Vendedores Únicos']
    productos_mas_vendidos = productos_mas_vendidos.merge(vendedores, on='Categoría', how='left')

    ticket_prom = df_filtrado.groupby('product_category_name').apply(
        lambda x: x['price'].sum() / x['order_id'].nunique()
    ).reset_index(name='Ticket Promedio')
    ticket_prom.rename(columns={'product_category_name': 'Categoría'}, inplace=True)
    productos_mas_vendidos = productos_mas_vendidos.merge(ticket_prom, on='Categoría', how='left')

    st.dataframe(productos_mas_vendidos.head(20))

    st.subheader("📊 Gráfico de Categorías Más Vendidas")
    fig_cat, ax_cat = plt.subplots(figsize=(12, 8))
    top_n = 15
    sns.barplot(
        data=productos_mas_vendidos.head(top_n),
        y='Categoría',
        x='Unidades Vendidas',
        palette='viridis',
        ax=ax_cat
    )
    ax_cat.set_title(f"Top {top_n} Categorías con Más Unidades Vendidas")
    ax_cat.set_xlabel("Unidades Vendidas")
    ax_cat.set_ylabel("Categoría")
    plt.tight_layout()
    st.pyplot(fig_cat)

    