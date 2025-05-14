#Ejemplo para desplegar en streamlit
# Se despliega así: `streamlit run main.py` dentro de la ruta del archivo sstreamlit


#Este archivo es una prueba para mostrar como se puede montar una pagina web con streamlit
# Streamlit es una libreria de python que permite crear aplicaciones web de manera sencilla y rapida

# Para instalar streamlit, puedes usar el siguiente comando:
# pip install streamlit

# Para ejecutar este archivo, puedes usar el siguiente comando:
# streamlit run <nombredelarchivo>.py

# Importar librerías

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np


st.title("📊 Aplicación Interactiva de Gráficos")

st.subheader("Como montar una pagina Streamlit?")

iframe_code = """
<iframe width="100%" height="400" src="https://s3-us-west-2.amazonaws.com/assets.streamlit.io/videos/hero-video.mp4"
title="Streamlit Tutorial" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture"
allowfullscreen></iframe>
"""

st.components.v1.html(iframe_code, height=420, scrolling=False)


# Graficos de streamlit
frecuencia = st.slider("Frecuencia", 1.0, 10.0, 2.0, step=0.1)
amplitud = st.slider("Amplitud", 0.1, 5.0, 1.0, step=0.1)

# Generar datos
x = np.linspace(0, 10, 500)
y = amplitud * np.sin(frecuencia * x)

# Crear gráfico
fig, ax = plt.subplots()
ax.plot(x, y, color="tomato", linewidth=2)
ax.set_title(f"y = {amplitud} × sin({frecuencia} × x)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(True)

# Mostrar gráfico
st.pyplot(fig)