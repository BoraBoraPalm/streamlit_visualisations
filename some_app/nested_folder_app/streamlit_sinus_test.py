import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sinus Slider", layout="centered")
st.title("Sinus-Plot mit Slider")

# Slider
#amplitude = st.slider("Amplitude", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
amplitude = 1
frequency = st.slider("Frequenz (Hz)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)

# Daten
t = np.linspace(0, 1, 1000)  # 1 Sekunde
y = amplitude * np.sin(2 * np.pi * frequency * t)

# Plot
fig, ax = plt.subplots()
ax.plot(t, y)
ax.set_xlabel("t (s)")
ax.set_ylabel("y")
ax.set_title(f"y(t) = {amplitude:.2f} · sin(2π · {frequency:.2f} · t)")
ax.set_ylim(-1,1)
ax.grid(True)

st.pyplot(fig)