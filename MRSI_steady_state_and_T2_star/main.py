import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sinus Slider", layout="centered")
st.title("MR Signal Influence")

st.markdown(
    "**This shows the influence of the steady state and the T2\\* decay to an example FID.** "
    "<br><br>Use the sidebar sliders to change **TR**, **α**, **T1**, **T2\\*** and " 
    "**TE** and see the impact on the time-domain signal and FFT.<br>" 
    "<br> 1. Plot: Shows the steady state due to repeated RF-excitation and T1 decay (sin(α) for measurable transverse signal)"
    "<br> 2. Plot: Shows the T2* decay after Echo Time (TE) and immediately after the RF excitation (thus offset via TE)"
    "<br> 3. Plot: Shows the steady state and T2* decay after TE offset"
    "<br> 4. Plot: Shows the spectrum of the seignal of plot 3", 
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("Controls")

    amplitude = 1
    frequency = st.slider("Frequency (Hz)", min_value=0.0, max_value=1000.0, value=500.0, step=1.0)

    TR = st.slider("TR (sec)", min_value=0.0, max_value=5.0, value=1.0, step=0.001)
    alpha = st.slider("alpha (°)", min_value=0, max_value=90, value=30, step=1)
    T1 = st.slider("T1 (sec)", min_value=0.0, max_value=5.0, value=1.2, step=0.1)
    T2_star = st.slider("T2* (sec)", min_value=0.0, max_value=0.1, value=0.034, step=0.001)

    TE = st.slider("TE (miliseconds)", min_value=0.0, max_value=30.0, value=12.0, step=0.1)

# Time vector
t = np.linspace(0, 0.2, 2000, endpoint=False)

# Sinus as basis signal
y = amplitude * np.sin(2 * np.pi * frequency * t)

# Alpha is in degrees -> convert to radians for sin/cos
a = np.deg2rad(alpha)

# safety for sliders starting at 0 (prevents divide-by-zero)
eps = 1e-12
T1_safe = max(T1, eps)
T2_safe = max(T2_star, eps)
TE_s = TE / 1000.0  # ms -> s

den = 1 - (np.cos(a)) * np.exp(-TR / T1_safe)
den = den if abs(den) > eps else eps

# 1) Just steady state
y_t1_steady = y * (np.sin(a) * (1 - np.exp(-TR / T1_safe))) / den
# 2) T2* decay with TE offset
y_t2_decay_with_TE = y * np.exp(-TE_s / T2_safe) * np.exp(-t / T2_safe)
# 3) T2* decay without TE offset
y_t2_decay_without_TE = y * np.exp(-t / T2_safe)
# 4) Steady state and T2* decay after TE offset
y_t1_steady_t2_decay_with_TE = y_t1_steady * np.exp(-TE_s / T2_safe) * np.exp(-t / T2_safe)

## Plots
fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(9, 8), constrained_layout=True)

ax[0].plot(t, y_t1_steady)
ax[0].set_xlabel("t (s)")
ax[0].set_ylabel("Signal")
ax[0].set_ylim(-1, 1)
ax[0].grid(True)
title_0 = (
    r"Steady-state: $S_0\cdot \sin(\alpha) \cdot\frac{\left(1-e^{-TR/T_1}\right)}"
    r"{1-\cos(\alpha)\,e^{-TR/T_1}}$"
)
ax[0].set_title(title_0)

ax[1].plot(t, y_t2_decay_without_TE, label="without TE")
ax[1].plot(t, y_t2_decay_with_TE, label="with TE")
ax[1].legend()
ax[1].set_ylim(-1, 1)
title_1 = (
    r"$T2^*$ decay (without TE): $S_0\,e^{-t/T_2^*}$"
    "\n"
    r"$T2^*$ decay (with TE): $S_0\,e^{-TE/T_2^*}\,e^{-t/T_2^*}$"
)
ax[1].set_title(title_1)
ax[1].grid(True)
ax[1].set_xlabel("t (s)")
ax[1].set_ylabel("Signal")


title_2 = "Steady State $\cdot$ T2* decay with TE"
ax[2].set_title(title_2)
ax[2].plot(t, y_t1_steady_t2_decay_with_TE)
ax[2].set_ylim(-1, 1)
ax[2].grid(True)
ax[2].set_xlabel("t (s)")
ax[2].set_ylabel("Signal")

# calculate the frequency vector
dt = t[1] - t[0]
nfft = len(y_t1_steady_t2_decay_with_TE)

Y = np.fft.rfft(y_t1_steady_t2_decay_with_TE, n=nfft)
f = np.fft.rfftfreq(nfft, d=dt)
A = np.abs(Y)

title_3 = "|FFT(Steady State $\cdot$ T2* decay with TE)|"
ax[3].plot(f, np.abs(A))
ax[3].set_title(title_3)
ax[3].set_ylim(bottom=0)
ax[3].set_ylim(top=400)
ax[3].set_xlabel("Frequency [Hz]")
ax[3].set_ylabel("|FFT|")
ax[3].grid(True, axis="y")

st.pyplot(fig)
