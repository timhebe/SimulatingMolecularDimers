import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from uncertainties import ufloat
from qutip import *

# Function definitions
def draw_dipole(ax, center, angle, length=1.0, color='b', lw=2):
    dx = length * np.cos(np.radians(angle))
    dy = length * np.sin(np.radians(angle))
    ax.arrow(center[0] - dx / 2, center[1] - dy / 2, dx, dy,
             head_width=0.05, head_length=0.1, fc=color, ec=color, length_includes_head=True, lw=lw)
    return np.array([dx, dy])

def draw_vector(ax, start, end, color='r', lw=2):
    ax.annotate("", xy=end, xycoords='data', xytext=start, textcoords='data',
                arrowprops=dict(arrowstyle="->", color=color, lw=lw))
    return np.array(end) - np.array(start)

def interaction_energy(r, r_12, d_1, d_2, g_1, g_2):
    length_r_12 = np.linalg.norm(r_12)
    dipole_orientation = abs(np.dot(d_1, d_2) - 3 * np.dot(d_1, r_12 / length_r_12) * np.dot(d_2, r_12 / length_r_12))
    interaction_strength = 3 * np.sqrt(g_1 * g_2) / (8 * np.pi * (k_0 * n * r) ** 3)
    return dipole_orientation * interaction_strength

def spectrum(J_12, omega_1, omega_2, omega_rabi, gamma_1, gamma_2, gamma_12, x_axis_stretch):
    laser_freqs = x_axis_stretch * 2 * np.pi * np.linspace(-5.0, 5.0, 1200)
    excited_state_1 = []
    excited_state_2 = []
    state_J = []
    state_I = []
    state_U = []

    # collapse operators
    c_ops = [np.sqrt(gamma_1 * 1e-6) * S_1_minus,
             np.sqrt(gamma_2 * 1e-6) * S_2_minus,
             np.sqrt(gamma_12 / 2) * (S_1_minus + S_2_minus),
             np.sqrt(gamma_12 / 2) * (S_1_minus - S_2_minus)]

    # loop through different laser frequencies
    for omega_laser in laser_freqs:
        H = ((rho_eg_eg * (omega_1 - omega_laser) + rho_ge_ge * (omega_2 - omega_laser)
              + omega_rabi * (rho_eg_1 + rho_ge_1 + rho_eg_2 + rho_ge_2))
             + J_12 * (rho_e1g2_g1e2 + rho_g1e2_e1g2))

        rho_ss = steadystate(H, c_ops)

        excited_state_1.append(expect(rho_eg_eg, rho_ss))
        excited_state_2.append(expect(rho_ge_ge, rho_ss))
        state_J.append(expect(J * J.dag(), rho_ss))
        state_I.append(expect(I * I.dag(), rho_ss))
        state_U.append(expect(U * U.dag(), rho_ss))

    return laser_freqs, excited_state_1, excited_state_2, state_J, state_I, state_U

def calculate_spectrum(excited_state_1, excited_state_2, state_U):
    return [x + y + 2 * z for x, y, z in zip(excited_state_1, excited_state_2, state_U)]

# Constants
h = 6.626e-34
gamma_1 = 17e6
gamma_2 = 17e6
n = 1.5
k_0 = 2 * np.pi / 590e-9

# Streamlit layout
st.title("Dipole-Dipole Coupling - An Interactive Visualization and Simulation")

st.sidebar.header("Dipole Orientation")
angle1 = st.sidebar.slider('Angle 1 (°)', 0, 360, 90)
angle2 = st.sidebar.slider('Angle 2 (°)', 0, 360, 90)
x = st.sidebar.slider('x (nm)', -8.0, 8.0, -2.0)
y = st.sidebar.slider('y (nm)', -8.0, 8.0, 0.0)
show_vector = st.sidebar.checkbox(r'Show vector $\vec{r}_{12}$', False)

# Plot 1: Dipole Orientation
fig, ax = plt.subplots()
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_xlabel('Position (in nm)')
ax.set_ylabel('Position (in nm)')
ax.set_aspect('equal')
ax.grid(True)
ax.set_title('1. Dipole-Dipole Orientation')

d1_vector = draw_dipole(ax, (x / 2, y / 2), angle1, color='blue', lw=2)
d2_vector = draw_dipole(ax, (-x / 2, -y / 2), angle2, color='green', lw=2)
distance = np.sqrt(x ** 2 + y ** 2)
ax.text(-4.5, 4, f'distance: {distance:.2f} nm', verticalalignment='top', horizontalalignment='left')
if show_vector:
    r_vector = draw_vector(ax, (x / 2, y / 2), (-x / 2, -y / 2), lw=2)
else:
    r_vector = np.array((-x / 2, -y / 2)) - np.array((x / 2, y / 2))

st.pyplot(fig)

# Plot 2: Dimer Coupling Simulation
st.sidebar.header("Dimer Coupling Simulation")
gamma_1_val = st.sidebar.slider('γ1 (MHz)', 10, 100, 17)
gamma_2_val = st.sidebar.slider('γ2 (MHz)', 10, 100, 17)
log_scale_x = st.sidebar.checkbox('Log scale x-axis', False)
log_scale_y = st.sidebar.checkbox('Log scale y-axis', True)

interaction_energy_r = np.vectorize(lambda r: interaction_energy(r, r_vector, d1_vector, d2_vector, gamma_1_val * 1e6, gamma_2_val * 1e6))
distances = np.linspace(0.2e-9, 20e-9, 200)
J_12_initial = interaction_energy(distance * 1e-9, r_vector, d1_vector, d2_vector, gamma_1_val * 1e6, gamma_2_val * 1e6)

fig2, ax2 = plt.subplots()
ax2.plot(distances * 1e9, interaction_energy_r(distances), label='simulation')
ax2.errorbar(12, 0.95 * 1e9, xerr=2, yerr=0.1 * 1e9, fmt='o', color='blue', label='Hettich et al. 2002')
ax2.errorbar(22, 2.972 * 1e9, xerr=5, yerr=0.2 * 1e9, fmt='o', color='green', label='Trebbia et al. 2022')
ax2.errorbar(15, 1.02 * 1e9, xerr=5, yerr=0.1 * 1e9, fmt='o', color='red', label='Lange et al. 2024')
ax2.scatter([distance], [J_12_initial], c='C0')
ax2.text(1, 1e13, r'$J_{12} \approx $' + f'{J_12_initial * 1e-9:.1f} GHz')
gamma_12_initial = abs(np.sqrt(gamma_1_val * 1e6 * gamma_1_val * 1e6) * np.dot(d1_vector, d2_vector))
ax2.text(1, 1e12, r'$\Gamma_{12} \approx $' + f'{gamma_12_initial * 1e-6:.1f} MHz')
ax2.set_xlabel('Distance (nm)')
ax2.set_ylabel(r'$|J_{12}|$ (in Hz)')
ax2.set_yscale('log' if log_scale_y else 'linear')
ax2.set_xscale('log' if log_scale_x else 'linear')
ax2.grid(True)
ax2.legend(loc="best")
ax2.set_title('2. Dipole-Dipole Coupling Strength')
st.pyplot(fig2)

# Plot 3: Dimer Spectra Simulation
st.sidebar.header("Dimer Spectra Simulation")
omega_2_val = st.sidebar.slider('ω2 (MHz)', 1, 20, 10)
omega_rabi_val = st.sidebar.slider('Rabi frequency (MHz)', 1, 20, 10)
x_axis_stretch_val = st.sidebar.slider('Laser Frequency Axis Stretch', 0.5, 2.0, 1.0)

# Parameters
J_12 = J_12_initial
gamma_12 = gamma_12_initial * 1e-9  # conversion to GHz
omega_1 = 2 * np.pi * 0.0  # Rabi frequency of system 1 (in GHz)
omega_2 = 2 * np.pi * 1.0  # Rabi frequency of system 2 (in GHz)
omega_rabi = 2 * np.pi * 0.01  # driving laser intensity

x_axis_stretch = 4.0

# excited and ground state
e = basis(2, 1)
g = basis(2, 0)

# mixing angle
theta = 0.5 * np.arctan2(J_12, 0.5 * (omega_2 - omega_1))

# basis in the coupled system of the molecules (see PhD thesis Hettich 2002, page 65)
J = np.sin(theta) * tensor(e, g) + np.cos(theta) * tensor(g, e)
I = np.cos(theta) * tensor(e, g) - np.sin(theta) * tensor(g, e)
U = tensor(e, e)

# projectors for molecule 1 and 2
rho_eg_eg = tensor(e * e.dag(), qeye(2))
rho_ge_ge = tensor(qeye(2), e * e.dag())
eg_operator = e * g.dag()
rho_eg_1 = tensor(eg_operator, qeye(2))
rho_eg_2 = tensor(qeye(2), eg_operator)
ge_operator = g * e.dag()
rho_ge_1 = tensor(ge_operator, qeye(2))
rho_ge_2 = tensor(qeye(2), ge_operator)
rho_e1g2_g1e2 = tensor(e, g) * tensor(g, e).dag()
rho_g1e2_e1g2 = tensor(g, e) * tensor(e, g).dag()

# dipole raising and lowering operators
S_1_plus = tensor(e * g.dag(), qeye(2))
S_2_plus = tensor(qeye(2), e * g.dag())
S_1_minus = tensor(g * e.dag(), qeye(2))
S_2_minus = tensor(qeye(2), g * e.dag())

# Simulate spectrum
laser_freqs, excited_state_1, excited_state_2, state_J, state_I, state_U = spectrum(J_12, omega_1, omega_2_val * 1e6, omega_rabi_val * 1e6, gamma_1, gamma_2, gamma_12, x_axis_stretch_val)
spectrum_aggregated = calculate_spectrum(excited_state_1, excited_state_2, state_U)

fig3, ax3 = plt.subplots()
ax3.plot(laser_freqs, spectrum_aggregated)
ax3.set_xlabel('Laser frequency (MHz)')
ax3.set_ylabel('Aggregated population')
ax3.set_title('3. Dimer Spectra Simulation')
ax3.grid(True)
st.pyplot(fig3)
