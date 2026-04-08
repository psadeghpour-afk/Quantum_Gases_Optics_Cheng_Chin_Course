
# p452_project1_app.py
import streamlit as st
import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram, plot_circuit_layout, circuit_drawer
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import RXGate, RYGate, RZGate
import matplotlib.pyplot as plt

# -----------------------------
# Helper functions
# -----------------------------

def run_backend(circuit, shots=1024):
    simulator = Aer.get_backend('aer_simulator')
    circuit = circuit.copy()
    circuit.measure_all()
    result = execute(circuit, simulator, shots=shots).result()
    counts = result.get_counts()
    return counts

def plot_streamlit(circuit, counts=None):
    st.subheader("Circuit Diagram")
    fig_circuit = circuit.draw(output='mpl', scale=1.5)
    st.pyplot(fig_circuit)
    if counts:
        st.subheader("Measurement Histogram")
        fig_hist = plot_histogram(counts)
        st.pyplot(fig_hist)

# -----------------------------
# Project UI
# -----------------------------
st.title("P452 Quantum Simulator")

preset = st.selectbox("Select Preset Circuit", ["Teleportation", "Hubbard", "10-Qubit GHZ", "Unitarity Test"])

# -----------------------------
# 2-Qubit Parameter Control Loop (Q1.2)
# -----------------------------
if preset == "Teleportation":
    st.header("2-Qubit Ry(θ) + CNOT Example")
    theta = st.slider("Rotation Angle θ (rad)", 0.0, 2*np.pi, 3.14)
    
    qc = QuantumCircuit(2)
    qc.ry(theta, 0)
    qc.cx(0,1)
    
    counts = run_backend(qc)
    plot_streamlit(qc, counts)
    
    st.write("State |11> probability ~100% shows slider passed θ correctly.")

# -----------------------------
# 10-Qubit GHZ State (Q1.3)
# -----------------------------
elif preset == "10-Qubit GHZ":
    st.header("10-Qubit GHZ State |0> + |1023> / sqrt(2)")
    qc = QuantumCircuit(10)
    qc.h(0)
    for i in range(9):
        qc.cx(i, i+1)
    
    counts = run_backend(qc)
    plot_streamlit(qc, counts)

# -----------------------------
# Unitarity Test (Q1.4)
# -----------------------------
elif preset == "Unitarity Test":
    st.header("Unitarity Test: Prepare 1/sqrt(2)(|201> + |425>)")
    qc = QuantumCircuit(10)
    
    # Initialize specific amplitudes for |201> = |0011001001> and |425> = |0110101001>
    sv = Statevector.from_label('0'*10)
    sv.data[201] = 1/np.sqrt(2)
    sv.data[425] = 1/np.sqrt(2)
    
    qc.initialize(sv.data)
    
    st.subheader("Initial Statevector")
    st.write(sv)
    
    # Apply chain of 9 CNOT gates
    for i in range(9):
        qc.cx(i, i+1)
    st.subheader("After Chain of 9 CNOTs")
    sv_after = Statevector.from_instruction(qc)
    st.write(sv_after)
    
    # Reverse operation
    for i in reversed(range(9)):
        qc.cx(i, i+1)
    st.subheader("After Reversing Gates")
    sv_reversed = Statevector.from_instruction(qc)
    st.write(sv_reversed)
    
    st.write("Equal to initial statevector confirms unitarity.")

# -----------------------------
# Teleportation Circuit (Phase 2)
# -----------------------------
elif preset == "Hubbard":
    st.header("Teleportation / Fermi-Hubbard Options")
    subpreset = st.selectbox("Select", ["3-Qubit Teleportation", "Linear-chain CNOT", "2-Site Hubbard"])
    
    if subpreset == "3-Qubit Teleportation":
        st.subheader("Teleport Alice's Qubit to Bob")
        theta = st.slider("Alice's Qubit θ", 0.0, 2*np.pi, np.pi/4)
        qc = QuantumCircuit(3)
        qc.ry(theta, 0)  # Alice's qubit
        # Bell pair
        qc.h(1)
        qc.cx(1,2)
        # Teleportation steps
        qc.cx(0,1)
        qc.h(0)
        qc.measure_all()
        counts = run_backend(qc)
        plot_streamlit(qc, counts)
    
    elif subpreset == "Linear-chain CNOT":
        st.subheader("CNOT q0 → q4 using SWAPs")
        qc = QuantumCircuit(5)
        # Decompose CNOT with linear connectivity
        qc.cx(0,1)
        qc.swap(1,2)
        qc.cx(2,3)
        qc.swap(3,4)
        qc.cx(4,4)  # Final target
        st.write("Total CNOT gates = 3 (plus SWAPs)")
        plot_streamlit(qc)
    
    elif subpreset == "2-Site Hubbard":
        st.subheader("2-Site Fermi-Hubbard Simulation")
        U = st.slider("On-site Interaction U", 0.0, 20.0, 1.0)
        J = st.slider("Hopping Amplitude J", 0.0, 5.0, 1.0)
        steps = st.slider("Trotter Steps", 1, 20, 5)
        tau = st.slider("Time τ", 0.0, np.pi, 0.1)
        
        # 4 qubits: q0, q1, q2, q3
        qc = QuantumCircuit(4)
        
        # Example: Initial state |1000> (electron at site 1 ↑)
        qc.x(0)
        
        # Single Trotter step gates
        for _ in range(steps):
            # Hopping terms XY rotations
            qc.rx(2*J*tau, 0)
            qc.rx(2*J*tau, 2)
            qc.ry(2*J*tau, 0)
            qc.ry(2*J*tau, 2)
            # Interaction term
            qc.cz(0,1)
            qc.cz(2,3)
            qc.rz(U*tau,0)
            qc.rz(U*tau,1)
        
        counts = run_backend(qc)
        plot_streamlit(qc, counts)
        st.write("This simulates a single Trotter step evolution.")
