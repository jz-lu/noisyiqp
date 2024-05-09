"""
Qiskit-based classical implementation of sparse IQP sampling.
"""
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, DensityMatrix
from qiskit_aer.noise import depolarizing_error
from qiskit.visualization import plot_histogram
import numpy as np
import numpy.random as npr
import argparse
import matplotlib.pyplot as plt
from constants import *
from generator import circuit_parameters

def parse_args():
    parser = argparse.ArgumentParser(description="Sparse IQP circuit sampling")
    parser.add_argument("N", type=int, help="Number of qubits")
    parser.add_argument("--runs", "-r", type=int, help="Number of runs", default=1)
    return parser.parse_args()

"""
Pseudocode algorithm for generating IQP circuits

Algo Gen_IQP_Circuit(n):
    C = blank_circuit(n)
    for i in range(n):
        C.add_gate(i, H)
        k = random({0, ..., 7})
        C.add_gate(i, T^k)
    for i, j in pairs(n):
        if random([0, 1]) < PROB_CS:
            k = random({0, ..., 3})
            C.add_gate(i, j, CS^k)
    for i in range(n):
        C.add_gate(i, H)
    return C
"""

def Gen(n, T_powers, CS_powers, CS_pairs, noisy=False, epsilon=EPSILON):
    # Create a blank quantum circuit with n qubits
    circ = QuantumCircuit(n)

    # Hadamard + T^k
    for i, k in enumerate(T_powers):
        circ.h(i)
        circ.p(k * PI / 4, i)
    
    # CS^k
    for ((i, j), k) in zip(CS_pairs, CS_powers):
        circ.cp(k * PI / 2, i, j)
    
    # Hadamard + maybe depolarizing error
    for i in range(n):
        circ.h(i)
        if noisy:
            depolarizing_gate = depolarizing_error(epsilon, 1)
            circ.append(depolarizing_gate, [i])
    
    return circ

def Show(circuit):
    # circuit.draw(output="mpl")
    print(circuit.draw())

if __name__ == '__main__':
    args = parse_args()
    N = args.N
    niter = args.runs
    PROB_CS = GAMMA * np.log(N) / N

    for i in range(niter):
        # Generate the circuit
        T_powers, CS_powers, CS_pairs = circuit_parameters(N, PROB_CS)
        circ = Gen(N, T_powers, CS_powers, CS_pairs, noisy=True)
        draw_circ = Gen(N, T_powers, CS_powers, CS_pairs)

        state = DensityMatrix.from_int(0, 2**N)
        state = state.evolve(circ)
        plt.clf()
        fig, axes = plt.subplots(1, 2)
        draw_circ.draw(output='mpl', ax=axes[0])
        axes[1].bar(range(2**N), state.probabilities())
        plt.suptitle(f"IQP Distribution, n = {N}")
        plt.tight_layout()
        plt.savefig(f"QDIST_{N}QB_{i}.jpeg")