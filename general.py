"""
Random generator to determine the powers and pairs
"""
import numpy as np
import numpy.random as npr
import argparse
from constants import *
from quantum import Gen, Show
from classical import approximate_distribution
from qiskit.quantum_info import Statevector, DensityMatrix
import matplotlib.pyplot as plt
from time import time
from generator import circuit_parameters

def parse_args():
    parser = argparse.ArgumentParser(description="Quantum & classical sparse IQP circuit sampling")
    parser.add_argument("N", type=int, help="Number of qubits")
    parser.add_argument("--runs", "-r", type=int, help="Number of runs", default=1)
    return parser.parse_args()

def time_transform(t):
    hours = int(t // 3600)
    minutes = int((t % 3600) // 60)
    seconds = int(t % 60)
    return f"{hours} hours, {minutes} minutes, {seconds} seconds"

if __name__ == '__main__':
    args = parse_args()
    N = args.N
    niter = args.runs
    PROB_CS = GAMMA * np.log(N) / N
    print(f"Prob(CS) = {PROB_CS}")
    q_times = np.zeros(niter)
    c_times = np.zeros(niter)

    for iter in range(niter):
        pairs_generated = False
        T_powers, CS_powers, CS_pairs = circuit_parameters(N, PROB_CS)

        while not pairs_generated:
            if len(CS_pairs) < 1 and N > 1:
                print("No CS pairs generated!")
                T_powers, CS_powers, CS_pairs = circuit_parameters(N, PROB_CS)
                continue
            else:
                pairs_generated = True

        # We start with the quantum circuit sampling
        circ = Gen(N, T_powers, CS_powers, CS_pairs, noisy=True)
        draw_circ = Gen(N, T_powers, CS_powers, CS_pairs)
        quantum_start = time()
        print("Starting quantum...")
        state = DensityMatrix.from_int(0, 2**N)
        state = state.evolve(circ)
        print("Quantum done.")
        quantum_elapsed = time() - quantum_start
        q_times[iter] = quantum_elapsed

        fig0, ax = plt.subplots()
        draw_circ.draw(output='mpl', ax=ax)

        # Save the circuit diagram
        fig0.savefig(f'Circuit_{N}QB_{iter}.jpeg')

        # Next we do the classical circuit sampling
        classical_start = time()
        print("Starting classical...")
        classical_distribution = approximate_distribution(N, T_powers, CS_powers, CS_pairs)
        print("Classical done.")
        classical_elapsed = time() - classical_start
        c_times[iter] = classical_elapsed

        np.save(f"QTIMES_{N}QB.npy", q_times)
        np.save(f"CTIMES_{N}QB.npy", c_times)
        print(f"[Iter {iter}] Saved quantum and classical times.")

        print(f"[Iter {iter}] QUANTUM TIME: {time_transform(quantum_elapsed)}")
        print(f"[Iter {iter}] CLASSICAL TIME: {time_transform(classical_elapsed)}")

        fig, axs = plt.subplots(1, 2)
        x = range(2**N)

        # Plot the first bar graph
        axs[0].bar(x, state.probabilities(), color='black')
        axs[0].set_title('IQP Circuit')
        axs[0].set_xlabel(f"Time = {round(quantum_elapsed, 2)} s")

        # Plot the second bar graph
        axs[1].bar(x, classical_distribution, color='orange')
        axs[1].set_title('Classical simulator')
        axs[1].set_xlabel(f"Time = {round(classical_elapsed, 2)} s")

        if iter == 0:
            plt.suptitle(fr"n = {N}, $\delta$ = {DELTA}, $\epsilon$ = {EPSILON}, scaling = {round(1+ELL, 3)}")

        # Adjust layout
        plt.tight_layout()

        plt.savefig(f"Compare_{N}QB_{iter}.jpeg")
        

    
    
    