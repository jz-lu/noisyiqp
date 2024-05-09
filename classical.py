"""
Classical algorithm for sparse IQP sampling with depolarizing error.
Technique: Fourier analysis in Z2.

TODO: synchronize the quantum and classical ones with the same circuit parameters!
"""

import numpy as np
import numpy.random as npr
import argparse
import time
import matplotlib.pyplot as plt
from constants import *
from generator import circuit_parameters

print(f"ell = {ELL}")

def parse_args():
    parser = argparse.ArgumentParser(description="Sparse IQP circuit sampling")
    parser.add_argument("N", type=int, help="Number of qubits")
    return parser.parse_args()

def generate_bitstrings(n):
    """
    Enumerate all bitstrings of length `n`, in lexicographic order.
    """
    def generate_bitstrings_helper(bitstring):
        if len(bitstring) == n:
            yield bitstring
        else:
            yield from generate_bitstrings_helper(bitstring + '0')
            yield from generate_bitstrings_helper(bitstring + '1')

    bitstrings = generate_bitstrings_helper('')
    return np.array([[int(bit) for bit in bitstring] for bitstring in bitstrings])

def wt_bdd_strings(n, bound):
    """
    Enumerate all bitstrings of length `n`, whose weight is <= `bound`
    """
    def generate_bitstrings_helper(bitstring, weight, bound):
        if len(bitstring) == n:
            yield bitstring
        elif weight < bound:
            yield from generate_bitstrings_helper(bitstring + '0', weight, bound)
            yield from generate_bitstrings_helper(bitstring + '1', weight + 1, bound)
        elif weight == bound:
            yield from generate_bitstrings_helper(bitstring + '0', weight, bound)

    bitstrings = generate_bitstrings_helper('', 0, bound)
    return np.array([[int(bit) for bit in bitstring] for bitstring in bitstrings])

def f(x, T_powers, CS_powers, CS_pairs):
    """
    Evaluate the function f(x) = <x|D|x>, where D is specified by the input.

    Input (all lists should be np.array):
        * x: functional input, a computational basis vector, i.e. a length-n list of bits.
        * n: number of qubits.
        * T_powers: length-n list of powers of T from 0 to 7
        * CS_powers: random-length list of powers of CS from 0 to 3
        * CS_pairs: random-length (same as `CS_powers`) list of pairs with a CS.
    """
    # Basic idea is just to iterate through the list and add up all the phases, outputting a final phase.
    T_phase = (np.sum(x * T_powers) * PI / 4) % (2 * PI)
    x = x[CS_pairs]
    CS_phase = np.sum(x[:,0] * x[:,1] * CS_powers * PI / 2) % (2 * PI)
    return np.exp(1j * (T_phase + CS_phase))

def approximate_fourier(s, n, T_powers, CS_powers, CS_pairs):
    """
    Approximately calculate Fourier coefficients of IQP circuit output
    distribution, using the phase-counting function above.

    Inputs:
        * s (np.array): Fourier index
        * n: number of qubits
    """
    zeta = 5 * n
    eta = int(n ** (ELL) / DELTA**2)
    repetitions = npr.randint(2, size=(zeta, eta, n)) # zeta x eta matrix of bitstrings

    # Write a function that calculates f*(y) f(y + s) and vectorize it
    eval = lambda x: np.conjugate(f(x, T_powers, CS_powers, CS_pairs)) * f(x + s, T_powers, CS_powers, CS_pairs)
    evaluations = np.squeeze(np.apply_along_axis(eval, axis=2, arr=repetitions))
    assert evaluations.shape == (zeta, eta), f"{evaluations.shape} != {(zeta, eta)}"

    # Now take the median-of-means and return it
    means = np.mean(evaluations, axis=1)
    median_of_means = np.median(means)
    return median_of_means / 2**n

def approximate_distribution(n, T_powers, CS_powers, CS_pairs):
    """
    Calculate the marginals of an approximate distribution using
    its Z2 Fourier coefficients.
    """
    y_vals = generate_bitstrings(n) # 2^n x n
    s_vals = wt_bdd_strings(n, ELL) # L x n
    s_dot_y = y_vals @ s_vals.T # (2^n x n) times (n x L) -> (2^n x L)
    phases = (-1) ** s_dot_y
    wts = (1-EPSILON) ** np.sum(s_vals, axis=1) # L x 1 exponentiated Hamming weights
    q = lambda s: approximate_fourier(s, n, T_powers, CS_powers, CS_pairs)
    fourier_coeffs = np.apply_along_axis(q, axis=1, arr=s_vals)

    distribution = np.real(phases @ (fourier_coeffs * wts)) # (2^n x L) times (L x 1) -> (2^n x 1)
    return distribution

def cumulative_sum(fake_dist, y):
    """
    Calculate marginal events (cumulative sums) from the fake distribution
    """
    pass

def marginal_2_sample(n, nsamples, cumsum, total):
    """
    Algorithm to approximately sample from the distribution, given cumulative marginals
    Paramteres: 
        * n = number of qubits
        * nsamples = number of samples
        * cumsum = function calculating cumulative marginal sums prefixed by argument
        * total = sum of entire distribution
    
    returns: vector of size `nsamples`, the approximate samples from the distribution
    """
    samples = np.zeros((nsamples, n))
    for idx in range(nsamples):
        y = []
        Sy = total

        for i in range(n):
            Syz = np.array([cumsum(y + [i]) for i in range(2)])
            if Syz[0] < 0:
                y = y + [1]
            elif Syz[1] < 0:
                y = y + [0]
            else:
                y = y + [0] if npr.rand() < Syz[0] / Sy else y + [1]
            
            Sy = Syz[y[-1]] # update the cumsum
        
        samples[idx,:] = np.array(y)
    
    return samples

if __name__ == '__main__':
    args = parse_args()
    N = args.N
    PROB_CS = GAMMA * np.log(N) / N
    T_powers, CS_powers, CS_pairs = circuit_parameters(N, PROB_CS)
    if len(CS_pairs) < 1:
        print(CS_pairs)
        print("No CS pairs generated!")
        exit(0)

    # Get the distribution and plot it
    start_time = time.time()
    distribution = approximate_distribution(N, T_powers, CS_powers, CS_pairs)
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    print("Time taken:", hours, "hours,", minutes, "minutes,", seconds, "seconds")

    # Plot the distribution
    # plt.bar(range(2**N), distribution)
    # plt.show()

    
