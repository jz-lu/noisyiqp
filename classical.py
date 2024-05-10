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

print(f"ell = {round(ELL, 3)}, scaling = {round(3 * ELL + 3, 3)}")

def parse_args():
    parser = argparse.ArgumentParser(description="Sparse IQP circuit sampling")
    parser.add_argument("N", type=int, help="Number of qubits")
    parser.add_argument("--sample", "-s", action="store_true", help="Sample from the distribution instead of output it")
    parser.add_argument("--nsamples", "-n", type=int, default=1, help="Number of samples")
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

def prepend_vector_to_rows(matrix, vector, append=False):
    """
    Prepends `vector` repeatedly to each row of `matrix`. 
    If `append` is True, then appends instead of prepends.
    Example:
        matrix = [[1, 2, 3], 
                  [4, 5, 6]]
        vector = [7, 8]
        output = [[7, 8, 1, 2, 3]
                  [7, 8, 4, 5, 6]]
    """
    # Repeat the vector for each row of the matrix
    repeated_vector = np.tile(vector, (matrix.shape[0], 1))
    # Concatenate the matrix and the repeated vector along the second axis
    result = None
    if append:
        result = np.concatenate((matrix, repeated_vector), axis=1)
    else:
        result = np.concatenate((repeated_vector, matrix), axis=1)
    return result

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
    zeta = 5 * n # corresponding to failure probability 2^(-5n)
    eta = int(n ** (ELL) / DELTA**2)
    repetitions = npr.randint(2, size=(zeta, eta, n)) # zeta x eta matrix of bitstrings (not same eta and zeta as in paper!)

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

def cumulative_sum(n, y, T_powers, CS_powers, CS_pairs):
    """
    Calculate Sy = \sum_{x: x[:k] == y} \widetilde{p}_\epsilon(x).
    """
    assert 0 <= len(y) <= n, f"y = {y} is too long!"
    q = lambda s: approximate_fourier(s, n, T_powers, CS_powers, CS_pairs)
    k = len(y)
    if k == 0: # empty string case, sum whole distribution
        return q(np.zeros(n)) * 2**(n) # just Fourier coefficient at s = 0, times 2^n
    else:
        s_vals = wt_bdd_strings(k, ELL) # (some) L' x k
        padded_s_vals = prepend_vector_to_rows(s_vals, np.zeros(n-k), append=True)
        s_dot_y = s_vals @ y
        phases = (-1) ** s_dot_y # L' x 1
        fourier_coeffs = np.apply_along_axis(q, axis=1, arr=padded_s_vals) # L' x 1
        wts = (1-EPSILON) ** np.sum(s_vals, axis=1) # L' x 1 exponentiated Hamming weights
        return np.real(np.dot(phases, fourier_coeffs * wts)) * 2**(n-k)

def marginal_2_sample(n, nsamples, T_powers, CS_powers, CS_pairs):
    """
    Algorithm to approximately sample from the distribution, given cumulative marginals
    Paramteres: 
        * n = number of qubits
        * nsamples = number of samples
        * distribution = 
    
    returns: vector of size `nsamples`, the approximate samples from the distribution
    """
    samples = np.zeros((nsamples, n))
    cumsum = lambda y: cumulative_sum(n, y, T_powers, CS_powers, CS_pairs)
    print("ready")
    total = cumsum([]) # sum of entire approximate distribution

    for idx in range(nsamples):
        print(f"Sampling iteration [{idx+1}]")
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
    
    return samples.astype(int)

if __name__ == '__main__':
    args = parse_args()
    N = args.N
    PROB_CS = GAMMA * np.log(N) / N
    T_powers, CS_powers, CS_pairs = circuit_parameters(N, PROB_CS)
    if len(CS_pairs) < 1:
        print(CS_pairs)
        print("No CS pairs generated!") # edge case: for very small qubits: post-select on having at least 1 CS gate
        exit(0)

    if args.sample:
        # Return a sample from the distribution
        print("Starting the sampling")
        nsamples = args.nsamples
        samples = marginal_2_sample(N, nsamples, T_powers, CS_powers, CS_pairs)
        np.save(f"ClassicalSamples_{N}QB.npy", samples)
        print("===== Samples =====")
        for i, sample in enumerate(samples):
            print(f"({i+1}) {sample}")
        print("===================")

    else:
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
        plt.bar(range(2**N), distribution)
        plt.show()

    
