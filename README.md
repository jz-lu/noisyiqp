# NoisyIQP
This is *NoisyIQP*, a tiny Python repository dedicated to the study of sparse instantaneous quantum polytime (IQP) 
circuits and their output distributions. Following the theoretical work by [Bremner, Montanaro, and Shepherd](https://quantum-journal.org/papers/q-2017-04-25-8/), we 
provide an implementation of both the quantum algorithm (circuit sampling) and classical algorithm (Fourier analysis) for sampling from a
sparse IQP circuit with noise. 

## Running the script
To compare the quantum and classical algorithm's output distributions `<k>` times on `<n>` qubits, run `python general.py <n> -r <k>`. For example, for 
`3` qubits and `5` runs, do `python general.py 3 -r 5`. 

There are a number of parameters that may be adjusted to your liking. 
```
DELTA: accuracy of approximation. 
EPSILON: noise rate of quantum circuit.
ALPHA: a "spread" parameter of the distribution. We have set this to 1 for convenience. 
GAMMA: a probability rate coefficient for CS gates. Refrain from changing this.
```
The smaller `DELTA` and `EPSILON` are and the larger `ALPHA` is, the faster the scaling of the classical algorithm. The exponent scales as `log(ALPHA / DELTA) / EPSILON`, so exercise caution increasing `ALPHA` and decreasing `DELTA` and exercise extreme caution decreasing `EPSILON`. As of now, the parameters are
set so that the sampling algorithm scalings like `O(n^3)`. But the overhead is very large, and will take around 30 seconds even for 3 qubits (compare 0.01s for the quantum algorithm). The easiest thing to do would be to increase `ALPHA` by an order of magnitude if you want better accuracy.

**Caution**: The quantum algorithm has a low overhead but is exponential time. The classical algorithm has a high overhead but is polynomial time to generate *one* sample. However, since we are plotting the whole (exponentially large) distribution, the specific work we are doing is actually exponential time, so the classical 
algorithm to generate te distribution plot is basically exponential time with high overhead. Hence, try not to run this script for more than 5 qubits (or do so at your, and your computer's, own risk!).

## Organization
* `general.py`: Calls `quantum.py` and `classical.py` to run the two algorithms, then compares their performance visually.
* `quantum.py`: Given a parameterization of a IQP circuit, uses `qiskit` backend simulations to calculate the output distributions approximately.
* `classical.py`: Given a parameterization of a IQP circuit, uses Fourier analysis to calculate the output distributions approximately.
* `generator.py`: Randomly generates a parameterization for the IQP circuit.

## Dependencies
1. `qiskit` v1.0.2 or later + `qiskit-aer` v0.14.1 or later.
2. `numpy` v1.26.3 or later.
3. `matplotlib` v3.8.0 or later.

## Notes
This repository was created as part of my work in the MIT 8.371 (Quantum Information Science II) final project for Spring 2024.