import numpy.random as npr
import numpy as np

def circuit_parameters(n, prob_cs):
    T_powers = np.array([npr.randint(8) for _ in range(n)])
    CS_powers = []
    CS_pairs = []
    for i in range(n-1):
        for j in range(i+1, n):
            if npr.rand() < prob_cs:
                CS_pairs.append([i, j])
                CS_powers.append(npr.randint(4))
    return T_powers, CS_powers, CS_pairs