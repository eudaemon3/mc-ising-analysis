import numpy as np
from numba import njit

@njit("UniTuple(f8[:], 2)(f8[:,:], i8, f8, f8)", nogil=True)
def metropolis_pbc_sweep(
    spin_array: np.ndarray,
    times: int,
    beta_J: np.float64,
    energy: np.float64,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Metropolis Algorithm for the 2D-Ising-Model using periodic boundary conditions.
    'times' now represents full Monte Carlo sweeps (N*N flip attempts).
    """
    spin_array = spin_array.copy()
    N = len(spin_array)
    N_sq = N * N

    net_spin = np.zeros(times)
    net_energy = np.zeros(times)

    for t in range(times):
        # für jeden Schritt ein Sweep über den gesamten Lattice
        for _ in range(N_sq):
            x = np.random.randint(0, N)
            y = np.random.randint(0, N)

            spin_i = spin_array[y, x]
            spin_f = -spin_i

            dE = 2.0 * spin_i * (
                spin_array[y, (x + 1) % N] +
                spin_array[y, (x - 1) % N] +
                spin_array[(y + 1) % N, x] +
                spin_array[(y - 1) % N, x]
            )
            if dE <= 0.0 or np.random.random() < np.exp(-beta_J * dE):
                spin_array[y, x] = spin_f
                energy += dE

        net_spin[t] = spin_array.sum()
        net_energy[t] = energy

    return net_spin, net_energy

@njit("UniTuple(f8[:], 2)(f8[:,:], i8, f8, f8)", nogil=True)
def metropolis_open_sweep(
    spin_array: np.ndarray,
    times: int,
    beta_J: np.float64,
    energy: np.float64
) -> tuple[np.ndarray, np.ndarray]:
    """
    Metropolis Algorithm for the 2D-Ising-Model using open boundary conditions.
    'times' now represents full Monte Carlo sweeps (N*N flip attempts).
    """
    spin_array = spin_array.copy()
    net_spin = np.zeros(times)
    net_energy = np.zeros(times)
    N = len(spin_array)
    N_sq = N * N

    for t in range(times):
        for _ in range(N_sq):
            x = np.random.randint(0, N)
            y = np.random.randint(0, N)

            spin_i = spin_array[y,x]
            spin_f = spin_i * -1
            E_i = 0
            E_f = 0

            if x > 0:
                E_i += -spin_i*spin_array[y,x-1]
                E_f += -spin_f*spin_array[y,x-1]
            if x < N-1:
                E_i += -spin_i*spin_array[y,x+1]
                E_f += -spin_f*spin_array[y,x+1]
            if y > 0:
                E_i += -spin_i*spin_array[y-1,x]
                E_f += -spin_f*spin_array[y-1,x]
            if y < N-1:
                E_i += -spin_i*spin_array[y+1,x]
                E_f += -spin_f*spin_array[y+1,x]
            dE = E_f - E_i

            if dE <= 0.0 or np.random.random() < np.exp(-beta_J * dE):
                spin_array[y, x] = spin_f
                energy += dE

        net_spin[t] = spin_array.sum()
        net_energy[t] = energy
    
    return net_spin, net_energy

@njit("UniTuple(f8[:], 2)(f8[:,:], i8, f8, f8)", nogil=True)
def metropolis_pbc_single(
    spin_array: np.ndarray,
    times: int,
    beta_J: np.float64,
    energy: np.float64,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Metropolis Algorithm for the 2D-Ising-Model using periodic boundary conditions.
    'times' now represents full Monte Carlo sweeps (N*N flip attempts).
    """
    spin_array = spin_array.copy()
    N = len(spin_array)

    net_spin = np.zeros(times)
    net_energy = np.zeros(times)

    for t in range(times):
    
        x = np.random.randint(0, N)
        y = np.random.randint(0, N)

        spin_i = spin_array[y, x]
        spin_f = -spin_i

        dE = 2.0 * spin_i * (
            spin_array[y, (x + 1) % N] +
            spin_array[y, (x - 1) % N] +
            spin_array[(y + 1) % N, x] +
            spin_array[(y - 1) % N, x]
        )
        if dE <= 0.0 or np.random.random() < np.exp(-beta_J * dE):
            spin_array[y, x] = spin_f
            energy += dE

        net_spin[t] = spin_array.sum()
        net_energy[t] = energy

    return net_spin, net_energy

@njit("UniTuple(f8[:], 2)(f8[:,:], i8, f8, f8)", nogil=True)
def metropolis_open_single(
    spin_array: np.ndarray,
    times: int,
    beta_J: np.float64,
    energy: np.float64
) -> tuple[np.ndarray, np.ndarray]:
    """
    Metropolis Algorithm for the 2D-Ising-Model using open boundary conditions.
    'times' now represents full Monte Carlo sweeps (N*N flip attempts).
    """
    spin_array = spin_array.copy()
    net_spin = np.zeros(times)
    net_energy = np.zeros(times)
    N = len(spin_array)

    for t in range(times):
        
        x = np.random.randint(0, N)
        y = np.random.randint(0, N)

        spin_i = spin_array[y,x]
        spin_f = spin_i * -1
        E_i = 0
        E_f = 0

        if x > 0:
            E_i += -spin_i*spin_array[y,x-1]
            E_f += -spin_f*spin_array[y,x-1]
        if x < N-1:
            E_i += -spin_i*spin_array[y,x+1]
            E_f += -spin_f*spin_array[y,x+1]
        if y > 0:
            E_i += -spin_i*spin_array[y-1,x]
            E_f += -spin_f*spin_array[y-1,x]
        if y < N-1:
            E_i += -spin_i*spin_array[y+1,x]
            E_f += -spin_f*spin_array[y+1,x]
        dE = E_f - E_i

        if dE <= 0.0 or np.random.random() < np.exp(-beta_J * dE):
            spin_array[y, x] = spin_f
            energy += dE

        net_spin[t] = spin_array.sum()
        net_energy[t] = energy
    
    return net_spin, net_energy