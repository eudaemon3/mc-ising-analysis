import numpy as np
from numba import njit
from lib.lattice_2d import Lattice2D

@njit("UniTuple(f8[:], 2)(f8[:,:], i8, f8, f8)", nogil=True)
def _metropolis_pbc(
    spin_array: np.ndarray,
    times: int,
    beta_J: np.float64,
    energy: np.float64,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Metropolis Algorithm for the 2D-Ising-Model using periodic boundary conditions.

    Parameters
    ----------
    spin_array : np.ndarray
        Square array of spins (+1 or -1) representing the lattice state.
    times : int
        Number of Metropolis updates.
    beta_J : np.float64
        Related to Temperature of the system (beta * J = J/(k_B*T)).
    energy : np.float64
        Initial energy of the spin configuration.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing total magnetization and energy for each update step.
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

        # Periodische Randbedingungen via Modulo
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
def _metropolis_open(
    spin_array: np.ndarray,
    times: int,
    beta_J: np.float64,
    energy: np.float64
) -> tuple[np.ndarray, np.ndarray]:

    """
    Metropolis Algorithm for the 2D-Ising-Model using open boundary conditions.

    Parameters
    ----------
    spin_array : np.ndarray
        Square array of spins (+1 or -1) representing the lattice state.
    times : int
        Number of Metropolis updates.
    beta_J : np.float64
        Related to Temperature of the system (beta * J = J/(k_B*T)).
    energy : np.float64
        Initial energy of the spin configuration.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing total magnetization and energy for each update step.
    """

    spin_array = spin_array.copy()
    net_spin = np.zeros(times-1)
    net_energy = np.zeros(times-1)
    N = len(spin_array)

    for t in range(times-1):
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

class IsingModel:
    """
    Custom Monte Carlo class for a 2D Ising lattice.

    The class runs Metropolis updates over a range of temperature values
    computes average magnetization and energy statistics.

    Attributes
    ----------
    lattice : Lattice2D
        The underlying square Ising Model lattice.
    time : int
        The number of Monte Carlo update steps to perform per temperature.
    beta_j : np.ndarray
        Array of inverse temperature times coupling constant values.
    """

    _DEFAULT_PATH = "/tmp/ising_data.txt"

    def __init__(self, lattice: Lattice2D, time_steps: int, beta_j: np.ndarray):
        self.lattice = lattice
        self.time = time_steps
        self.beta_j = beta_j

    @staticmethod
    def _run_metropolis(
        spin_array: np.ndarray,
        times: int,
        beta_J: float,
        energy: float,
        pbc: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        if pbc:
            return _metropolis_pbc(spin_array, times, beta_J, energy)
        else:
            return _metropolis_open(spin_array, times, beta_J, energy)


    def get_spin_energy(self, pbc: bool = False):
        len_beta = len(self.beta_j)
        ms = np.zeros(len_beta)
        E_means = np.zeros(len_beta)
        E_stds = np.zeros(len_beta)
        eval_time = self.time // 10

        for i, bj in enumerate(self.beta_j):
            print((f'Caclulating System for beta*J = {bj:.2f}'))
            spins, energies = self._run_metropolis(self.lattice.lattice, self.time, bj, self.lattice.get_energy(), pbc)
            ms[i] = spins[-eval_time:].mean()/self.lattice.length**2
            E_means[i] = energies[-eval_time:].mean()
            E_stds[i] = energies[-eval_time:].std()
        return ms, E_means, E_stds
    
    def save_txt(self, path: str = _DEFAULT_PATH) -> None:
        ms, E_means, E_stds = self.get_spin_energy()

        data = np.column_stack([ms, E_means, E_stds])
        header = "m,E_mean,E_std"

        np.savetxt(path, data, header=header, fmt="%.8f", delimiter=",")

    @staticmethod
    def load_txt(path: str = _DEFAULT_PATH) -> tuple[np.ndarray]:
        try:
            data = np.loadtxt(path, comments="#", delimiter=",")
        except OSError as exc:
            raise FileNotFoundError(f"file not found: {path}") from exc

        if data.ndim != 2 or data.shape[1] != 3:
            raise ValueError(
                f" Unexpected format in {path}: "
                f" expect (0,3), found {data.shape}"
            )

        return data[:, 0], data[:, 1], data[:, 2]


    
