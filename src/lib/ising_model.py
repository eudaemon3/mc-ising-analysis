import numpy as np
from numba import njit
from scipy.signal import correlate, correlation_lags

from lib.ising_analysis import IsingResult
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
def _metropolis_open(
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
        # für jeden Schritt ein Sweep über den gesamten Lattice
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

class IsingModel:
    def __init__(self, lattice: Lattice2D, time_steps: int, beta_j: np.ndarray):
        self.lattice = lattice
        self.time = time_steps
        self.beta_j = np.asarray(beta_j, dtype=np.float64)

    @staticmethod
    def _autocorr(series: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        values = np.asarray(series, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError("series must be one-dimensional")

        y = values - values.mean()
        if np.allclose(y, 0.0):
            return np.ones(values.size, dtype=np.float64), 0.5

        raw = correlate(y, y, mode="full", method="fft")
        lags = correlation_lags(values.size, values.size, mode="full")
        acorr = raw[lags >= 0]
        acorr /= acorr[0]

        if acorr.size == 0:
            return acorr, 0.0

        stop_index = np.where(acorr[1:] <= 0.0)[0]
        if stop_index.size == 0:
            cutoff = acorr.size
        else:
            cutoff = stop_index[0] + 1
        tau_int = 0.5 + float(acorr[1:cutoff].sum())

        return acorr, tau_int

    def _simulate_system(self, beta_value: float, pbc: bool,
        ) -> tuple[np.ndarray, np.ndarray]:
        initial_energy = self.lattice.get_energy()
        if pbc:
            return _metropolis_pbc(self.lattice.lattice, self.time, beta_value, initial_energy)
        else:
            return _metropolis_open(self.lattice.lattice, self.time, beta_value, initial_energy)

    def run_analysis(self, pbc: bool = False) -> IsingResult:
        beta_values = self.beta_j
        temperatures = 1.0 / beta_values
        n_beta = beta_values.size

        magnetization = np.zeros(n_beta, dtype=np.float64)
        energy_mean = np.zeros(n_beta, dtype=np.float64)
        energy_std = np.zeros(n_beta, dtype=np.float64)
        susceptibility = np.zeros(n_beta, dtype=np.float64)
        heat_capacity = np.zeros(n_beta, dtype=np.float64)
        autocorrelation_time = np.zeros(n_beta, dtype=np.float64)
        sample_window = np.zeros(n_beta, dtype=np.int64)
        
        spin_density_history = np.zeros((n_beta, self.time), dtype=np.float64)
        burn_in_idx = int(self.time * 0.2) 
        eq_length = self.time - burn_in_idx
        autocorrelation = np.zeros((n_beta, eq_length), dtype=np.float64)
        autocorrelation_lags = np.arange(eq_length, dtype=np.int64)

        for index, beta_value in enumerate(beta_values):
            print(f"Calculating analysis for beta*J = {beta_value:.2f}")
            spins, energies = self._simulate_system(beta_value, pbc)

            spin_density = spins / (self.lattice.length**2)
            spin_density_history[index] = spin_density

            eq_spins = spin_density[burn_in_idx:]
            eq_energies = energies[burn_in_idx:]

            acorr, tau_int = self._autocorr(eq_spins)
            sample_window[index] = len(eq_spins)

            magnetization[index] = eq_spins.mean()
            energy_mean[index] = eq_energies.mean()
            energy_std[index] = eq_energies.std()
            susceptibility[index] = beta_value * self.lattice.length**2 * eq_spins.var()
            heat_capacity[index] = (beta_value**2) * eq_energies.var() / self.lattice.length**2

            autocorrelation[index, :acorr.size] = acorr
            autocorrelation_time[index] = tau_int

        return IsingResult(
            temperature=temperatures,
            magnetization=magnetization,
            energy_mean=energy_mean,
            energy_std=energy_std,
            susceptibility=susceptibility,
            heat_capacity=heat_capacity,

            autocorrelation_time=autocorrelation_time,
            sample_window=sample_window,
            autocorrelation_lags=autocorrelation_lags,
            autocorrelation=autocorrelation,
            
            spin_density=spin_density_history,
            time_steps=self.time,
        )