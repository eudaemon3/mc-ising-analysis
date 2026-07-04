import numpy as np
from scipy.signal import correlate, correlation_lags

from lib.ising_analysis import IsingResult
from lib.lattice_2d import Lattice2D
import lib.metropolis_routines as routine 


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

    def _simulate_system(self, beta_value: float, pbc: bool, sweep: bool) -> tuple[np.ndarray, np.ndarray]:
        initial_energy = self.lattice.get_energy()
        metropolis_routine = None
        if pbc:
            if sweep:
                metropolis_routine = routine.metropolis_pbc_sweep(self.lattice.lattice, self.time, beta_value, initial_energy)
            else:
                metropolis_routine = routine.metropolis_pbc_single(self.lattice.lattice, self.time, beta_value, initial_energy)
        else:
            if sweep:
                metropolis_routine = routine.metropolis_open_sweep(self.lattice.lattice, self.time, beta_value, initial_energy)
            else:
                metropolis_routine = routine.metropolis_open_single(self.lattice.lattice, self.time, beta_value, initial_energy)
        return metropolis_routine

    def run_analysis(self, pbc: bool = False, sweep: bool = False) -> IsingResult:
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
        
        spin_history = np.zeros((n_beta, self.time), dtype=np.float64)
        burn_in_idx = int(self.time * 0.2) 
        eq_length = self.time - burn_in_idx
        autocorrelation = np.zeros((n_beta, eq_length), dtype=np.float64)
        autocorrelation_lags = np.arange(eq_length, dtype=np.int64)

        for index, beta_value in enumerate(beta_values):
            print(f"Calculating analysis for beta*J = {beta_value:.2f}")
            spins, energies = self._simulate_system(beta_value, pbc, sweep)

            spin_density = spins / (self.lattice.length**2)
            spin_history[index] = spins

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
            
            spin_density=spin_history,
            time_steps=self.time,
        )