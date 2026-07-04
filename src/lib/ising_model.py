import numpy as np
from scipy.signal import correlate, correlation_lags
from mpi4py import MPI

from lib.ising_analysis import IsingResult
from lib.lattice_2d import Lattice2D
import lib.metropolis_routines as routine 


class IsingModel:
    def __init__(self, lattice: Lattice2D, time_steps: int, beta_j: np.ndarray):
        self.lattice = lattice
        self.time = time_steps
        self.beta_j = np.asarray(beta_j, dtype=np.float64)

    @staticmethod
    def _autocorr(series: np.ndarray) -> tuple[np.ndarray, float]:
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

    def run_analysis(self, pbc: bool = False, sweep: bool = False,
        comm=MPI.COMM_WORLD, seed_offset: int = 0) -> IsingResult | None:

        beta_values = self.beta_j
        temperatures = 1.0 / beta_values
        n_beta = beta_values.size

        # 1. Distribute beta_j values evenly among processes
        # -------------------------------------------------------------------------
        rank = comm.Get_rank()
        size = comm.Get_size()
        local_B_indices = np.array_split(np.arange(n_beta), size)[rank] 
        local_count = local_B_indices.size

        # every subprocess performs caluculation based on index in beta_j array
        # local count -> number of temperature values for each subprocess
        local_magnetization = np.zeros(local_count, dtype=np.float64)
        local_energy_mean = np.zeros(local_count, dtype=np.float64)
        local_energy_std = np.zeros(local_count, dtype=np.float64)
        local_susceptibility = np.zeros(local_count, dtype=np.float64)
        local_heat_capacity = np.zeros(local_count, dtype=np.float64)
        local_autocorrelation_time = np.zeros(local_count, dtype=np.float64)
        local_sample_window = np.zeros(local_count, dtype=np.int64)
        
        burn_in_idx = int(self.time * 0.2) 
        eq_length = self.time - burn_in_idx
        local_spin_history = np.zeros((local_count, self.time), dtype=np.float64)
        local_autocorrelation = np.zeros((local_count, eq_length), dtype=np.float64)
        autocorrelation_lags = np.arange(eq_length, dtype=np.int64)

        # 2. Each process: simulate system for for every given beta_j value
        # -------------------------------------------------------------------------
        for local_pos, index in enumerate(local_B_indices):
            beta_value = beta_values[index]
            np.random.seed(seed_offset + index)
            print(f"Process {rank}: Calculating beta*J = {beta_value:.2f}")
            spins, energies = self._simulate_system(beta_value, pbc, sweep)

            spin_density = spins / (self.lattice.length**2)
            local_spin_history[local_pos] = spins

            eq_spins = spin_density[burn_in_idx:]
            eq_energies = energies[burn_in_idx:]

            acorr, tau_int = self._autocorr(eq_spins)
            local_sample_window[local_pos] = len(eq_spins)

            local_magnetization[local_pos] = eq_spins.mean()
            local_energy_mean[local_pos] = eq_energies.mean()
            local_energy_std[local_pos] = eq_energies.std()
            local_susceptibility[local_pos] = beta_value * self.lattice.length**2 * eq_spins.var()
            local_heat_capacity[local_pos] = (beta_value**2) * eq_energies.var() / self.lattice.length**2

            local_autocorrelation[local_pos, :acorr.size] = acorr
            local_autocorrelation_time[local_pos] = tau_int

        # 3. Gather toghether the data from all processes
        # -------------------------------------------------------------------------
        gather_data = comm.gather(
            {
                "indices": np.asarray(local_B_indices, dtype=np.int64),
                "magnetization": local_magnetization,
                "energy_mean": local_energy_mean,
                "energy_std": local_energy_std,
                "susceptibility": local_susceptibility,
                "heat_capacity": local_heat_capacity,
                "autocorrelation_time": local_autocorrelation_time,
                "sample_window": local_sample_window,
                "autocorrelation": local_autocorrelation,
                "spin_history": local_spin_history,
            },
            root=0,
        )

        # 4. Reconstruct result arrays from all sub-processes
        # -------------------------------------------------------------------------
        if rank != 0:
            return None

        magnetization = np.zeros(n_beta, dtype=np.float64)
        energy_mean = np.zeros(n_beta, dtype=np.float64)
        energy_std = np.zeros(n_beta, dtype=np.float64)
        susceptibility = np.zeros(n_beta, dtype=np.float64)
        heat_capacity = np.zeros(n_beta, dtype=np.float64)
        autocorrelation_time = np.zeros(n_beta, dtype=np.float64)
        sample_window = np.zeros(n_beta, dtype=np.int64)
        spin_history = np.zeros((n_beta, self.time), dtype=np.float64)
        autocorrelation = np.zeros((n_beta, eq_length), dtype=np.float64)

        for payload in gather_data:
            indices = payload["indices"]
            for l, index in enumerate(indices):
                magnetization[index] = payload["magnetization"][l]
                energy_mean[index] = payload["energy_mean"][l]
                energy_std[index] = payload["energy_std"][l]
                susceptibility[index] = payload["susceptibility"][l]
                heat_capacity[index] = payload["heat_capacity"][l]
                autocorrelation_time[index] = payload["autocorrelation_time"][l]
                sample_window[index] = payload["sample_window"][l]
                spin_history[index] = payload["spin_history"][l]
                autocorrelation[index] = payload["autocorrelation"][l]

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