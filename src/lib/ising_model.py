import numpy as np
from mpi4py import MPI

from lib.analysis_result import IsingResult
from lib.lattice_2d import Lattice2D
from lib.helpers import METRIC_SCHEMA, jackknife_error, autocorr
import lib.metropolis_routines as routine 

class IsingModel:
    """
    A class to perform Monte Carlo simulations of the 2D Ising Model.

    Attributes
    ----------
    lattice : Lattice2D
        Instance of the 2D lattice providing the initial spin configuration and energy.
    time : int
        Base number of Monte Carlo steps to evaluate for each temperature point.
    beta_j : np.ndarray
        Array of dimensionless inverse temperature coupling values (beta * J = 1/T) to simulate.
    mult : np.ndarray
        Per-beta multiplier applied to the base simulation length, allowing for 
        longer equilibration/sampling at critical temperatures.

    Example
    -------
    >>> lattice = Lattice2D(32, 0.25)
    >>> model = IsingModel(lattice, time_steps=5000, beta_j=np.array([0.2, 0.4]))
    >>> result = model.run_analysis()
    """
    def __init__(self, 
            lattice: Lattice2D, 
            time_steps: int, 
            beta_j: np.ndarray, 
            multiplier: np.ndarray | None = None
        ) -> None:

        self.lattice = lattice
        self.time = time_steps
        self.beta_j = np.asarray(beta_j, dtype=np.float64)

        if multiplier is None:
            self.mult = np.ones_like(beta_j)
        else:
            self.mult = np.asarray(multiplier, dtype=np.int32)
  
    def _simulate_system(
            self, beta_value: float, pbc: bool, sweep: bool, index: int
        ) -> tuple[np.ndarray, np.ndarray]:
        """
        Executes a single Metropolis Monte Carlo simulation for a specific beta value.
        """
        routine_map = {
            (True, True): routine.metropolis_pbc_sweep,
            (True, False): routine.metropolis_pbc_single,
            (False, True): routine.metropolis_open_sweep,
            (False, False): routine.metropolis_open_single
        }
        
        selected_routine = routine_map[(pbc, sweep)]
        initial_energy = self.lattice.get_energy()
        eval_time = self.time * self.mult[index]
        return selected_routine(self.lattice.lattice, eval_time, beta_value, initial_energy)

    def _analyze_single_beta(
            self, index: int, beta_value: float, pbc: bool, sweep: bool
        ) -> dict:
        """
        Performs a simulation at a given inverse temperature, discards burn-in data, 
        and calculates thermodynamic observables along with their statistical errors.
        """
        spins, energies = self._simulate_system(beta_value, pbc, sweep, index)
        
        burn_in_idx = int(self.time * self.mult[index] * 0.2)
        spin_density = spins / (self.lattice.length**2)
        eq_spins = spin_density[burn_in_idx:]
        eq_energies = energies[burn_in_idx:]

        acorr, tau_int = autocorr(eq_spins)
        
        tau_for_stride = max(float(tau_int), 0.5)
        thinning_stride = max(int(np.ceil(5.0 * tau_for_stride)), 1)
        thinned_sample_count = max(1, len(eq_spins) // thinning_stride)

        length_sq = self.lattice.length**2

        mean_metric = lambda d: d.mean()
        Chi_metric = lambda d: beta_value * length_sq * d.var()
        Cv_metric = lambda d: (beta_value**2) * d.var() / length_sq
        return {
            "index": index,
            "eval_time_steps": spins.size,
            "spin_history": spins,
            "autocorrelation": acorr,
            "autocorrelation_time": tau_int,
            "sample_window": len(eq_spins),
            "thinning_stride": thinning_stride,
            "thinned_sample_count": thinned_sample_count,
            "magnetization": eq_spins.mean(),
            "energy_mean": eq_energies.mean(),
            "susceptibility": beta_value * length_sq * eq_spins.var(),
            "heat_capacity": (beta_value**2) * eq_energies.var() / length_sq,
            "magnetization_err": jackknife_error(eq_spins, mean_metric, tau_int),
            "energy_mean_err": jackknife_error(eq_energies, mean_metric, tau_int),
            "Chi_err": jackknife_error(eq_spins, Chi_metric, tau_int),
            "Cv_err": jackknife_error(eq_energies, Cv_metric, tau_int)
        }

    def run_analysis(
            self, 
            pbc: bool = False, 
            sweep: bool = False, 
            comm=MPI.COMM_WORLD, 
            seed_offset: int = 0
        ) -> IsingResult | None:
        """
        Executes the Monte Carlo simulations across all specified beta_j values, 
        utilizing MPI to distribute the workload among available parallel workers.

        Parameters
        ----------
        pbc : bool, default=False
            If True, enforces periodic boundary conditions on the lattice.
        sweep : bool, default=False
            If True, updates the lattice using full-sweep Metropolis steps rather 
            than single-spin updates.
        comm : MPI.Comm, optional
            MPI communicator used to parallelize the simulation workload.
        seed_offset : int, default=0
            Offset applied to the beta index to ensure distinct random seeds 
            across distributed tasks.

        Returns
        -------
        IsingResult | None
            A populated dataclass containing the aggregated analysis results on the 
            root process. Returns None for all other processes.

        Example
        -------
        >>> result = model.run_analysis(pbc=True, sweep=True)
        """
        rank = comm.Get_rank()
        size = comm.Get_size()
        
        # 1. Distribute beta_j values evenly among available processes
        # ------------------------------------------------------------------------- 
        local_indices = np.array_split(np.arange(self.beta_j.size), size)[rank]

        # 2. Each process analyzes its assigned subset of beta_j values 
        # ------------------------------------------------------------------------- 
        local_results = []
        for index in local_indices:
            beta_value = self.beta_j[index]
            np.random.seed(seed_offset + index)
            print(f"Process {rank}: Calculating beta*J = {beta_value:.2f}")
            
            result_dict = self._analyze_single_beta(index, beta_value, pbc, sweep)
            local_results.append(result_dict)

        # 3. Gather together the processed data from all MPI workers
        # ------------------------------------------------------------------------- 
        gathered_data = comm.gather(local_results, root=0)

        if rank != 0:
            return None

        all_results = []
        for process_list in gathered_data:
            for single_beta_dict in process_list:
                all_results.append(single_beta_dict)

        # 4. Allocate space for the result data
        # ------------------------------------------------------------------------- 
        n_beta = self.beta_j.size
        max_time_steps = int(self.time * np.max(self.mult))
        max_burn_in = int(max_time_steps * 0.2)
        max_eq_length = max_time_steps - max_burn_in

        final_data = {
            "autocorrelation_lags": np.arange(max_eq_length, dtype=np.int64),
            "spin_density": np.zeros((n_beta, max_time_steps), dtype=np.float64),
            "autocorrelation": np.zeros((n_beta, max_eq_length), dtype=np.float64),
            "eval_time_steps": np.zeros(n_beta, dtype=np.int64),
        }
        for metric, dtype in METRIC_SCHEMA.items():
            final_data[metric] = np.zeros(n_beta, dtype=dtype)

        # 5. Fill final results dictionary with the calculated values
        # ------------------------------------------------------------------------- 
        for res in all_results:
            target_idx = res["index"] # global index of current beta_j
            
            spin_history = res["spin_history"]
            final_data["eval_time_steps"][target_idx] = res["eval_time_steps"]
            final_data["spin_density"][target_idx, :spin_history.size] = spin_history
            acorr_data = res["autocorrelation"]
            final_data["autocorrelation"][target_idx, :acorr_data.size] = acorr_data
            
            for metric_name in METRIC_SCHEMA.keys():
                metric_value = res[metric_name]
                final_data[metric_name][target_idx] = metric_value

        return IsingResult(**final_data)