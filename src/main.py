# -----------------------------------------------------------------------------------------
# Imports and Docker plot setup
# -----------------------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from mpi4py import MPI

from lib.analysis_result import AnalysisResults
from lib.ising_model import IsingModel
from lib.lattice_2d import Lattice2D

plt.style.use("science")
plt.rcParams.update({"font.size": 14})

# -----------------------------------------------------------------------------------------
# Ising Model Calculation Parameters
# -----------------------------------------------------------------------------------------
lattice_size = 64
spin_probabilities = (0.25, 0.75)
time_steps = 80_000
beta_j = np.linspace(0.20, 0.75, 12)

random_seed = 1337

def main() -> None:
    # -------------------------------------------------------------------------------------
    # Perform Analysis of the Ising Model 
    # -------------------------------------------------------------------------------------
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    np.random.seed(random_seed)
    lattices = [Lattice2D(lattice_size, lattice_value) for lattice_value in spin_probabilities]

    results = []
    for model_index, lattice in enumerate(lattices):
        model = IsingModel(lattice, time_steps, beta_j)
        result = model.run_analysis(False, True, comm=comm, seed_offset=model_index * 100_000)
        if rank == 0:
            results.append(result)

    if rank != 0:
        return

    # -------------------------------------------------------------------------------------
    # Display Analysis Results
    # -------------------------------------------------------------------------------------
    isingAnalysis = AnalysisResults(results, beta_j, time_steps)
    summary_figure = isingAnalysis.plot_summary()
    autocorrelation_figure = isingAnalysis.plot_autocorrelation(5)
    auto_time_figure = isingAnalysis.plot_autocorrelation_time()

    summary_path = "/workspace/results/xx_ising_summary.png"
    autocorrelation_path = "/workspace/results/xx_ising_autocorrelation.png"
    auto_time_path = "/workspace/results/xx_ising_autocorrelation_time.png"

    summary_figure.savefig(summary_path, dpi=200)
    autocorrelation_figure.savefig(autocorrelation_path, dpi=200)
    auto_time_figure.savefig(auto_time_path, dpi=200)

    plt.close(summary_figure)
    plt.close(autocorrelation_figure)
    plt.close(auto_time_figure)
    
    # -------------------------------------------------------------------------------------
    # Optional: Detailed Autocorrelation Analysis Output
    # -------------------------------------------------------------------------------------
    # stride = isingAnalysis.get_sampling_stride(safety_factor=5.0)
    # tau_max = np.max(np.vstack([result.autocorrelation_time for result in results]), axis=0)
    # eq_samples_min = np.min(np.vstack([result.sample_window for result in results]), axis=0)
    # effective_samples = np.maximum(1, np.floor(eq_samples_min / stride).astype(np.int64))

    # print("beta_j  tau_int(max)  stride  effective_samples")
    # for beta_value, tau_value, stride_value, n_eff in zip(beta_j, tau_max, stride, effective_samples, strict=True):
    #     print(f"{beta_value:5.2f}  {tau_value:11.2f}  {stride_value:6d}  {n_eff:17d}")

if __name__ == "__main__":
    main()