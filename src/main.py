import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scienceplots

from lib.ising_analysis import AnalysisResults
from lib.ising_model import IsingModel
from lib.lattice_2d import Lattice2D

plt.style.use("science")
plt.rcParams.update({"font.size": 14})

lattice_size = 64
time_steps = 80_000
spin_probabilities = (0.25, 0.75)
beta_j = np.arange(0.20, 0.75, 0.05)

def main() -> None:
    results = []
    for lattice_value in spin_probabilities:
        lattice = Lattice2D(lattice_size, lattice_value)
        model = IsingModel(lattice, time_steps, beta_j)
        results.append(model.run_analysis(True))

    isingAnalysis = AnalysisResults(results)
    summary_figure = isingAnalysis.plot_summary()
    autocorrelation_figure = isingAnalysis.plot_autocorrelation(beta_j)

    summary_path = "/workspace/results/ising_summary.png"
    autocorrelation_path = "/workspace/results/ising_autocorrelation.png"

    summary_figure.savefig(summary_path, dpi=200)
    autocorrelation_figure.savefig(autocorrelation_path, dpi=200)

    plt.close(summary_figure)
    plt.close(autocorrelation_figure)


if __name__ == "__main__":
    main()
