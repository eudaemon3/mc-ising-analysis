from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

FONTSIZE = 16

@dataclass(frozen=True)
class IsingResult:
    temperature: np.ndarray
    magnetization: np.ndarray
    energy_mean: np.ndarray
    energy_std: np.ndarray
    susceptibility: np.ndarray
    heat_capacity: np.ndarray
    autocorrelation_time: np.ndarray
    sample_window: np.ndarray
    autocorrelation_lags: np.ndarray
    autocorrelation: np.ndarray
    spin_density: np.ndarray
    time_steps: int

class AnalysisResults:
    def __init__(self, results: list[IsingResult]):
        if not results:
            raise ValueError("results must not be empty")
        self.results = results

    def plot_summary(self) -> plt.Figure:
        labels = [r'$75\%$ spin up', r'$75\%$ spin down']
        figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
        styles = [
            {"color": "blue", "linestyle": "-"},
            {"color": "crimson", "linestyle": "-"},
        ]
        plot_specs = [
            (axes[0, 0], "magnetization", r"Spin density $m$"),
            (axes[0, 1], "energy_mean", r"Mean energy $\langle E \rangle$"),
            (axes[1, 0], "susceptibility", r"Susceptibility $\chi$"),
            (axes[1, 1], "heat_capacity", r"Specific heat $C_V$"),
        ]

        for axis, attribute_name, ylabel in plot_specs:
            for index, (result, label) in enumerate(zip(self.results, labels, strict=True)):
                style = styles[index % len(styles)]
                axis.plot(
                    result.temperature,
                    getattr(result, attribute_name),
                    marker="d",
                    **style,
                    label=label,
                )
            axis.set_xlabel(r"Temperature $(k_B / J)\cdot T$",fontsize=FONTSIZE)
            axis.set_ylabel(ylabel, fontsize=FONTSIZE)
            axis.grid(True)
            axis.legend(frameon=True, facecolor="white", framealpha=1)

        figure.suptitle("Ising model analysis", fontsize=FONTSIZE+4)
        return figure

    def plot_autocorrelation(self, beta_j: np.ndarray, beta_index: int) -> plt.Figure:
        labels = [r"$75\%$ spin up", r"$75\%$ spin down"]
        if not 0 < beta_index <= beta_j.size:
            raise ValueError("beta index is not a valid number")

        figure, (axis_corr, axis_spin) = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
        styles = [
            {"color": "blue", "linestyle": "-"},
            {"color": "crimson", "linestyle": "-"},
        ]

        for index, (result, label) in enumerate(zip(self.results, labels, strict=True)):
            style = styles[index % len(styles)]
            tau_int = result.autocorrelation_time[beta_index]
            axis_corr.plot(
                result.autocorrelation_lags,
                result.autocorrelation[beta_index],
                label=rf"{label} ($\tau_{{int}}={tau_int:.2f}$)",
                **style,
            )

            sample_window = int(result.sample_window[beta_index])
            sample_start = max(result.time_steps - sample_window, 0)
            axis_spin.plot(
                np.arange(result.time_steps),
                result.spin_density[beta_index],
                label=label,
                **style,
            )
            axis_spin.axvspan(sample_start, result.time_steps, color='gray', alpha=0.3)

        axis_corr.set_xlabel(r"Lag $\ell$", fontsize=FONTSIZE)
        axis_corr.set_ylabel(r"Autocorrelation $c(\ell)$", fontsize=FONTSIZE)
        axis_corr.set_title(rf"Autocorrelation at $\beta J={beta_j[beta_index]:.2f}$", fontsize=FONTSIZE+2)
        axis_corr.grid(True, alpha=1.0)
        axis_corr.legend(frameon=True, facecolor="white", framealpha=1)

        axis_spin.set_xlabel(r"MC steps $t$", fontsize=FONTSIZE)
        axis_spin.set_ylabel(r"Spin density $m(t)$", fontsize=FONTSIZE)
        axis_spin.set_title("Sample Spin Measurements", fontsize=FONTSIZE+2)
        axis_spin.grid(True, alpha=1.0)
        axis_spin.legend(frameon=True, facecolor="white", framealpha=1)

        return figure
