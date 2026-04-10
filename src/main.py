import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import subprocess
import scienceplots

from lib.ising_model import IsingModel
from lib.lattice_2d import Lattice2D

plt.style.use('science')
plt.rcParams.update({'font.size': 14})
matplotlib.use("Agg")

steps = 1_000_000
size = 100

beta_j = np.arange(0.2, 1, 0.02) # good range for phase change
# beta_j = np.arange(0.1, 2, 0.05) # good range for Cv

lattice_n = Lattice2D(size, 0.75) # 75% spin down
lattice_p = Lattice2D(size, 0.25) # 25% spin down

Ising_n = IsingModel(lattice_n, steps, beta_j)
Ising_p = IsingModel(lattice_p, steps, beta_j)

Ising_n.save_txt("/tmp/Ising_n_good_PS.txt")
Ising_p.save_txt("/tmp/Ising_p_good_PS.txt")

ms_n, E_means_n, E_stds_n = Ising_n.load_txt("/tmp/Ising_n_good_PS.txt")
ms_p, E_means_p, E_stds_p = Ising_p.load_txt("/tmp/Ising_p_good_PS.txt")

ms_n, E_means_n, E_stds_n = Ising_n.get_spin_energy()
ms_p, E_means_p, E_stds_p = Ising_p.get_spin_energy()

plt.figure(figsize=(6,4))
plt.plot(1/beta_j, ms_n, 'b--o', label=r'75\% negative spin')
plt.plot(1/beta_j, ms_p, 'r--o', label=r'75\% positive spin')
plt.xlabel(r'Temperature $T\frac{k_B}{J}$', fontsize=16)
plt.ylabel(r'average spin $\bar{m}$', fontsize=16)
plt.legend(facecolor='white', framealpha=1, fontsize=16, frameon=True)
plt.grid(True)
plt.savefig("/tmp/spin_Temp.png")

plt.figure(figsize=(6,4))
plt.plot(1/beta_j, E_stds_n*beta_j, 'b-+', label=r'75\% negative spin')
plt.plot(1/beta_j, E_stds_p*beta_j, 'r-+', label=r'75\% positive spin')
plt.xlabel(r'Temperature $T\frac{k_B}{J}$', fontsize=16)
plt.ylabel(r'Heat Capacity $C_V / k_B^2$')
plt.legend(facecolor='white', framealpha=1, fontsize=16, frameon=True)
plt.grid(True)
plt.savefig("/tmp/Cv_Temp.png")

subprocess.run(["code", "/tmp/spin_Temp.png"])
subprocess.run(["code", "/tmp/Cv_Temp.png"])