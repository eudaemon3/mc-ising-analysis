import numpy as np
import matplotlib.pyplot as plt
import subprocess

class Lattice2D:
    """
    A representation of a 2D square lattice for the Ising Model.

    Attributes
    ----------
    length : int
        The linear dimension (N) of the N x N square lattice.
    spin_prob (float): 
        The probability of initializing a spin to -1.
    lattice : np.ndarray
        The N x N numpy array containing the generated spin configuration.

    Example
    -------
    >>> Lattice2D(64, 0.25) # creates a 64x64 grid with 75% of spins pointing up.
    >>> Lattice2D(64, 0.75) # creates a 64x64 grid with 75% of spins pointing down.
    """

    def __init__(self, N: int, spin_prob: float):
        if not 0 <= spin_prob <= 1:
            raise ValueError("spin_prob must be between 0 and 1.")

        self.length = N # assume square lattice
        self.spin_prob = spin_prob
        init_rnd = np.random.random((N,N))

        lattice = np.ones((N,N))
        lattice[init_rnd < spin_prob] = -1

        self.lattice = lattice

    def _sum_edges(self, pos:tuple[int]) -> int:
        """
        Calculate the interaction energy of a single spin with its nearest neighbors.
        """
        E_i = 0
        y,x = pos
        sigma_i = self.lattice[y,x]

        if x > 0:
            E_i += -sigma_i*self.lattice[y,x-1]
        if x < self.length-1:
            E_i += -sigma_i*self.lattice[y,x+1]
        if y > 0:
            E_i += -sigma_i*self.lattice[y-1,x]
        if y < self.length-1:
            E_i += -sigma_i*self.lattice[y+1,x]

        return E_i
    
    def get_energy(self) -> np.float64:
        """
        Compute the total energy of the current lattice configuration.
        """
        E_tot = 0
        for y in range(self.length):
            for x in range(self.length):
                E_tot += self._sum_edges((y,x))
        return -E_tot
    
    def _show_img(self) -> None:
        """
        Display the initial lattice configuration as an image.
        """
        plt.figure(5,5)
        plt.savefig("/tmp/lattice_init.png")
        subprocess.run(["code", "/tmp/lattice_init.png"]) 