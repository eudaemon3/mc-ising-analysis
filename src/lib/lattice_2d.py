import numpy as np
import matplotlib.pyplot as plt
import subprocess

class Lattice2D:
    """
    Custom 2D-Ising-Model lattice class.
    The lattice is represented by an N by N array of spins taking values of +1 or -1.

    Attributes:
        N (int): 
            The size of the square lattice.
        spin_prob (float): 
            The probability of initializing a spin to -1.
    """

    def __init__(self, N: int, spin_prob: float):
        if not 0 <= spin_prob <= 1:
            raise ValueError("spin_prob must be between 0 and 1.")

        self.length = N # assume square lattice
        init_rnd = np.random.random((N,N))

        lattice = np.ones((N,N))
        lattice[init_rnd < spin_prob] = -1

        self.lattice = lattice

    def _sum_edges(self, pos:tuple[int]) -> int:
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
        E_tot = 0
        for y in range(self.length):
            for x in range(self.length):
                E_tot += self._sum_edges((y,x))
        return -E_tot
    
    def _show_img(self) -> None:
        plt.figure(5,5)
        plt.imshow(self.lattice)
        # plt.savefig("/tmp/lattice_init.png")
        # subprocess.run(["code", "/tmp/lattice_init.png"])