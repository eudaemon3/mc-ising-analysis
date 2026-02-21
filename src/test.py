import numpy as np
import matplotlib
matplotlib.use("WebAgg")
import matplotlib.pyplot as plt

t = np.linspace(-2*np.pi,2*np.pi,1000)
plt.plot(t, np.atan(t))
plt.grid(True)
plt.show()