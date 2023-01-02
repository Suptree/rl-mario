import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(0, 1, 100)
plt.figure(0)
plt.plot(x, 4*x*(1-x))
plt.show()