import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define simulation parameters
c = 0.1    # Speed of sound in air (m/s)
dx = 0.05    # Spatial step (m)
dt = dx/c    # Time step (s)
t_end = 1.0  # End time of simulation (s)

# Define grid
x = np.arange(0, 1, dx)
y = np.arange(0, 1, dx)
X, Y = np.meshgrid(x, y)

# Define initial conditions
f = np.zeros((len(x), len(y)))
f[len(x)//2, len(y)//2] = 0.1

# Define update coefficients
c1 = (c*dt/dx)**2
c2 = 2*(1-c1)

# Define arrays for storing previous and current values
f_prev = f.copy()
f_next = np.zeros_like(f)

# Perform time integration
for t in np.arange(0, t_end, dt):
    # Apply boundary conditions
    f_prev[:, 0] = 0
    f_prev[:, -1] = 0
    f_prev[0, :] = 0
    f_prev[-1, :] = 0

    # Update interior points using FDTD method
    for i in range(1, len(x)-1):
        for j in range(1, len(y)-1):
            f_next[i, j] = c1*(f_prev[i+1, j] + f_prev[i-1, j] + f_prev[i, j+1] + f_prev[i, j-1] - 4*f_prev[i, j]) + c2*f_prev[i, j] - f_next[i, j]

    # Update current and previous values
    f_prev, f_next = f_next, f_prev

# Create 3D surface plot of solution
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, f_prev.T, cmap='jet')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f')
plt.show()
