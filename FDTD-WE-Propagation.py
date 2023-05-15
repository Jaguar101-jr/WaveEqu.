import numpy as np
import plotly.graph_objs as go

# Set the parameters
L = 1  # Length of the domain in x and y directions
c = 1  # Speed of sound
dx = 0.01  # Grid size in x direction
dy = 0.01  # Grid size in y direction
dt = dx / (c * np.sqrt(2))  # Time step
T = 2  # Total simulation time
stoppage_time = 1.5 # Stoppage time

# Define the initial condition
def initial_condition(x, y):
    return np.exp(-(x-L/2)**2-(y-L/2)**2)

# Create the grid
x = np.arange(0, L+dx, dx)
y = np.arange(0, L+dy, dy)
X, Y = np.meshgrid(x, y)

# Define the FDTD parameters
c1 = (c*dt/dx)**2
c2 = 2*(1-c1)

# Initialize the wave field
f_prev = initial_condition(X, Y)
f_now = np.zeros_like(f_prev)

# Run the FDTD simulation until stoppage time
t = 0
while t < stoppage_time:
    # Propagate the wave one time step forward
    for i in range(1, len(x)-1):
        for j in range(1, len(y)-1):
            f_next = c1*(f_prev[i+1, j] + f_prev[i-1, j] + f_prev[i, j+1] + f_prev[i, j-1] - 4*f_prev[i, j]) + c2*f_prev[i, j] - f_now[i, j]
            f_now[i, j] = f_next

    # Update the wave field for the next time step
    f_prev, f_now = f_now, f_prev

    # Update the current time
    t += dt

# Plot the wave field in 3D using Plotly
Z = f_prev.T  # Transpose for correct orientation
fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z)])
fig.update_layout(scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='Amplitude'))
fig.show()
