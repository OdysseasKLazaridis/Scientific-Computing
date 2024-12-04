

import numpy as np
import matplotlib.pyplot as plt

# Diffusion coefficients
Dp = 1
Dq = 8
C = 4.5
# Time
Nt = 2000 # Max time
dt = 1e-2 # Time step (for Euler solver)
# Grid
N = 43 # Max grid pointZz
x = np.linspace(-1,41,N)
y = np.linspace(-1,41,N)
h = 1 # Grid step (dx = dy = h)
# Defs
p_laplace = np.zeros([N, N])
q_laplace = np.zeros([N, N])
p = np.zeros([N, N, Nt])
q = np.zeros([N, N, Nt])

def ContourPlot(M,k):
    p = np.linspace(0, 42, M.shape[0])
    X,Y = np.meshgrid(p, p)
    plt.imshow(M)
    plt.contourf(X, Y, M, 200)
    plt.colorbar()
    plt.savefig(str(k))
    plt.show()
# Boundary Conditions
def NeumannBounds(p,q):
    p[0,:,k] = p[1,:,k]; p[N-1,:,k] = p[N-2,:,k] # p
    p[:,0,k] = p[:,1,k]; p[:,N-1,k] = p[:,N-2,k]
    q[0,:,k] = q[1,:,k]; q[N-1,:,k] = q[N-2,:,k] # q
    q[:,0,k] = q[:,1,k]; q[:,N-1,k] = q[:,N-2,k]
    return p,q

for K in range(12,13):
    p[11:32,11:32,0] = C + 0.1
    q[11:32,11:32,0] = K/C + 0.2
    for k in range(Nt-1):
        for i in range(N-1):
            for j in range(N-1):
                # Neumann
                p,q=NeumannBounds(p,q)
                # Integration of equations with Forward Euler solver
                p_laplace[i,j] = (p[i-1,j,k] + p[i+1,j,k] + p[i,j+1,k] + p[i,j-1,k] - 4*p[i,j,k]) / (h**2)
                q_laplace[i,j] = (q[i-1,j,k] + q[i+1,j,k] + q[i,j+1,k] +q[i,j-1,k] - 4*q[i,j,k]) / (h**2)
                # Forward Euler
                p[i,j,k+1] = p[i,j,k] + dt*((Dp * p_laplace[i,j]) + ((p[i,j,k]* p[i,j,k]) * q[i,j,k]) + C - ((K + 1) * p[i,j,k]))
                q[i,j,k+1] = q[i,j,k] + dt*((Dq * q_laplace[i,j]) - ((p[i,j,k]* p[i,j,k]) * q[i,j,k]) + (K * p[i,j,k]))
    print("For K = {}\np".format(K))
    ContourPlot(p[1:42,1:42,k],K)
    print("q")
    ContourPlot(q[1:42,1:42,k],k)