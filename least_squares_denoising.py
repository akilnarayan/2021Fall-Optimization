# Uses a discretized first derivative to denoise a signal.

from numpy import linspace, log, sin, ones, eye, diag
from numpy.linalg import inv
from numpy.random import randn

from matplotlib.pyplot import plot, xlabel, ylabel, legend, show, subplot, title, ioff

ioff()

M = 100
t = linspace(-1, 1, M)

sig = 0.05
y = log(t**2 + 1)/(1.5 + sin(3*t)) + sig*randn(M)

# Regularization parameters
lmbdas = [100, 1000, 1e4, 1e5]

# solution x should match the data y
A = eye(M)

for ind, lmbda in enumerate(lmbdas):
    # First derivative denoising
    R = (diag(ones(M), k=0)[:-1,:] + diag(-ones(M-1), k=1)[:-1,:])/M

    # Solving the normal equations explicitly is generally not a good idea
    x = inv(A.T @ A + lmbda*R.T @ R) @ (A.T @ y)

    ## Visualization
    subplot(int("22"+str(ind+1)))
    plot(t, y, 'b.', label="Noisy data")
    plot(t, x, 'r-', linewidth=1, label="Denoised")
    legend(frameon=False)

    xlabel('$t$')
    ylabel('$y$')
    title('$\\lambda={0:1.0e}$'.format(lmbda))

show()
