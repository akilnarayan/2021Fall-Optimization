# Uses anisotropic smoothing to mitigate overfitting

from numpy import sin, cos, linspace, log, zeros, diag, ones, eye
from numpy.linalg import inv

from matplotlib.pyplot import plot, xlabel, ylabel, legend, show, ioff, subplot, title

ioff()

# Defines the dictionary of functions:
funs = [lambda t: t**2,      \
        lambda t: t**3,      \
        lambda t: sin(30*t), \
        lambda t: cos(30*t), \
        lambda t: sin(t),    \
        lambda t: cos(t)]

N = len(funs)
M = 10
t = linspace(-1, 1, M)

y = log(t**2 + 1)/(1.5 + sin(3*t))

# Regularization parameters
lmbdas = [1e-2, 1e0, 1e2, 1e4]

# Form least squares problem, A x = y:
A = zeros((M,N))
for i, fun in enumerate(funs):
    A[:,i] = fun(t)

R = eye(N)
R[2,2], R[3,3] = 20, 20

for ind, lmbda in enumerate(lmbdas):
    # First derivative denoising

    # Solving the normal equations explicitly is generally not a good idea
    tplot = linspace(-1, 1, 200)
    x = inv(A.T @ A + lmbda*R.T @ R) @ (A.T @ y)

    yplot = zeros(tplot.shape)
    for xval, fun in zip(x, funs):
        yplot += xval*fun(tplot)

    ## Visualization
    subplot(int("22"+str(ind+1)))
    plot(t, y, 'b.', label="Noisy data")
    plot(tplot, yplot, 'r-', linewidth=1, label="Model")
    legend(frameon=False)

    xlabel('$t$')
    ylabel('$y$')
    title('$\\lambda={0:1.0e}$'.format(lmbda))

show()
