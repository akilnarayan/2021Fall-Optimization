# Uses Tikhonov regularization to smooth out a model fit

from numpy import sin, cos, linspace, log, zeros, eye
from numpy.linalg import inv
from numpy.random import randn

from matplotlib.pyplot import plot, xlabel, ylabel, legend, show, ioff

ioff()

# Defines the dictionary of functions:
funs = [(lambda k: (lambda t: t**k))(j) for j in range(13)]
# Feature vectors are polynomials up to degree 12

N = len(funs)
M = N+2
t = linspace(0, 1, M)

sig = 1e-4
y = 2 + t**7 - t**4 + sig*randn(M)

# Regularization parameter
lmbda = 1e10

# Form least squares problem, A x = y:
A = zeros((M,N))
for i, fun in enumerate(funs):
    A[:,i] = fun(t)

# This is numerically unstable (compared to other methods). We are just doing
# it this way for demonstration
x = inv(A.T @ A) @ (A.T @ y)
xr = inv(A.T @ A + lmbda*eye(N)) @ (A.T @ y)

tplot = linspace(0, 1, 100)
yplot = zeros(tplot.shape)
yrplot = zeros(tplot.shape)
for xval, fun in zip(x, funs):
    yplot += xval*fun(tplot)
for xval, fun in zip(xr, funs):
    yrplot += xval*fun(tplot)

## Visualization
plot(tplot, yplot, 'b', label="Least squares fit")
plot(tplot, yrplot, 'k', label="Regularized Least squares fit")
plot(t, y, 'r.', markersize=8, label="Data")
legend(frameon=False)

xlabel('$t$')
ylabel('$y$')

show()
