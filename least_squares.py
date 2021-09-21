from numpy import sin, cos, linspace, log, zeros
from numpy.linalg import inv

from matplotlib.pyplot import plot, xlabel, ylabel, legend, show

# Defines the dictionary of functions:
funs = [lambda t: t**2,   \
        lambda t: t**3,   \
        lambda t: sin(t), \
        lambda t: cos(t)]

N = len(funs)
M = 10
t = linspace(-1, 1, M)

y = log(t**2 + 1)/(1.5 + sin(3*t))

# Form least squares problem, A x = y:
A = zeros((M,N))
for i, fun in enumerate(funs):
    A[:,i] = fun(t)

# This is numerically unstable (compared to other methods). We are just doing
# it this way for demonstration
x = inv(A.T @ A) @ (A.T @ y)

tplot = linspace(-1, 1, 100)
yplot = zeros(tplot.shape)
for xval, fun in zip(x, funs):
    yplot += xval*fun(tplot)

## Visualization
plot(tplot, yplot, 'b', label="Least squares fit")
plot(t, y, 'r.', markersize=8, label="Data")
legend(frameon=False)
for m in range(M):
    plot([t[m], t[m]], [y[m], A[m,:]@x], 'k--', linewidth=1)

xlabel('$t$')
ylabel('$y$')

show()
