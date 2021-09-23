# Demonstrates the L curve and a good choice of regularization parameter.

from numpy import linspace, logspace, log, sin, ones, eye, diag, zeros
from numpy.linalg import inv, norm
from numpy.random import randn

from matplotlib.pyplot import plot, xlabel, ylabel, legend, show, subplot, loglog, title, ioff, annotate, xlim, ylim, figure

ioff()

M = 100
t = linspace(-1, 1, M)

sig = 0.05
y = log(t**2 + 1)/(1.5 + sin(3*t)) + sig*randn(M)

# Regularization parameters
lambdas = logspace(-5, 10, 100)

regularization_residuals = zeros(len(lambdas))
misfit_residuals  = zeros(len(lambdas))

# solution x should match the data y
A = eye(M)

# First derivative denoising
R = (diag(ones(M), k=0)[:-1,:] + diag(-ones(M-1), k=1)[:-1,:])/M

for ind, lmbda in enumerate(lambdas):

    # Solving the normal equations explicitly is generally not a good idea
    x = inv(A.T @ A + lmbda*R.T @ R) @ (A.T @ y)

    regularization_residuals[ind] = norm(R @ x)
    misfit_residuals[ind] = norm(A @ x - y)

## Visualization
ind = 66
loglog(misfit_residuals, regularization_residuals, 'r.-')
xlimits, ylimits = xlim(), ylim()
loglog(misfit_residuals[ind], regularization_residuals[ind], 'b.', 'markersize', 10)

annotate('Small $\lambda$', (misfit_residuals[0], regularization_residuals[0]))
annotate('Large $\lambda$', (misfit_residuals[-1], regularization_residuals[-1]))
annotate('$\lambda$={0:1.2e}'.format(lambdas[ind]), (misfit_residuals[ind], regularization_residuals[ind]))

xlim(xlimits)
ylim(ylimits)
xlabel('Data misfit residual $\\|A x - b\\|$')
ylabel('Regularization residual $\\| R x \\|$')

figure()
inds = [0, ind, len(lambdas)-1]
for q, lmbda in enumerate(lambdas[inds]):

    # Solving the normal equations explicitly is generally not a good idea
    x = inv(A.T @ A + lmbda*R.T @ R) @ (A.T @ y)

    ## Visualization
    subplot(int("31"+str(q+1)))
    plot(t, y, 'b.', label="Noisy data")
    plot(t, x, 'r-', linewidth=1, label="Denoised")
    legend(frameon=False)

    xlabel('$t$')
    ylabel('$y$')
    title('$\\lambda={0:1.0e}$'.format(lmbda))

show()
