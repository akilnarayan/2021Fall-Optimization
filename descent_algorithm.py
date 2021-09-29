"""
Demonstrates rudimentary descent algorithm, along with good/bad choices for
1. Initialization
2. Descent direction
3. Stepsize
4. Termination criterion

To see how these all affect things, change "good_X" to "bad_X" on any of lines 55-58.
"""

from numpy import sin, cos, exp, linspace, meshgrid, zeros, reshape, vstack, min, max
from numpy.linalg import norm
from matplotlib.pyplot import figure, contour, plot, semilogy, subplot, show, xlabel, ylabel, title, subplots_adjust
from matplotlib.cm import jet

from descent_algorithm_utils import good_init, good_descent, good_stepsize, good_termination
from descent_algorithm_utils import  bad_init,  bad_descent,  bad_stepsize,  bad_termination

# The function and its gradient
def f(xy):
    if len(xy.shape) > 1:
        x = xy[:,0]
        y = xy[:,1]
    else:
        x = xy[0]
        y = xy[1]

    fxy = cos(x + 0.5*y)*cos(0.5*x + y)
    fxy += (10 - 10*exp(-(3*x**2 + y**2)))
    return fxy

def gradf(xy): 
    if len(xy.shape) > 1:
        x = xy[:,0]
        y = xy[:,1]
        gf = zeros([len(x), 2])
    else:
        x = xy[0]
        y = xy[1]
        gf = zeros([1, 2])

    # x derivative
    gf[:,0] += 60*x*exp(-(3*x**2 + y**2))
    gf[:,0] += -sin(x+0.5*y)*cos(0.5*x+y)
    gf[:,0] += -0.5*cos(x+0.5*y)*sin(0.5*x+y)

    # y derivative
    gf[:,1] += 20*y*exp(-(3*x**2 + y**2))
    gf[:,1] += -0.5*sin(x+0.5*y)*cos(0.5*x+y)
    gf[:,1] += -cos(x+0.5*y)*sin(0.5*x+y)

    return gf

# Make some choices
init        = good_init
descent     = good_descent
stepsize    = good_stepsize
terminate   = good_termination

# Metrics
max_iter = 50
xy_k = zeros([max_iter+1, 2])  # iterates
f_k  = zeros(max_iter+1)       # function values
t_k  = zeros(max_iter+1)       # stepsizes
gf_k = zeros(max_iter+1)       # gradient norms

k = 0 # Iteration count

# Initialize
xy_k[0,:] = init()
f_k[0] = f(xy_k[0,:])
gf = gradf(xy_k[0,:])
gf_k[0] = norm(gf)

while (k < max_iter) and not terminate(xy_k, f_k, t_k, gf_k, k):

    # Descent direction
    d_k = descent(gf)

    # Stepsize
    t_k[k] = stepsize(d_k, xy_k[k,:], gf, f)

    # Update
    xy_k[k+1,:] = xy_k[k,:] + t_k[k]*d_k

    # Compute metrics for iterate k+1
    f_k[k+1] = f(xy_k[k+1,:])
    gf = gradf(xy_k[k+1,:])
    gf_k[k+1] = norm(gf)

    k += 1

# Truncate unused allocation
xy_k = xy_k[:k+1,:]
f_k = f_k[:k+1]
gf_k = gf_k[:k+1]
t_k = t_k[:k]

grid = linspace(-7, 7, 300)
x, y = meshgrid(grid, grid)

fxy = reshape(f(vstack((x.flatten(),y.flatten())).T), x.shape)

# Contour plot
fig = figure(figsize=(12, 8))
subplot(2,2,1)
contour(x, y, fxy, 20, cmap=jet)
plot(xy_k[0,0], xy_k[0,1], 'o', color='k', markerfacecolor="None")
plot(xy_k[-1,0], xy_k[-1,1], 'ko-')
xlabel('$x_1$'); ylabel('$x_2$'); title('$f(x_1, x_2)$')

# Surface plot
ax = subplot(2,2,2, projection="3d")
ax.plot_surface(x, y, fxy, antialiased=False, linewidth=0, cmap=jet)
xlabel('$x_1$'); ylabel('$x_2$'); 

ax = subplot(2,2,3)
contour(x, y, fxy, 20, cmap=jet)
plot(xy_k[:,0], xy_k[:,1], 'o-', color='k', markerfacecolor="None")
plot(xy_k[1:,0], xy_k[1:,1], 'ko-')
xmin, ymin = min(xy_k[:,0]) - 0.2, min(xy_k[:,1]) - 0.2
xmax, ymax = max(xy_k[:,0]) + 0.2, max(xy_k[:,1]) + 0.2
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])
xlabel('$x_1$'); ylabel('$x_2$'); 
title('Descent algorithm iteration')

fig2 = figure(figsize=(8, 6))
subplot(3,1,1)
plot(range(k), t_k, 'k.-')
xlabel('Iteration index $k$')
title('Step size $t_k$')

subplot(3,1,2)
plot(range(k+1), f_k, 'k.-')
xlabel('Iteration index $k$')
title('Function value $f(x_k)$')

subplot(3,1,3)
semilogy(range(k+1), gf_k, 'k.-')
xlabel('Iteration index $k$')
title('Gradient norm $\\|\\nabla f(x_k)\\|$')

subplots_adjust(hspace=0.5)

show()
