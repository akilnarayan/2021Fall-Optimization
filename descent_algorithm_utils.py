import numpy as np

def good_init():
    return np.array([-2.08,-2.08])

def bad_init():
    return np.array([-2.1,-2.1])

def good_descent(gf):
    return -gf

def bad_descent(gf):
    d = -gf

    # Rotate (negative) gradient counterclockwise by 85 degrees
    th = 85/90*np.pi/2

    rot = np.zeros([2,2])
    rot[0,0], rot[1,1] = np.cos(th), np.cos(th)
    rot[1,0], rot[0,1] = np.sin(th), -np.sin(th)

    return (rot @ d.T).T

def good_stepsize(d, xy0, gfxy, f):
    # Use backtracking linesearch
    # Warning: These are *not* universally good values!
    s = 0.5
    alpha = 0.5
    beta = 0.5

    assert np.inner(d, gfxy) < 0, "Direction d is not a descent direction"

    fold = f(xy0)
    xy = xy0 + s*d
    fnew = f(xy)
    i = 0

    while (fold - fnew) < -alpha*(s*beta**i)*(np.inner(gfxy, d)):
        i += 1
        xy = xy0 + (s*beta**i)*d
        fnew = f(xy)

    return s*(beta**i)

def bad_stepsize(d, xy, gfxy, f):
    # Bad choice of a constant stepsize
    return 1.

def good_termination(xy_k, f_k, t_k, gf_k, k):

    if k == 0:
        return False

    # Just gradient norm
    tolerance = 1e-5
    return gf_k[k] < tolerance

def bad_termination(xy_k, f_k, t_k, gf_k, k):

    if k == 0:
        return False

    # Return difference between successive values of f
    tolerance = 1e-2
    return f_k[k-1] - f_k[k] < tolerance
