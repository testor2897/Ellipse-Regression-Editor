"""Regression of ellipses from data points."""

import os
import sys

import numpy as np
import scipy as sp
from numpy.linalg import eig, inv
from scipy.optimize import minimize

from editor import (configure_plot, ec, expand_parameters, para2coord,
                    plot_ellipse, prepare_plot, print_parameter,
                    sumErrorVector)
from mainDialog import selectPointData

try:
    from skimage.measure import EllipseModel
    scikitloaded = True
except Exception:
    scikitloaded = False
    print("\nscikit-image not installed (optional)\n")


n = len(sys.argv)
if n > 1:
    for i in range(1, n):
        if sys.argv[i] == "/debug":
            ec.debugModus = True


def scalePoints(x, y, enableMove=True, enableScale=True):
    """
    Scale measuring data for optimized calculations.

        Parameters:
            x, y (array): data points
            enableMove (bool): True=move data central around average values
            enableScale (bool):True=Scale numbers to <2

        Returns:
            [x, y, avgx, avgy, sfx, sfy] scaled data points, move values, scale factors

    """
    if enableMove:
        # calculate average of x and y
        avgx = np.mean(x)
        avgy = np.mean(y)
    else:
        avgx = avgy = 0

    if enableScale:
        # calculating scale factors
        sfx = (np.max(x) - np.min(x)) / 2
        sfy = (np.max(y) - np.min(y)) / 2
        # uniform scale factor = max(sfx, sfy)
        # is easier for unscaling
        # sfx = sfy = max(sfx, sfy)        
    else:
        sfx = sfy = 1

    # creating scaled and centroid data points
    # in order to keep calculation results small
    x = (x - avgx) / sfx
    y = (y - avgy) / sfy
    return [x, y, avgx, avgy, sfx, sfy]


def unscaleParameter(C, avgx, avgy, sfx, sfy):
    """
    Convert scaled results (for scaled data) back to unscaled results.

        Parameters:
            C (array): parameters to be unscaled
            avgx, avgy (float): move values
            sfx, sfy (float):scale factors

        Returns:
            [a, b, c, d, e, f] unscaled parameters

    """
    # apply scale and offset in order to restore original coordinate system
    sfx2 = sfx * sfx
    sfy2 = sfy * sfy
    a = C[0] * sfy2
    b = C[1] * sfx * sfy
    c = C[2] * sfx2
    d = -2 * a * avgx - b * avgy + C[3] * sfx * sfy2
    e = -2 * c * avgy - b * avgx + C[4] * sfx2 * sfy
    f = (-a * avgx * avgx - b * avgx * avgy - c * avgy * avgy - d * avgx -
         e * avgy + C[5] * sfx2 * sfy2)
    return [a, b, c, d, e, f]


# https://www.py4u.net/discuss/261988
# Nicky van Foreest (mod. algorithm)
# http://www.atmos.albany.edu/facstaff/andrea/webmaps/dale/PYcode/fitEllipse2_new.py
# Prof. Andrea Lopez Lang (mod. algorithm)
def fitEllipse(x, y, useNewWay=True):
    """Fit datapoints to ellipse."""
    # creating scaled and centroid data points
    # in order to keep calculation results small
    x, y, avgx, avgy, sfx, sfy = scalePoints(x, y)

    """
    Follows an approach suggested by Fitzgibbon, Pilu and Fischer in
    Fitzgibbon, A.W., Pilu, M., and Fischer R.B., Direct least squares fitting
    of ellipsees, Proc. of the 13th Internation Conference on Pattern
    Recognition, pp 253–257, Vienna, 1996.
    Discussed on http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html
    and uses relationships
    found at http://mathworld.wolfram.com/Ellipse.html.

    ***Update from Authors: Andrew Fitzgibbon, Maurizio Pilu, Bob Fisher
    http://research.microsoft.com/en-us/um/people/awf/ellipse/
    Reference: "Direct Least Squares Fitting of Ellipses", IEEE T-PAMI, 1999
    Citation:  Andrew W. Fitzgibbon, Maurizio Pilu, and Robert B. Fisher
    Direct least-squares fitting of ellipses,
    IEEE Transactions on Pattern Analysis and Machine Intelligence, 21(5),
    476--480, May 1999
    @Article{Fitzgibbon99,
     author = "Fitzgibbon, A.~W.and Pilu, M. and Fisher, R.~B.",
     title = "Direct least-squares fitting of ellipses",
     journal = pami,
     year = 1999,
     volume = 21,
     number = 5,
     month = may,
     pages = "476--480"
    }
    """
    D = np.hstack((x * x, x * y, y * y, x, y, np.ones_like(x)))
    S, C = np.dot(D.T, D), np.zeros([6, 6])
    C[0, 2], C[2, 0], C[1, 1] = 2, 2, -1

    if useNewWay:
        """
        This is a more bulletproof version than that in the paper, incorporating
        scaling to reduce roundoff error, correction of behaviour when the input
        data are on a perfect hyperbola, and returns the geometric parameters
        of the ellipse, rather than the coefficients of the quadratic form.
        """

        # First break matrix into blocks
        tmpA = S[0:3, 0:3]
        tmpB = S[0:3, 3:6]
        tmpC = S[3:, 3:]
        tmpD = C[0:3, 0:3]
        tmpE = np.dot(inv(tmpC), tmpB.conj().T)
        tmpF = np.dot(tmpB, tmpE)

        E, V = eig(np.dot(inv(tmpD), (tmpA - tmpF)))
        # Find the negative (as det(tmpD) < 0) eigenvalue
        n = np.argmax(E)
        # Extract eigenvector corresponding to negative eigenvalue
        C = V[:, n]
        # Recover the bottom half...
        tmpE = -1 * tmpE
        evec_y = np.dot(tmpE, C)
        param = np.concatenate((C, evec_y))

    else:
        # corrected version
        E, V = eig(np.dot(inv(S), C))
        # Find the negative (as det(tmpD) < 0) eigenvalue
        n = np.argmax(E)
        #  Extract eigenvector corresponding to negative eigenvalue
        param = V[:, n]

    result = unscaleParameter(param, avgx, avgy, sfx, sfy)
    return result


# A fast and robust ellipse detector based on top-down least-square fitting
# Yongtao Wang et. al.
def fitEllipse2(x, y):
    """Fit datapoints to ellipse."""
    # creating scaled and centroid data points
    # in order to keep calculation results small
    x, y, avgx, avgy, sfx, sfy = scalePoints(x, y)

    Sx = np.sum(x[:])
    Sy = np.sum(y[:])
    Sxx = np.sum(x[:] * x[:])
    Syy = np.sum(y[:] * y[:])
    Sxy = np.sum(x[:] * y[:])
    Sxxx = np.sum(x[:] * x[:] * x[:])
    Syyy = np.sum(y[:] * y[:] * y[:])
    Sxxy = np.sum(x[:] * x[:] * y[:])
    Sxyy = np.sum(x[:] * y[:] * y[:])
    Sxxxx = np.sum(x[:] * x[:] * x[:] * x[:])
    Syyyy = np.sum(y[:] * y[:] * y[:] * y[:])
    Sxxyy = np.sum(x[:] * x[:] * y[:] * y[:])
    Sxxxy = np.sum(x[:] * x[:] * x[:] * y[:])
    Sxyyy = np.sum(x[:] * y[:] * y[:] * y[:])

    A = np.array([[Sxxxx, Sxxxy, Sxxyy, Sxxx, Sxxy],
                  [Sxxxy, Sxxyy, Sxyyy, Sxxy, Sxyy],
                  [Sxxyy, Sxyyy, Syyyy, Sxyy, Syyy],
                  [Sxxx, Sxxy, Sxyy, Sxx, Sxy],
                  [Sxxy, Sxyy, Syyy, Sxy, Syy]])
    B = np.array([Sxx, Sxy, Syy, Sx, Sy])
    # C has polynomial coefficients a, b, c, d, e
    C = np.dot(inv(A), B)
    # add f=-1
    C = np.append(C, -1)

    result = unscaleParameter(C, avgx, avgy, sfx, sfy)
    result = checkSymmetryProblem(result, avgx, avgy)

    return result


# Tom Judd
# http://juddzone.com/ALGORITHMS/least_squares_ellipse.html
def fitEllipse3(x, y):
    """Fit datapoints to ellipse."""
    # creating scaled and centroid data points
    # in order to keep calculation results small
    x, y, avgx, avgy, sfx, sfy = scalePoints(x, y)

    J = np.hstack((x * x, x * y, y * y, x, y))
    K = np.ones_like(x)   # column of ones

    JT = J.transpose()
    JTJ = np.dot(JT, J)
    InvJTJ = np.linalg.inv(JTJ)
    # C has polynomial coefficients a, b, c, d, e
    C = np.dot(InvJTJ, np.dot(JT, K))
    # add f=-1
    C = np.append(C, -1)

    result = unscaleParameter(C, avgx, avgy, sfx, sfy)
    result = checkSymmetryProblem(result, avgx, avgy)

    return result


def checkSymmetryProblem(parameters, avgx, avgy):
    """Fix strange symmetry problem (occurs sometimes) of regression."""
    from math import cos, sin

    result = parameters
    parameters = expand_parameters(result)
    # caclulate points of fitted ellipse
    xc, yc = parameters[6:8]
    rx, ry = parameters[8:10]
    phi = parameters[10]

    # Translation with xc, yc
    actX = avgx - xc
    actY = avgy - yc

    # Rotation with -phi
    cosphi = cos(-phi)
    sinphi = sin(-phi)
    x = cosphi * actX - sinphi * actY
    y = sinphi * actX + cosphi * actY

    if x**2 > rx**2:
        x0 = xc + rx * cos(phi)
        y0 = yc + rx * sin(phi)
        xcNew = x0 + rx * cosphi
        ycNew = y0 - rx * sinphi
        result = para2coord([xcNew, ycNew, rx, ry, phi])
    if y**2 > ry**2:
        x0 = xc + ry * sin(phi)
        y0 = yc - ry * cos(phi)
        xcNew = x0 - ry * sinphi
        ycNew = y0 - ry * cosphi
        result = para2coord([xcNew, ycNew, rx, ry, phi])
    return result


# http://work.thaslwanter.at/thLib/html/_modules/fits.html
# Thomas Haslwanter (mod. Algorithm)
def fitEllipse4(x, y):
    """
    Ellipse fit by Taubin's Method.

    Parameters
    ----------
    x : array
        x-coordinates of the ellipse points
    y : array
        y-coordinates of the ellipse points

    Returns
    -------
    C : array
        Ellipse parameters
        C = [a b c d e f]
        is the vector of algebraic parameters of the fitting ellipse:
        ax^2 + bxy + cy^2 +dx + ey + f = 0

    Notes
    -----
    Among fast non-iterative ellipse fitting methods,
    this is perhaps the most accurate and robust.

    This method fits a quadratic curve (conic) to a set of points;
    if points are better approximated by a hyperbola, this fit will
    return a hyperbola. To fit ellipses only, use "Direct Ellipse Fit".

    Published in
    G. Taubin, "Estimation Of Planar Curves, Surfaces And Nonplanar
    Space Curves Defined By Implicit Equations, With
    Applications To Edge And Range Image Segmentation",
    IEEE Trans. PAMI, Vol. 13, pages 1115-1138, (1991)
    """
    # creating scaled and centroid data points
    # in order to keep calculation results small
    x, y, avgx, avgy, sfx, sfy = scalePoints(x, y)

    Z = np.hstack((x ** 2, x * y, y ** 2, x, y, np.ones_like(x)))
    M = Z.T.dot(Z) / len(x)

    P = np.array([[M[0, 0] - M[0, 5] ** 2, M[0, 1] - M[0, 5] * M[1, 5],
                  M[0, 2] - M[0, 5] * M[2, 5], M[0, 3], M[0, 4]],
                  [M[0, 1] - M[0, 5] * M[1, 5], M[1, 1] - M[1, 5] ** 2,
                  M[1, 2] - M[1, 5] * M[2, 5], M[1, 3], M[1, 4]],
                  [M[0, 2] - M[0, 5] * M[2, 5], M[1, 2] - M[1, 5] * M[2, 5],
                  M[2, 2] - M[2, 5] ** 2, M[2, 3], M[2, 4]],
                  [M[0, 3], M[1, 3], M[2, 3], M[3, 3], M[3, 4]],
                  [M[0, 4], M[1, 4], M[2, 4], M[3, 4], M[4, 4]]])

    Q = np.array([[4 * M[0, 5], 2 * M[1, 5], 0, 0, 0],
                 [2 * M[1, 5], M[0, 5] + M[2, 5], 2 * M[1, 5], 0, 0],
                 [0, 2 * M[1, 5], 4 * M[2, 5], 0, 0],
                 [0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 1]])

    E, V = sp.linalg.eig(P, Q)
    sortID = np.argsort(E)
    A = V[:, sortID[0]]
    A = np.hstack((A, -A[:3].T.dot(M[:3, 5])))

    result = unscaleParameter(A, avgx, avgy, sfx, sfy)
    result = checkSymmetryProblem(result, avgx, avgy)
    return result


# Casey
# https://stackoverflow.com/questions/47873759/how-to-fit-a-2d-ellipse-to-given-points
def fitEllipse5(x, y):
    """Fit datapoints to ellipse."""
    # creating scaled and centroid data points
    # in order to keep calculation results small
    x, y, avgx, avgy, sfx, sfy = scalePoints(x, y)

    # Formulate and solve the least squares problem ||Ax - b ||^2
    A = np.hstack([x**2, x * y, y**2, x, y])
    b = np.ones_like(x)
    C = np.linalg.lstsq(A, b, rcond=None)[0].squeeze()
    # add f=-1
    C = np.append(C, -1)

    result = unscaleParameter(C, avgx, avgy, sfx, sfy)
    result = checkSymmetryProblem(result, avgx, avgy)
    return result


# Nicolás Guarín-Zapata
# https://gist.github.com/nicoguaro/9a60896ee40d4450011f2a8e308bc2ef
def fitEllipse6(x, y, method=0):
    """Fit datapoints to ellipse."""
    # get start parameter
    if method == 0:
        coef_0 = fitEllipse(x, y)
    elif method == 2:
        coef_0 = fitEllipse2(x, y)
    else:
        coef_0 = fitEllipse4(x, y)
    cons = {"type": "ineq", "fun": ellip_const}
    opts = {'disp': False, "ftol": 1e-8, "maxiter": 300}
    # distances are used for optimization
    res = minimize(sumErrorVector, coef_0, args=(x, y), method="SLSQP",
                   tol=1e-8, options=opts, constraints=cons)
    # a, b, c, d, e, f = res.x
    return res.x


# https://stackoverflow.com/questions/39693869/fitting-an-ellipse-to-a-set-of-data-points-in-python
def fitEllipse7(x, y):
    """Fit datapoints to ellipse."""
    # creating scaled and centroid data points
    # in order to keep calculation results small
    x, y, avgx, avgy, sfx, sfy = scalePoints(x, y)
    points = np.hstack([x, y])

    ell = EllipseModel()
    ell.estimate(points)
    if ell.params is None:
      return ell.params
      
    xc, yc, a, b, phi = ell.params
    result = para2coord([xc, yc, a, b, phi])
    # unscale results
    #xc = xc * sfx + avgx
    #yc = yc * sfy + avgy
    #a = a * sfx
    #b = b * sfy

    result = para2coord([xc, yc, a, b, phi])
    result = unscaleParameter(result, avgx, avgy, sfx, sfy)
    
    return result


def ellip_const(coef):
    """Constraints for opitimization (scipy.optimize.minimize)."""
    return 4 * coef[0] * coef[2] - coef[1] ** 2


def ellipse_angle_of_rotation2(a):
    """
    Calculate angle of ellipse rotation from coefficients of ellipse equation.

        Parameters:
            parameters (array/list): [a, b, c, d, e, f]

        Returns:
            angle (float): angle of ellipse rotation
    """
    a, b, c = a[0], a[1] / 2, a[2]
    if b == 0:
        if a > c:
            return 0
        else:
            return np.pi / 2
    else:
        if a > c:
            return np.arctan(2 * b / (a - c)) / 2
        else:
            return np.pi / 2 + np.arctan(2 * b / (a - c)) / 2


# ---------------------------------------------------------
#     main programme
#
#     Elliptical regression
#     comparison of algorithms
#     Input data from:
#     - div. samples
#     - new input
#     - edited data
#     - files
#
#     author: Sebastian Mainusch (2021)
#             with help of many others, Thanks!
# ---------------------------------------------------------

# correct path for packaging
homePath = os.path.dirname(__file__) or '.'
homePath = homePath.replace("\\", "/")
os.chdir(homePath)

# Ensure UTF-8 encoding for redirected output
sys.stdout.reconfigure(encoding='utf-8')

# General Settings
np.set_printoptions(precision=4, suppress=True,
                    formatter={'float': '{: 0.4g}'.format})

# Sample data
titleSample = [None] * 11
xSample = [None] * 11
ySample = [None] * 11

titleSample[0] = "Randomized Ellipse"
arc = 0.8
t = np.arange(0, arc * np.pi, 0.04)
xSample[0] = np.round(1.5 * np.cos(t) + 2 + 0.1 * np.random.rand(len(t)), 3)
ySample[0] = np.round(np.sin(t) + 1. + 0.1 * np.random.rand(len(t)), 3)

titleSample[1] = "exact ellipse"
xSample[1] = np.array([-0.5980779, -1, 0, 2, 2, 5])
ySample[1] = np.array([0, 1, -0.4907132, -1, 3, 1])

titleSample[2] = "pretty linear data"
xSample[2] = np.array([3.0751, 2.9036, 2.7675, 2.6505, 2.5423, 2.4352])
ySample[2] = np.array([3.2164, 3.133, 3.0626, 2.9991, 2.9378, 2.8748])

titleSample[3] = "8 data points"
xSample[3] = np.array([128, 256, 440, 640, 768, 896, 1152, 1280])
ySample[3] = np.array([100, 250, 510, 160, 400, 520, 750, 900])

titleSample[4] = "9 data points"
xSample[4] = np.array([3.012, 2.778, 2.747, 2.620, 2.587,
                      2.499, 2.382, 2.209, 1.967])
ySample[4] = np.array([3.347, 3.243, 3.031, 2.996, 2.796,
                      2.809, 2.928, 2.757, 2.519])

titleSample[5] = "9 data points"
xSample[5] = np.array([1.99, 1.18, 4.42, 4.68, 3.15, 4.96, 0.16, 0.66, 4.81])
ySample[5] = np.array([2.28, 3.43, 4.81, 1.65, 4.71, 2.38, 3.67, 1.19, 4.99])

titleSample[6] = "small ellipse section"
xSample[6] = np.array([2.70, 4.30, 0.79, 3.36, 0.30, 0.12, 1.78, 0.49])
ySample[6] = np.array([1.85, 2.26, 3.50, 2.13, 1.31, 1.47, 1.04, 3.02])

titleSample[7] = "8 data points"
xSample[7] = np.array([1, 10, 30, 50, 60, 72, 79, 8])
ySample[7] = np.array([0, 5, 15, 25, 30, 15, 0, -15])

titleSample[8] = "12 data points"
xSample[8] = np.array([-2611.2, -2594.3, -2476.0, -2466.3, -2238.9, -2202.0,
                      -2122.0, -2082.4, -2028.0, -1996.0, -1949.3, -1861.7])
ySample[8] = np.array([-1134.6, -1166.4, -1368.2, -1383.4, -1697.3, -1741.8,
                      -1833.2, -1876.1, -1932.8, -1965.0, -2010.3, -2090.3])

titleSample[9] = "transformed polar coordinates"
theta = np.array([0.0, 0.4488, 0.8976, 1.3464, 1.7952, 2.244, 2.6928, 3.1416,
                  3.5904, 4.0392, 4.488, 4.9368, 5.3856, 5.8344, 6.2832])
r = np.array([4.6073, 2.8383, 1.0795, 0.8545, 0.5177, 0.3130, 0.0945, 0.4303,
                0.3165, 0.4654, 0.5159, 0.7807, 1.2683, 2.5384, 4.7271])
xSample[9] = np.round(r * np.cos(theta), 3)
ySample[9] = np.round(r * np.sin(theta), 3)


def get_ellipse_pts(params, npts=100, tmin=0, tmax=2 * np.pi):
    """
    Return npts points on the ellipse described by the parametric parameters.

        Parameters:
            parameters (array/list): center [Xc, Yc], radii [Rx, Ry] and angle [phi]
            npts (int): number of points to be created
            tmin (float): start angle
            tmax(float): end angle

        Returns:
            x, y (arrays): ellipse points
    """
    Xc, Yc, Rx, Ry, phi = params
    # An numpy.array of the parametric variable t.
    t = np.linspace(tmin, tmax, npts)
    x = Xc + Rx * np.cos(t) * np.cos(phi) - Ry * np.sin(t) * np.sin(phi)
    y = Yc + Rx * np.cos(t) * np.sin(phi) + Ry * np.sin(t) * np.cos(phi)
    return x, y


titleSample[10] = "Example by Christian"
# Test the algorithm with an example elliptical arc.
npts = 100
tmin, tmax = np.pi / 6, 4 * np.pi / 3
Xc, Yc = 4, -3.5
Rx, Ry = 7, 3
phi = np.pi / 4
# Get some points on the ellipse (no need to specify the eccentricity).
xSample[10], ySample[10] = get_ellipse_pts((Xc, Yc, Rx, Ry, phi),
                                           npts, tmin, tmax)
# add some noise to sample points
noise = 0.1
xSample[10] += noise * np.random.normal(size=npts)
ySample[10] += noise * np.random.normal(size=npts)
xSample[10] = np.round(xSample[10], 7)
ySample[10] = np.round(ySample[10], 7)

ec.x, ec.y, ec.title = selectPointData(xSample, ySample, titleSample)


while not (ec.x is None):

    # Auxiliary values for plot
    ec.t = np.arange(0, 2 * np.pi, 0.010471976)

    prepare_plot()

    print("Data Points\nx:\n", ec.x, "\ny:\n", ec.y)
    # Transpose Coordinates an make it 2 dimensional
    ec.x = ec.x[:, np.newaxis]
    ec.y = ec.y[:, np.newaxis]

    # parameters of the ellipse - cartesian equation
    # a x^2 + b x y + c y^2 + d x + e y + f = 0
    # parameters of the ellipse - parametric equation
    # xc, yc = center, rx, ry = radii, phi = angle of main axis
    # para = [ a, b, c, d, e, f, xc, yc, rx, ry, phi]
    print_parameter(None, "")

    para = expand_parameters(fitEllipse(ec.x, ec.y, useNewWay=False), ec.x, ec.y)
    plot_ellipse(para, text="Fitzgibbon et al. => N. van Foreest (mod.)")

    para = expand_parameters(fitEllipse(ec.x, ec.y, useNewWay=True), ec.x, ec.y)
    plot_ellipse(para, text="Fitzgibbon et al. =>  A. Lopez Lang (mod.)")

    para = expand_parameters(fitEllipse2(ec.x, ec.y), ec.x,  ec.y)
    plot_ellipse(para, text="statistical approach => Y. Wang et al. (mod.)")

    para = expand_parameters(fitEllipse3(ec.x, ec.y), ec.x, ec.y)
    plot_ellipse(para, text="statistical approach => T. Judd (mod.)")

    para = expand_parameters(fitEllipse4(ec.x, ec.y), ec.x, ec.y)
    plot_ellipse(para, text="Taubin et al. => T. Haslwanter (mod.)")

    para = expand_parameters(fitEllipse5(ec.x, ec.y), ec.x, ec.y)
    plot_ellipse(para, text="Numpy lstsq => Casey (mod.)")

    para = expand_parameters(fitEllipse6(ec.x, ec.y, method=0), ec.x, ec.y)
    plot_ellipse(para, text="Scipy minimize => N. Guarín-Zapata (mod.)")

    if scikitloaded:
        para = expand_parameters(fitEllipse7(ec.x, ec.y), ec.x, ec.y)
        plot_ellipse(para, text="scikit-image: EllipseModel.estimate")

    plot_ellipse(para, text="Custom", color="darkorange")

    configure_plot()

    print("\nLegend:")
    for thisLabel in ec.actLabel:
        print(thisLabel)

    # show selection menu again
    ec.x, ec.y, ec.title = selectPointData(xSample, ySample, titleSample)

# reset stdout
try:
    sys.stdout.close()
except Exception:
    pass
