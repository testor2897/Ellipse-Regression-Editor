"""Create and manage result and editor window."""
import csv
import re
import sys
from math import atan2, copysign, cos, isnan, log10, nan, sin, sqrt
from tkinter import messagebox

import matplotlib.patches as mpatches
import numpy as np
from matplotlib.ticker import AutoMinorLocator
from matplotlib.widgets import TextBox
from pylab import (Button, Line2D, Slider, Text, legend, plot, plt, xlabel,
                   ylabel)

import editorConfig as ec
from fileDialogs import get_open_path, get_save_path
from quarticFormula import quartic_formula

if not ("is_number" in globals()):
    def is_number(text):
        """
        Check, whether <text> represents a number.

            Parameters:
                text (str): A string

            Returns:
                bool value (True := text can be converted to number)
        """
        try:
            float(text)  # für Float- und Int-Werte
            return True
        except ValueError:
            return False


def sign(number):
    """Return sign of number."""
    return copysign(1, number)


def para2coord(parameters):
    """
    Calculate a, b, c, d, e and f from Xc, Yc, Rx, Ry and phi (ellipse + circle with a=c and b=0).

    Scale coefficients with residual of center point = result of ellipse function of (xc, yc)
    => better comparison of residuals.

        Parameters:
            parameters (array/list): center [Xc, Yc], radii [Rx, Ry] and angle [phi]

        Returns:
            coefficients (list): [a, b, c, d, e, f]
            ellipse formula a*x^2 + b*xy + c*y^2 + d*x + e*y + f = 0 (circle with c=a and b=0)
    """
    xc, yc, rx, ry, phi = parameters

    # Angle > 180° will be reduced
    if phi > np.pi:
        phi = phi % np.pi

    cosPhi = cos(phi)
    sinPhi = sin(phi)

    a = rx * rx * sinPhi * sinPhi + ry * ry * cosPhi * cosPhi
    b = 2 * (ry * ry - rx * rx) * sinPhi * cosPhi
    c = rx * rx * cosPhi * cosPhi + ry * ry * sinPhi * sinPhi
    d = -2 * a * xc - b * yc
    e = -2 * c * yc - b * xc
    f = a * xc * xc + b * xc * yc + c * yc * yc - rx * rx * ry * ry

    # Normalize coefficients in order to make residuals comparable
    # R(xc, yc) = a * xc² + b * xc * yc + c * yc² + d * xc + e * yc + f != 1
    sf = a * xc * xc + b * xc * yc + c * yc * yc + d * xc + e * yc + f
    result = np.array([a, b, c, d, e, f])
    return result / sf


def ellipse_center(parameters):
    """
    Calculate center of ellipse from coefficients of ellipse equation.

        Parameters:
            parameters (array/list): [a, b, c, d, e, f]

        Returns:
            center (array): xc, yc
    """
    a, b, c, d, e, f = parameters[0:6]

    num = b * b - 4 * a * c
    if num == 0:
        return np.array([nan, nan])
    xc = (c * d * 2.0 - b * e) / num
    yc = (a * e * 2.0 - b * d) / num
    return np.array([xc, yc])


def ellipse_axis_length(parameters):
    """
    Calculate axes of ellipse from coefficients of ellipse equation.

        Parameters:
            parameters (array/list): [a, b, c, d, e, f]

        Returns:
            axes (array): rx, ry
    """
    a, b, c, d, e, f = parameters[0:6]

    num = b * b - 4 * a * c
    if num == 0:
        return np.array([nan, nan])

    part1 = sqrt(b * b + (a - c) * (a - c))
    part2 = 2.0 * (a * e * e + c * d * d - b * d * e + num * f)
    rx = -sqrt(abs(part2 * (a + c - part1))) / num
    ry = -sqrt(abs(part2 * (a + c + part1))) / num
    axes = np.abs(np.array([rx, ry]))
    return axes


def ellipse_angle_of_rotation(parameters):
    """
    Calculate rotation of ellipse from coefficients of ellipse equation.

        Parameters:
            parameters (array/list): [a, b, c, d, e, f]

        Returns:
            phi (float): angle of rotation
    """
    a, b, c, d, e, f = parameters[0:6]

    if round(a - c, 8) == 0:
        phi = -sign(a) * sign(b) * np.pi / 4
    else:
        phi = 0.5 * atan2(b, (a - c))

    if phi < (-np.pi / 2):
        phi = np.pi + phi
    return phi


def expand_parameters(parameters, x=None, y=None):
    """
    Calculate parametric parameters of ellipse from coefficients of ellipse equation.

        Parameters:
            parameters (array/list): [a, b, c, d, e, f]

        Returns:
            result (array): coefficients of ellipse, parametric parameters and sum of errors
    """
    decimalPlaces = 11

    center = ellipse_center(parameters)
    axes = ellipse_axis_length(parameters)
    phi = ellipse_angle_of_rotation(parameters)
    if axes[1] > axes[0]:
        axes = axes[::-1]
        phi = phi - np.pi / 2

    if phi < (-np.pi / 2):
        phi = np.pi + phi

    parameters = para2coord(np.hstack((center, axes, phi)))
    result = np.hstack((parameters[0:6], center, axes, phi))

    # calculate geometric errors (sum of distances^2 and sum of residuals)
    if y is not None:
        sumD = sumErrorVector(result, x, y)
        sumR = sumResiduals(parameters, x, y)
        result = np.hstack((result, sumD, sumR))
    return np.round(result, decimalPlaces)


def EllipseErrorVector(parameters, x_orig, y_orig):
    """
    Calculate geometric errors (distances^2) between points and regression ellipse.

        Parameters:
            parameters (array): coefficients of ellipse (optional), parametric parameters
            x_orig (array): x coordiantes of data points
            y_orig (array): y coordiantes of data points

        Returns:
            result (array): [errorVector, points_x, points_y]
                            first column = distance^2
                            second, third column = x,y of nearest point on ellipse
    """
    # check whether expanded parameters (coefficients of ellipse + parametric parameters) are used
    if len(parameters) >= 11:
        xc, yc, rx, ry, phi = parameters[6:11]
    else:
        xc, yc, rx, ry, phi = parameters[0:5]

    x = x_orig.flatten()
    y = y_orig.flatten()

    if ry < rx:
        minRadius_2 = ry * ry
    else:
        minRadius_2 = rx * rx

    cosphi = cos(-phi)
    sinphi = sin(-phi)
    aE = 1 / (rx * rx)
    bE = 1 / (ry * ry)

    # Translation with xc, yc
    x = x - xc
    y = y - yc

    n = x.size
    points_x = x
    points_y = y
    errorVector = np.full_like(x, 0)

    # Rotation with -phi
    for i in range(n):
        actX = x[i]
        actY = y[i]
        x[i] = cosphi * actX - sinphi * actY
        y[i] = sinphi * actX + cosphi * actY

    a = -aE**2 * bE**2
    b = (2 * aE * bE) * (aE + bE)

    for i in range(n):
        if np.round(x[i], 10) == 0 and np.round(y[i], 10) == 0:
            points_x[i] = 0
            points_y[i] = sign(sign(y[i]) + 0.5) * ry
            errorVector[i] = ry**2
        else:
            x_2 = (x[i] * x[i])
            y_2 = (y[i] * y[i])

            c = aE * bE**2 * x_2 + aE**2 * bE * y_2 - aE**2 - bE**2 - 4 * aE * bE
            d = 2 * (aE + bE - aE * bE * (x_2 + y_2))
            e = aE * x_2 + bE * y_2 - 1

            gamma = quartic_formula(a, b, c, d, e)

            distance_2 = x_2 + y_2
            if distance_2 < minRadius_2:
                distance_2 = minRadius_2
            newX = 0
            newY = 0
            for j in range(1, gamma[0] + 1):
                actY = y[i] / (1 - gamma[j] * bE)
                actX = x[i] / (1 - gamma[j] * aE)
                actdistance_2 = (x[i] - actX)**2 + (y[i] - actY)**2
                if actdistance_2 <= distance_2:
                    newX = actX
                    newY = actY
                    distance_2 = actdistance_2
            points_x[i] = newX
            points_y[i] = newY
            errorVector[i] = distance_2

    cosphi = cos(phi)
    sinphi = sin(phi)
    # Rotation with phi
    actX = points_x
    actY = points_y
    points_x = cosphi * actX - sinphi * actY
    points_y = sinphi * actX + cosphi * actY

    # retranslation with xc, yc
    # Translation with xc, yc
    points_x += xc
    points_y += yc

    # first column = distance^2
    # second, third column = x,y of point on ellipse
    result = [errorVector, points_x, points_y]
    return result


def sumErrorVector(parameters, x, y):
    """
    Calculate sum of geometric errors (distances^2) between points and regression ellipse.

        Parameters:
            parameters (array): coefficients of ellipse, parametric parameters (optional)
            x (array): x coordiantes of data points
            y (array): y coordiantes of data points

        Returns:
            result (array): sum of geometric errors (distances^2)
    """
    if parameters.size < 11:
        parameters = expand_parameters(parameters)
    distances_2 = EllipseErrorVector(parameters, x, y)[0]
    return np.sum(distances_2)


def sumResiduals(parameters, x, y):
    """
    Calculate sum of residuals^2.

        Parameters:
            parameters (array): coefficients of ellipse
            x (array): x coordiantes of data points
            y (array): y coordiantes of data points

        Returns:
            result (array): sum of residuals^2
    """
    a, b, c, d, e, f = parameters[0:6]
    return np.sum((a * x ** 2 + b * x * y + c * y ** 2 + d * x + e * y + f) ** 2)


def superscript(text):
    """Convert text [0123456789+-] in superscript text."""
    super = list(map(chr, [8304, 185, 178, 179, 8308, 8309, 8310,
                           8311, 8312, 8313, 8314, 32, 8315, 32, 32]))
    return "".join([super[ord(c) - ord('0')] for c in str(text)])


def print_parameter(parameters, text="", fileobject=None):
    """
    Output of regression results on console or file.

        Parameters:
            parameters (array): coefficients of ellipse, parametric parameters and sum of errors
                                (use expand_parameters before!)
            text (str): legend title of this print
            fileobject (fileobject): file for logging the results

    """
    chLCR = chr(9532)  # LIGHT VERTICAL AND HORIZONTAL = cross
    chLH = chr(9472)  # LIGHT HORIZONTAL
    chLV = chr(9474)  # LIGHT VERTICAL

    if text == "":
        cTitles = "a b c d e f xc yc rx ry phi[-] Dist.\u00b2 Res.\u00b2".split()
        if fileobject is not None:
            fileobject.write(",".join(cTitles) + "\n")
        else:
            print()
            print("Ellipse fitting")
            print("a\u00b7x\u00b2 + b\u00b7xy + c\u00b7y\u00b2 + d\u00b7x + e\u00b7y + f = 0")
            print()
            header = "i" + chLV
            i = 0
            for headertext in cTitles:
                i += 1
                if i == 6:
                    actEnd = chLV
                    i = 1
                else:
                    actEnd = " "
                header += ("{0: <8}").format(headertext) + actEnd
            print(header.rstrip())
            line = chLH + chLCR + 53 * chLH + chLCR + 44 * chLH + chLCR + 15 * chLH
            print(line)
    else:
        print(("{0:<1}").format(text), end=chLV)

    if parameters is None:
        return

    i = 0
    j = 0
    if fileobject is not None:
        formatCode = "{:.6g}"
        fileText = ""
        for result in parameters:
            if isnan(result):
                fileText += "-,"
            if abs(round(result, 9)) == 0.0:
                fileText += "0,"
            else:
                fileText += formatCode.format(result) + ","
        fileobject.write(fileText[:-1] + "\n")

    for result in parameters:
        i += 1
        if isnan(result):
            formattedResult = "-"
        elif abs(round(result, 9)) == 0.0:
            formattedResult = "0"
        else:
            # format phi with fixed format deleting trailing zeros
            if j + i == 12:
                formatCode = '{:.3f}'
                formattedResult = formatCode.format(result).rstrip('0').rstrip('.')
            else:
                decPlaces = int(log10(abs(result) + 1e-9))
                if decPlaces >= 10:
                    formatCode = "{:.2g}"
                elif decPlaces >= -1:
                    formatCode = "{:.3g}"
                else:
                    formatCode = "{:.2e}"
                formattedResult = (formatCode.format(result).replace("e-0", "e-").replace("e+0", "e"))
                index = formattedResult.find("e")
                if index > -1:
                    # superscript characters are not supported on every output console / file
                    if fileobject is None:
                        formattedResult = (formattedResult[0: index + 1] +
                                           superscript(formattedResult[index + 1:]))
                    else:
                        formattedResult = (formattedResult[0: index + 1] +
                                           formattedResult[index + 1:])
        if i == 6:
            actEnd = chLV
            i = 1
            j += 6
        else:
            actEnd = " "
        if j + i == 15:
            ec.width = len(formattedResult)
        print("{0: <{width}}".format(formattedResult, width=ec.width), end=actEnd)
    ec.width = 8
    print()


def prepare_plot():
    """Prepare Matplotlib output (no regression data yet)."""
    ec.maxIndex = -1
    ec.colorindex = 0
    ec.width = 8
    ec.parameterTextarray = []
    ec.parameterArray = []
    ec.editArtists = []
    ec.editModus = False
    ec.customText = None

    fig, ax = plt.subplots()
    ec.fig = fig
    ec.ax = ax
    ec.fig.subplots_adjust(left=0.05, top=0.85, right=0.65, bottom=0.25)
    ec.plt = plt
    ax.set_picker(True)
    plot(ec.x, ec.y, 'bo', label='data points')
    xlabel('x')
    ylabel('y')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_facecolor('snow')
    # old default: ax.set_facecolor('lightgray')

    # calculate optimized values for x- and y-axis
    minX = np.min(ec.x)
    maxX = np.max(ec.x)
    minY = np.min(ec.y)
    maxY = np.max(ec.y)
    deltaX = abs(maxX - minX)
    deltaY = abs(maxY - minY)
    midX = 0.5 * (maxX + minX)
    midY = 0.5 * (maxY + minY)
    # make sure both axis are scaled uniformely (aspect ration 16 / 10)
    if 1.6 * deltaY > deltaX:
        deltaX = deltaY
    minX = midX - 1.6 * deltaX
    maxX = midX + 1.6 * deltaX
    minY = midY - deltaX
    maxY = midY + deltaX
    plt.xlim(minX, maxX)
    plt.ylim(minY, maxY)


def plot_ellipse(para, text="", color=None, updateOnly=False):
    """
    Plot specific regression ellipse (each separate).

        Parameters:
            parameters (array): coefficients of ellipse, parametric parameters and sum of errors
                                (use expand_parameters beforehand for new data)
            text (str): legend title of this specific plot
            color (str): color for this specific plot, otherwise automatic coloring
            updateOnly (bool): True=data is already plotted, update only
    """
    if updateOnly:
        para = expand_parameters(para, ec.x, ec.y)
    else:
        ec.maxIndex += 1

    if (color is not None or updateOnly):
        actColor = color
    else:
        actColor = ec.colors[ec.colorindex]
        if actColor == "w":
            actColor = "gray"
        ec.colorindex += 1

    # Ergebnisse auf der Console ausgeben
    if text != "Custom":
        print_parameter(para, str(ec.maxIndex))

    # caclulate points of fitted ellipse
    xc, yc = np.round(para[6:8], 3)
    rx, ry = np.round(para[8:10], 3)
    phi = np.round(para[10], 4)
    phiGrad = np.round(para[10] * 180 / np.pi, 5)

    results = para[0:6]
    pArray = []
    for result in results:
        if isnan(result):
            formattedResult = "-"
        elif abs(round(result, 8)) == 0.0:
            formattedResult = "0"
        else:
            decPlaces = int(log10(abs(result) + 1e-8))
            formatCode = '{:.4g}' if (decPlaces >= -1) else '{:.3e}'
            formattedResult = formatCode.format(result).replace('e-0', 'e-').replace('e+0', 'e')
            index = formattedResult.find("e")
            if index > -1:
                formattedResult = (formattedResult[0:index + 1] +
                                   superscript(formattedResult[index + 1:]))
            if formattedResult[0:1] == "-":
                formattedResult = "- " + formattedResult[1:]
            else:
                formattedResult = "+ " + formattedResult
        pArray.append(formattedResult)

    results = np.hstack((para[6:11], phiGrad, para[11:13]))
    tArray = []
    actIndex = 0
    for result in results:
        if isnan(result):
            formattedResult = "-"
        elif abs(round(result, 8)) == 0.0:
            formattedResult = "0"
        else:
            if actIndex < 4 or actIndex > 5:
                decPlaces = int(log10(abs(result) + 1e-8))
                formatCode = '{:.4g}' if (decPlaces >= -1) else '{:.3e}'
                formattedResult = formatCode.format(result).replace('e-0', 'e-').replace('e+0', 'e')
                index = formattedResult.find("e")
                if index > -1:
                    formattedResult = (formattedResult[0:index + 1] +
                                       superscript(formattedResult[index + 1:]))
            else:
                formatCode = '{:.3f}'
                formattedResult = formatCode.format(result).rstrip('0').rstrip('.')
        tArray.append(formattedResult.ljust(ec.width, ' '))
        actIndex += 1
    tArray[5] = tArray[5].replace(' ', '\u00b0', 1)

    if text != "":
        actNrtext = str(ec.maxIndex) + ":   "

    xx = xc + rx * np.cos(ec.t) * cos(phi) - ry * np.sin(ec.t) * sin(phi)
    yy = yc + rx * np.cos(ec.t) * sin(phi) + ry * np.sin(ec.t) * cos(phi)

    # calculate error vector
    errorVector, nearX, nearY = EllipseErrorVector(para, ec.x, ec.y)

    # calculate x- and y-axis
    n = int(len(xx) / 2)
    axis1x = np.array([xx[0], xx[n]])
    axis1y = np.array([yy[0], yy[n]])
    n = int(len(xx) / 4)
    axis2x = np.array([xx[n], xx[3 * n]])
    axis2y = np.array([yy[n], yy[3 * n]])

    coeff = actNrtext + eval(repr(r'$x_c=$' + tArray[0] + r'$y_c=$' + tArray[1] + r'$r_x=$' +
                             tArray[2] + r'$r_y=$' + tArray[3] + r'$\alpha$=' + tArray[4] +
                             '\u2259 ' + tArray[5] + r'$\ \ \ \sum d_i^2=$' + tArray[6] +
                             r'$\ \ \sum R_i^2=$' + tArray[7]))

    eq = eval(repr(pArray[0] + r'$\ x^2\ $'))
    if abs(round(para[1], 7)) > 0:
        eq = eq + eval(repr(pArray[1] + r'$\ x \cdot y\ $'))
    eq = eq + eval(repr(pArray[2] + r'$\ y^2\ $'))
    if abs(round(para[3], 7)) > 0:
        eq = eq + eval(repr(pArray[3] + r'$\ x\ $'))
    if abs(round(para[4], 7)) > 0:
        eq = eq + eval(repr(pArray[4] + r'$\ y\ $'))
    eq = eq + eval(repr(pArray[5] + r'$\ =\ 0$'))

    if updateOnly:
        ec.editArtists[0].set_data(xx, yy)
        ec.editArtists[1].set_data(xc, yc)
        ec.editArtists[2].set_data(nearX, nearY)
        ec.editArtists[3].set_data(axis1x, axis1y)
        ec.editArtists[4].set_data(axis2x, axis2y)
        ec.editArtists[5].set_data(axis1x[0], axis1y[0])
        ec.editArtists[6].set_data(axis1x[1], axis1y[1])
        ec.editArtists[7].set_data(axis2x[0], axis2y[0])
        ec.editArtists[8].set_data(axis2x[1], axis2y[1])
        if text == "Custom":
            ec.customText.set_text(coeff)
            ec.customText.set_visible(True)
            for i in range(1, len(ec.actLabel)):
                if ec.actLabel[i][2:].strip()[0:6] == "Custom":
                    customIndex = i
                    break
            ec.actEquation[customIndex] = eq
            ec.tb_parameter.set_text(ec.actEquation[customIndex].split(r'$\ \ \ ')[0])
            ec.tb_parameter.set_color(ec.actColor[customIndex])
    else:
        actArtists = []
        actLine, = ec.ax.plot(xx, yy, color=actColor, dashes=[ec.colorindex, 2.5 * ec.colorindex],
                              linewidth=1, label=actNrtext + text, picker=True)
        actArtists.append(actLine)
        actLine, = ec.ax.plot(xc, yc, '.', color=actColor)
        actArtists.append(actLine)
        actLine, = ec.ax.plot(nearX, nearY, 'x', color=actColor, ms=5)
        actArtists.append(actLine)
        actLine, = ec.ax.plot(axis1x, axis1y, ':', color='orange', linewidth=0.5)
        actArtists.append(actLine)
        actLine, = ec.ax.plot(axis2x, axis2y, ':', color='orange', linewidth=0.5)
        actArtists.append(actLine)
        actLine, = ec.ax.plot(axis1x[0], axis1y[0], '+', color='tab:gray')
        actArtists.append(actLine)
        actLine, = ec.ax.plot(axis1x[1], axis1y[1], '+', color='tab:gray')
        actArtists.append(actLine)
        actLine, = ec.ax.plot(axis2x[0], axis2y[0], '+', color='tab:gray')
        actArtists.append(actLine)
        actLine, = ec.ax.plot(axis2x[1], axis2y[1], '+', color='tab:gray')
        actArtists.append(actLine)

        if text == "Custom":
            for thisArtist in actArtists:
                if thisArtist.get_linestyle() != "None":
                    thisArtist.set_linestyle('solid')
                thisArtist.set_linewidth(2.0)
                thisArtist.set_picker(False)
                thisArtist.set_visible(False)
            ec.editArtists = actArtists

        ec.parameterTextarray.append((coeff, actColor, eq, actNrtext + text))
        ec.parameterArray.append(para)


def configure_plot():
    """Configure plot after calling plot_ellipse()."""
    formatCode = '{:.3g}'

    # set general plot parameter
    ec.legend = legend(loc='lower right', fontsize="9", labelcolor='linecolor', facecolor="lightyellow")
    ec.fig.canvas.manager.set_window_title('Direct Least Squares Fitting of Ellipses')
    ec.fig.patch.set_facecolor('white')
    for line in ec.legend.get_lines():
        if not line.get_visible():
            line.set_visible(True)

    actTitle = "Elliptical Regression (" + ec.title + ")"
    formula = (r'$\binom{x(t)}{y(t)} = $' + r'$\binom{x_c+r_x \cdot cos(t) \cdot cos(\alpha)' +
            r'-r_y \cdot sin(t) \cdot sin(\alpha)}{y_c+r_x \cdot cos(t)\cdot sin(\alpha)' +
            r'+r_y \cdot sin(t) \cdot cos(\alpha)}$' + '\t' + r'0$\leqq$t<2$\pi$' + '\n')
    plt.title(formula, fontsize=18)
    plt.suptitle(actTitle, fontsize=14, fontweight='bold', ha='right')

    actBackend = ec.plt.get_backend().lower()
    mng = plt.get_current_fig_manager()

    # for backend "TkAgg" only:
    # due to scaling problems on some systems...
    # do not maximize window before every element is drawn
    if actBackend == "tkagg":
        mng.window.state('normal')
    elif actBackend == "wxagg":
        mng.frame.Maximize(True)
    elif actBackend == "qt4agg":
        mng.window.showMaximized()

    # get dynamic data [created in plot_ellipse()]
    npTextarray = np.asarray(ec.parameterTextarray, dtype=object)
    ec.actText = npTextarray[:, 0:1].flatten()
    ec.actColor = npTextarray[:, 1:2].flatten()
    ec.actEquation = npTextarray[:, 2:3].flatten()
    ec.actLabel = npTextarray[:, 3:4].flatten()
    ec.actPara = np.asarray(ec.parameterArray)
    ec.actXc = ec.actPara[:, 6:7].flatten()
    ec.actYc = ec.actPara[:, 7:8].flatten()
    ec.actRx = ec.actPara[:, 8:9].flatten()
    ec.actRy = ec.actPara[:, 9:10].flatten()
    ec.actAlpha = ec.actPara[:, 10:11].flatten() / np.pi * 180

    parameterText = plt.text(.01, .99, ec.actText[0], horizontalalignment='left',
                             verticalalignment='top', transform=ec.ax.transAxes,
                             fontdict=ec.font, color=ec.actColor[0],
                             family='DejaVu Sans Mono', size='9', linespacing=0.1, picker=True)
    for i in range(1, len(ec.actText)):
        parameterText = ec.ax.annotate("\n" + ec.actText[i], xycoords=parameterText, xy=(0, 0),
                                       va='top', transform=ec.ax.transAxes, color=ec.actColor[i],
                                       family='DejaVu Sans Mono', size='9', linespacing=0.1, picker=True)
    if ec.actLabel[i][2:].strip()[0:6] == "Custom":
        ec.customText = parameterText
        ec.customText.set_visible(False)

    # use later draw event to maximize window ...
    ec.cid = plt.connect('draw_event', on_draw)
    # enable ESC key for quitting
    plt.connect('key_press_event', on_press)
    # enable picking of objects
    ec.cid2 = plt.connect('pick_event', on_pick)

    ec.tb_parameter = plt.text(.01, .01, '', ha='left', va='bottom', transform=ec.ax.transAxes,
                               bbox=dict(boxstyle="square,pad=0.3", ec="none",
                                         fc="lightyellow", alpha=1))

    labelArray = ['Load Parameter', 'Edit Ellipse', 'Save Parameter']

    for i in range(0, 3):
        ax_Text = ec.fig.add_axes([0.69, 0.8 - 0.08 * i, 0.075, 0.04])
        ax_Text.set_frame_on(False)
        actButton = Button(ax_Text, labelArray[i])
        actButton.label.set_fontsize(12)

        fancybox = mpatches.FancyBboxPatch((0, 0), 1, 1, edgecolor="black", facecolor="0.9",
                                           boxstyle="round,pad=0.1", mutation_aspect=3,
                                           transform=ax_Text.transAxes, clip_on=False)

        ax_Text.add_patch(fancybox)
        if i == 0:
            actButton.on_clicked(click_load)
            ec.bt_Load = actButton
        elif i == 1:
            actButton.on_clicked(toggle_edit)
            ec.bt_Edit = actButton
        elif i == 2:
            actButton.on_clicked(click_save)
            ec.ax_Save = ax_Text
            ec.ax_Save.set_visible(False)

    ec.sliderAxes = []
    ec.sliderButtons = []
    ec.sliderArtists = []
    ec.sliderTexts = []

    paraArray = ["ec.actXc", "ec.actYc", "ec.actRx", "ec.actRy", "ec.actAlpha"]
    labelArray = [r'$x_c=\ $', r'$y_c=\ $', r'$r_x=\ $', r'$r_y=\ $', r'$\alpha=$']

    for i in range(0, 5):
        formatCode = '{:.2g}'
        thisPara = eval(paraArray[i])
        ax_slider = ec.fig.add_axes([0.69, 0.57 - 0.075 * i, 0.08, 0.05], visible=False)
        ax_Button_M = ec.fig.add_axes([0.6825, 0.56 - 0.075 * i, 0.0075, 0.02], visible=False)
        ax_Button_P = ec.fig.add_axes([0.77, 0.56 - 0.075 * i, 0.0075, 0.02], visible=False)
        ax_textbox = ec.fig.add_axes([0.69, 0.56 - 0.075 * i, 0.08, 0.02], visible=False)
        tempMin = np.min(thisPara)
        tempMin -= abs(tempMin)
        tempMax = np.max(thisPara)
        tempMax += abs(tempMax)
        tempMin = float(formatCode.format(tempMin))
        tempMax = float(formatCode.format(tempMax))
        tempStep = float(formatCode.format((tempMax - tempMin) / 400.0))
        if i == 4:
            tempMin = -89.999
            tempMax = 90
            tempStep = 0.5
        act_slider = Slider(ax_slider, label=labelArray[i],
                            valmin=tempMin, valinit=tempMin, valmax=tempMax)
        act_slider.valtext.set_visible(False)
        act_mButton = Button(ax_Button_M, "-")
        act_mButton.on_clicked(lambda event, val=-tempStep, slider=act_slider:
                                 clickSlider(event, val, slider))
        act_pButton = Button(ax_Button_P, "+")
        act_pButton.on_clicked(lambda event, val=tempStep, slider=act_slider:
                                 clickSlider(event, val, slider))
        act_textbox = TextBox(ax_textbox, '', initial=tempMin)
        act_slider.on_changed(lambda val, slider=act_slider, textbox=act_textbox,
                              parameter=thisPara: changeSlider(val, textbox, parameter))
        act_textbox.on_text_change(lambda val, slider=act_slider, textbox=act_textbox:
                                   changeText(val, slider, textbox))
        ec.sliderAxes.extend([ax_slider, ax_textbox, ax_Button_M, ax_Button_P])
        ec.sliderButtons.append([act_mButton, act_pButton])
        ec.sliderArtists.append(act_slider)
        ec.sliderTexts.append(act_textbox)

    if actBackend == "tkagg":
        ec.fig.canvas.get_tk_widget().focus_force()
    plt.show()


def on_press(event):
    """Event procedure for <ESC> key."""
    if event.key.lower() == "escape":
        ec.noTextChange = True
        # quit on pressing ESC
        ec.plt.close()
        return

    keySelection = ("left", "right", "home", "end")
    if event.key.lower() in keySelection:
        ec.noTextChange = True
    else:
        ec.noTextChange = False
    sys.stdout.flush()


def on_draw(event):
    """Event procedure for (first) draw_event."""
    actBackend = ec.plt.get_backend().lower()

    # for backend "TkAgg" only:
    # maximize window after every element is drawn
    if actBackend == "tkagg":
        mng = ec.plt.get_current_fig_manager()
        mng.window.state('zoomed')

    # only maximize once, disconnect event
    ec.plt.disconnect(ec.cid)


def on_pick(event):
    """Event procedure for picking elements."""
    # exit in edit mode (no curve picking)
    if ec.editModus:
        return

    if ec.oldLine is not None:
        if isinstance(ec.oldLine, Line2D):
            ec.oldLine.set_linewidth(1)
    if ec.oldText is not None:
        if isinstance(ec.oldText, Text):
            ec.oldText.set_weight('normal')
    if event is None:
        ec.tb_parameter.set_text("")
        ec.oldLine = None
        ec.oldText = None
        return

    if isinstance(event.artist, Line2D):
        thisline = event.artist
        actLabel = thisline.get_label()

        for compText in ec.ax.get_children():
            if isinstance(compText, Text):
                compLabel = compText.get_text().replace("\n", "")
                if actLabel[0:3] == compLabel[0:3]:
                    ec.oldLine = thisline
                    ec.oldLine.set_linewidth(2)
                    actIndex = int(actLabel[0])
                    ec.tb_parameter.set_text(ec.actEquation[actIndex].split(r'$\ \ \ ')[0])
                    ec.tb_parameter.set_color(ec.actColor[actIndex])
                    ec.oldText = compText
                    ec.oldText.set_weight('bold')
                    ec.bt_Edit.label.set_text("Edit Ellipse " + str(actIndex))
                    break

    elif isinstance(event.artist, Text):
        thisText = event.artist
        actLabel = thisText.get_text().replace("\n", "")

        for compLine in ec.ax.lines:
            if isinstance(compLine, Line2D):
                compLabel = compLine.get_label()
                if actLabel[0:3] == compLabel[0:3]:
                    ec.oldLine = compLine
                    ec.oldLine.set_linewidth(2)
                    actIndex = int(actLabel[0])
                    ec.tb_parameter.set_text(ec.actEquation[actIndex].split(r'$\ \ \ ')[0])
                    ec.tb_parameter.set_color(ec.actColor[actIndex])
                    ec.oldText = thisText
                    ec.oldText.set_weight('bold')
                    ec.bt_Edit.label.set_text("Edit Ellipse " + str(actIndex))
                    break
    else:
        ec.tb_parameter.set_text("")
        ec.bt_Edit.label.set_text("Edit Ellipse")
        ec.oldLine = None
        ec.oldText = None
    ec.plt.draw()


def click_load(event):
    """Event procedure for clicking Load Parameter button."""
    wildcard = (("csv files", "*.csv"), ("all files", "*.*"))
    actTitle = "Regression: Load Custom Parameter"
    filename = get_open_path(filetypes=wildcard, title=actTitle)
    if filename != "":
        with open(filename, "r") as csvfile:
            try:
                headerstart = csvfile.readline().replace(";",",").replace(" ",",").replace("\t",",").split(",", 1)[0]
                headerExists = not (is_number(headerstart))
                if headerExists:
                    headerLines = 1
                else:
                    headerLines = 0
                    csvfile.seek(0)
                actLine = csvfile.readline()
                actDelimiter = (
                    csv.Sniffer().sniff(actLine, delimiters=" \t;,").delimiter
                )
            except Exception:
                csvfile.seek(0)
                if headerExists:
                    actLine = csvfile.readline()
                try:
                    actDelimiter = csv.Sniffer().sniff(csvfile.read()).delimiter
                except Exception:
                    messagebox.showerror(
                        title=None, message="Could not determine delimiter!"
                    )
                    return
        try:
            data = np.genfromtxt(filename, delimiter=actDelimiter, skip_header=headerLines,
                                 max_rows=1, dtype=np.float64)
            if len(data) < 11:
                messagebox.showerror(title=None, message="Not enough parameters in: " + filename)
                return
        except Exception:
            messagebox.showerror(title=None, message="ERROR reading file: " + filename)
            return
        for i in range(1, len(ec.actLabel)):
            if ec.actLabel[i][2:].strip()[0:6] == "Custom":
                actIndex = i
                break
        ec.actPara[actIndex] = data
        ec.actXc[actIndex] = data[6]
        ec.actYc[actIndex] = data[7]
        ec.actRx[actIndex] = data[8]
        ec.actRy[actIndex] = data[9]
        ec.actAlpha[actIndex] = data[10] / np.pi * 180
        if ec.editModus:
            toggle_edit(None)
        toggle_edit(None)


def click_save(event):
    """Event procedure for clicking Save Parameter button."""
    wildcard = (("csv files", "*.csv"), ("all files", "*.*"))
    actTitle = 'Regression: Save Custom Parameter'
    filename = get_save_path(filetypes=wildcard, title=actTitle, initialfile="PARA_*.csv")

    if filename != "":
        try:
            fo = open(filename, "w")
            # parameters of the ellipse - cartesian equation
            # a x^2 + b x y + c y^2 + d x + e y + f = 0
            # parameters of the ellipse - parametric equation
            # xc, yc = center, rx, ry = radii, phi = angle of main axis
            # para = [ a, b, c, d, e, f, xc, yc, rx, ry, phi]
            print_parameter(None, "", fo)
            for i in range(1, len(ec.actLabel)):
                if ec.actLabel[i][2:].strip()[0:6] == "Custom":
                    actIndex = i
                    break
            print_parameter(ec.actPara[actIndex], text=actIndex, fileobject=fo)
            n = len(ec.x)
            if n > 0:
                fo.write("Points (x,y)\n")
                arr = np.hstack((ec.x, ec.y))
                for row in arr:
                    text = str(row[0]) + "," + str(row[1]) + "\n"
                    fo.write(text)
            fo.close()
        except Exception:
            messagebox.showerror(title=None, message="ERROR writing file: " + filename)


def toggle_edit(event):
    """Event procedure for clicking Edit Ellipse / Stop Editing button."""
    if ec.editModus:
        for compLine in ec.ax.lines:
            if isinstance(compLine, Line2D):
                compLabel = compLine.get_label()[2:].strip()
                if compLabel[0:6] == "Custom":
                    ec.editModus = False
                    ec.bt_Edit.label.set_text('Edit Ellipse')
                    for thisArtist in ec.sliderAxes:
                        thisArtist.set_visible(False)
                    ec.customText.set_visible(False)
                    ec.ax_Save.set_visible(False)
                    ec.tb_parameter.set_text("")
                    break
        ec.cid2 = ec.plt.connect('pick_event', on_pick)
    else:
        ec.plt.disconnect(ec.cid2)
        ec.customText.set_visible(True)
        for i in range(1, len(ec.actLabel)):
            if ec.actLabel[i][2:].strip()[0:6] == "Custom":
                customIndex = i
                break
        if ec.oldText is not None:
            actIndex = int(ec.oldText.get_text().replace("\n", "")[0])
            ec.actXc[customIndex] = ec.actXc[actIndex]
            ec.actYc[customIndex] = ec.actYc[actIndex]
            ec.actRx[customIndex] = ec.actRx[actIndex]
            ec.actRy[customIndex] = ec.actRy[actIndex]
            ec.actAlpha[customIndex] = ec.actAlpha[actIndex]
            ec.actPara[customIndex] = ec.actPara[actIndex]
            on_pick(None)
        else:
            actIndex = customIndex
        plot_ellipse(ec.actPara[actIndex], text="Custom", updateOnly=True)
        ec.bt_Edit.label.set_text("Stop Editing")
        ec.ax_Save.set_visible(True)
        for thisArtist in ec.editArtists:
            thisArtist.set_visible(True)
        for thisArtist in ec.sliderAxes:
            thisArtist.set_visible(True)
        for thisArtist in ec.sliderArtists:
            thisArtist.eventson = False

        formatCode = '{:.5g}'

        data = ec.actPara[actIndex]

        for i in range(0, 5):
            if i == 4:
                actVal = data[i + 6] / np.pi * 180
            else:
                actVal = data[i + 6]

            actText = formatCode.format(actVal)
            actText = re.sub(r'^0(?=\.)', '', actText, 1).strip()
            ec.sliderTexts[i].set_val(actText)
            ec.sliderArtists[i].set_val(actVal)

        for thisArtist in ec.sliderArtists:
            thisArtist.eventson = True
        ec.editModus = True
    ec.plt.draw()


def clickSlider(event, val, slider):
    """Event procedure for clicking on +/- next to slider (edit mode)."""
    actNumber = slider.val + val

    if actNumber < slider.valmin:
        return
    elif actNumber > slider.valmax:
        return
    slider.set_val(actNumber)


def changeSlider(val, textbox, parameter):
    """Event procedure for changing slider (edit mode)."""
    formatCode = '{:.5g}'

    textbox.eventson = False

    for i in range(1, len(ec.actLabel)):
        if ec.actLabel[i][2:].strip()[0:6] == "Custom":
            customIndex = i
            break

    parameter[customIndex] = val
    actText = formatCode.format(val)
    actText = re.sub(r'^0(?=\.)', '', actText, 1).strip()
    textbox.set_val(actText)
    textbox.eventson = True

    newCoeff = para2coord([ec.actXc[customIndex], ec.actYc[customIndex], ec.actRx[customIndex],
                          ec.actRy[customIndex], ec.actAlpha[customIndex] * np.pi / 180])
    ec.actPara[customIndex] = expand_parameters(newCoeff, ec.x, ec.y)
    plot_ellipse(ec.actPara[customIndex], text="Custom", updateOnly=True)
    ec.plt.draw()


def changeText(val, slider, textbox):
    """Event procedure for changing value via inputbox (edit mode)."""
    if (ec.noTextChange or re.search(r",$|\.$|\..*0$", val) or len(val) == 0):
        ec.noTextChange = False
        return

    formatCode = '{:.5g}'
    textbox.eventson = False

    val = val.replace(",", ".")
    if is_number(val):
        actNumber = float(formatCode.format(float(val)))
        if actNumber < slider.valmin:
            actNumber = float(formatCode.format(slider.valmin))
        elif actNumber > slider.valmax:
            actNumber = float(formatCode.format(slider.valmax))
        slider.set_val(actNumber)
    else:
        actText = formatCode.format(slider.val)
        actText = re.sub(r'^0(?=\.)', '', actText, 1).strip()
        textbox.set_val(actText)

    textbox.eventson = True
