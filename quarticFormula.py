"""Calculate solution of quartic equation."""
from math import acos, copysign, cos, log, pi, sqrt

ONE = 1.0
TWO = 2.0
HALF = 0.5


# Berechnung der Maschinengenauigkeit Espilon (d. h. der kleinsten
# positiven Maschinenzahl, fuer die auf dem Rechner gilt:
# 1 + Espilon > 1):
# https://de.wikipedia.org/wiki/Maschinengenauigkeit
def EpsilonComputer():
    """Calcualte precision of computer."""
    temp = TWO
    epsilon = ONE

    while ONE < temp:
        epsilon *= HALF
        temp = ONE + epsilon

    return TWO * epsilon


# quadratic formula
# solutions for: a*x^2 + b*x + c = 0
# result:
# Number of real solutions, solution 1, solution 2
# complex numbers:
# [real part, imaginary part]
def quadratic_formula(a, b, c):
    """Calculate solution of quadratic equation."""
    result = [None] * 3

    # transform polynom => a=1 (x^2 + b*x + c)
    b /= a
    c /= a
    a = 1

    delta = b**2 - 4 * c
    wDelta = sqrt(abs(delta))

    if delta < 0:         # 2 complex solutions
        result[0] = 0
        result[1] = [-b / 2, wDelta / 2]
        result[2] = [-b / 2, -wDelta / 2]
    else:                 # 2 real solutions
        result[0] = 2
        result[1] = (-b + wDelta) / 2
        result[2] = (-b - wDelta) / 2

    # result[0]: number of real solutions
    # result[1]: 1. solution
    # result[2]: 2. solution
    return result


# cardanic formula
# solutions for: a*x^3 + b*x^2 + c*x + d = 0
# result:
# Number of real solutions, solution 1, solution 2, solution 3
# complex numbers:
# [real part, imaginary part]
def cardanic_formula(a, b, c, d):
    """Calculate solution of cardanic equation."""
    sqr3 = 1.73205080756888
    result = [None] * 4

    # transform polynom => a=1 (x^3 + b*x^2 + c*x + d)
    b /= a
    c /= a
    d /= a
    # a = 1

    p = (3 * c - b**2) / 3
    q = (2 * (b**3) - 9 * b * c + 27 * d) / 27

    D = (q / 2)**2 + (p / 3)**3  # discriminant
    term1 = -b / 3

    if D > 0:  # 1 real solution, 2 complex solutions
        result[0] = 1
        u = -q / 2 + sqrt(D)
        u = copysign(1, u) * (abs(u))**(1 / 3)
        v = -q / 2 - sqrt(D)
        v = copysign(1, v) * (abs(v))**(1 / 3)
        result[1] = u + v + term1
        term2 = -(u + v) / 2 + term1
        term3 = (u - v) / 2 * sqr3
        result[2] = [term2, term3]
        result[3] = [term2, -term3]
    elif D == 0:  # 3 real solutions
        result[0] = 3
        term1 = -b / 3
        if p == 0:
            result[1] = result[2] = result[3] = term1
        else:
            term2 = 3 * q / p
            result[1] = term2 + term1
            result[2] = result[3] = -term2 / 2 + term1
    else:
        result[0] = 3
        term1 = -b / 3
        term2 = sqrt(-4 / 3 * p)
        term3 = 1 / 3 * acos(-q / 2 * sqrt(-27 / p**3))
        result[1] = -term2 * cos(term3 + pi / 3) + term1
        result[2] = term2 * cos(term3) - b / 3
        result[3] = -term2 * cos(term3 - pi / 3) + term1

    # result[0]: number of real solutions (first entries)
    # result[1]: 1. solution
    # result[2]: 2. solution
    # result[3]: 3. solution
    return result


# quartic formula
# solutions for: a*x^4 + b*x^3 + c*x^2 + d*x + e = 0
# result:
# Number of real solutions, solution 1, solution 2, solution 3, solution 4
# complex numbers:
# [real part, imaginary part]
def quartic_formula(a, b, c, d, e):
    """Calculate solution of quartic equation."""
    result = [None] * 5
    decimalPlaces = int(-log(EpsilonComputer(), 10)) - 1
    # transform polynom => a=1 (x^4 + b*x^3 + c*x^2 + d*x + e)
    b /= a
    c /= a
    d /= a
    e /= a
    # a = 1

    # reorganize parameters (for programme compatibility)
    # x^4 + a*x^3 + b*x^2 + c*x +d
    a = round(b, decimalPlaces)
    b = c
    c = round(d, decimalPlaces)
    d = round(e, decimalPlaces)

    # an obvious root is 0
    if d == 0:
        tempResult = cardanic_formula(1, a, b, c)
        result[0] = tempResult[0] + 1
        result[1] = 0
        result[2] = tempResult[1]
        result[3] = tempResult[2]
        result[4] = tempResult[3]

    # transform polynom => a=0 (x^4 + 0 + b*x^2 + c*x + d)
    elif a != 0:
        qa = a * a
        Alpha = -3 * qa / 8 + b
        beta = qa * a / 8 - a * b / 2 + c
        gamma = -3 * qa * qa / 256 + qa * b / 16 - a * c / 4 + d

        # solve transformed polynom
        tempResult = quartic_formula(1, 0, Alpha, beta, gamma)

        # back transformation of results
        # real numbers: tempresult(0) = number of real results
        for i in range(1, tempResult[0] + 1):
            tempResult[i] = tempResult[i] - a / 4
        # complex numbers
        for i in range(tempResult[0] + 1, 5):
            tempResult[i][0] = tempResult[i][0] - a / 4
        for i in range(0, 5):
            result[i] = tempResult[i]

    # biquadratic case
    elif c == 0:
        qb = b * b
        term1 = qb / 4 - d
        result[0] = 0

        if term1 < 0:
            term2 = sqrt(qb / 4 + abs(term1))
            term1 = sqrt((term2 - b / 2) / 2)
            term3 = sqrt((term2 + b / 2) / 2)
            result[1] = [term1, term3]
            result[2] = [-term1, -term3]
            result[3] = [term1, -term3]
            result[4] = [-term1, term3]
        else:
            for i in range(1, 3):
                j = (-1)**i
                term2 = -b / 2 + j * sqrt(term1)
                term3 = sqrt(abs(term2))

                if term2 < 0:
                    result[result[0] + 2 - j] = [0, term3]
                    result[result[0] + 3 - j] = [0, -term3]
                else:
                    result[result[0] + 1] = term3
                    result[result[0] + 2] = -term3
                    result[0] = result[0] + 2

    # all other cases
    else:
        tempResult = cardanic_formula(8, 20 * b, 16 * b**2 - 8 * d, 4 * b**3 - 4 * b * d - c**2)
        for i in range(2, tempResult[0] + 1):
            if tempResult[i] > tempResult[1]:
                tempResult[1] = tempResult[i]
        result[0] = 0
        u = sqrt(b + 2 * tempResult[1])

        for j in range(-1, 2, 2):
            v = -j * c / (2 * u) + b + tempResult[1]
            tempResult2 = quadratic_formula(1, j * u, v)
            # real numbers: tempresult(0) = number of real results
            for i in range(1, tempResult2[0] + 1):
                result[result[0] + i] = tempResult2[i]
            # complex numbers
            for i in range(tempResult2[0] + 1, 3):
                result[4 - j - i + result[0]] = tempResult2[i]
            result[0] = result[0] + tempResult2[0]

    # result[0]: number of real solutions (first entries)
    # result[1]: 1. solution
    # result[2]: 2. solution
    # result[3]: 3. solution
    # result[4]: 4. solution
    return result
