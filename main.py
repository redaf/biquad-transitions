#!/bin/env python3

import numpy as np
from numpy import array, ones
from scipy.signal import lfilter, lfilter_zi, lfiltic, butter, iirpeak
import matplotlib.pyplot as plt

def main():
    t = np.linspace(0, 4, 2000)
    x = np.sin(2*np.pi*0.5*t)

    s = len(x) * 3 // 8

    x0 = x[0:s]
    x1 = x[s:]

    b0, a0 = butter(2, 0.01)
    b1, a1 = iirpeak(0.01, 2, 2)
    # z = lfilter_zi(b0, a0)

    xi = [x[0], x[0]]
    yi = np.zeros(2)
    z = lfiltic(b0, a0, yi, xi)

    # BAD
    # ##########################
    y0, z0 = lfilter(b0, a0, x0, zi=z)
    y1, z1 = lfilter(b1, a1, x1, zi=z0)
    y_bad = np.concatenate((y0, y1))

    # GOOD
    # ##########################
    y0, z0 = lfilter(b0, a0, x0, zi=z)

    xm = [x0[-1], x0[-2]]
    ym = [y0[-1], y0[-2]]

    zm = lfiltic(b1, a1, ym, xm)
    zz = bq_new_z(b1, a1, ym, xm)
    print(zm - zz)
    y1, z1 = lfilter(b1, a1, x1, zi=zm)

    y_good = np.concatenate((y0, y1))

    # DF1 (Good as well)
    # ##########################
    # xi = [x[0], x[0]]
    # yi = [0, 0]
    # y0, xm, ym = bq_df1(b0, a0, x0, xi, yi)
    # y1, _, _ = bq_df1(b1, a1, x1, xm, ym)
    # y_good = np.concatenate((y0, y1))

    # PLOT
    # ##########################
    plt.plot(t, x, t, y_bad, t, y_good)
    plt.show()

if __name__ == '__main__':
    main()

def bq_df2t(b, a, x, z):
    y = np.zeros(len(x))
    b0 = b[0]
    b1 = b[1]
    b2 = b[2]
    a0 = a[0]
    a1 = a[1]
    a2 = a[2]
    z1 = z[0]
    z2 = z[1]
    assert(a0 == 1.0)

    for i, xi in enumerate(x):
        y[i] = b0 * x[i] + z1
        z1 = z2 + b1 * x[i] - a1 * y[i]
        z2 = b2 * x[i] - a2 * y[i]

    z[0] = z1
    z[1] = z2
    return y, z

def bq_df1(b, a, x, x_p, y_p):
    y = np.zeros(len(x))
    b0 = b[0]
    b1 = b[1]
    b2 = b[2]
    a0 = a[0]
    a1 = a[1]
    a2 = a[2]
    y1 = y_p[0]
    y2 = y_p[1]
    x1 = x_p[0]
    x2 = x_p[1]
    assert(a0 == 1.0)

    for i, xi in enumerate(x):
        y[i] = b0 * x[i] + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        x2 = x1
        x1 = x[i]
        y2 = y1
        y1 = y[i]

    x_p[0] = x1
    x_p[1] = x2
    y_p[0] = y1
    y_p[1] = y2

    return y, x_p, y_p

# lfiltic implementation
def bq_new_z(b, a, y, x):
    assert(len(b) == 3)
    assert(len(a) == 3)
    assert(len(y) == 2)
    assert(len(x) == 2)

    z = np.zeros(2)

    z[0] = b[1] * x[0] + b[2] * x[1]
    z[1] = b[2] * x[0]
    z[0] -= a[1] * y[0] + a[2] * y[1]
    z[1] -= a[2] * y[0]

    return z
