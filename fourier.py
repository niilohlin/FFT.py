
import scipy.integrate as integrate
import scipy
import numpy as np
import matplotlib.pyplot as plt

def complex_quadrature(func, a, b, **kwargs):
    realFunc = lambda x: scipy.real(func(x))
    imagFunc = lambda x: scipy.imag(func(x))

    real_integral = integrate.quad(realFunc, a, b, **kwargs)
    imag_integral = integrate.quad(imagFunc, a, b, **kwargs)
    return (real_integral[0] + 1j * imag_integral[0], real_integral[1:], imag_integral[1:])

def plot(f):
    xValues = list(map(lambda x: x / 10, range(-100, 100)))
    yValues = list(map(f, xValues))
    plt.plot(xValues, yValues)

def plot_complex(f):
    tValues = map(lambda x: x / 10, range(-100, 100))
    complexValues = map(f, tValues)
    xValues = list(map(scipy.real, complexValues))
    yValues = list(map(scipy.imag, complexValues))
    plt.plot(xValues, yValues)

def fourier(f, xi, minimum=-np.inf, maximum=np.inf):
    func = lambda x: (f(x) * np.exp(-2j * np.pi * x * xi))

    return np.absolute(complex_quadrature(func, minimum, maximum)[0])

if __name__ == "__main__":
    func = lambda x: 1 if -1 < x < 1 else 0
    hat = lambda xi: fourier(func, xi)
    plot(func)
    plot(hat)
    plt.show()

