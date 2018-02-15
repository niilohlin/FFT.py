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

xValues = list(map(lambda x: x / 10, range(0, 1024)))
#xValues = list(range(-10, 10))

def plot(f):
    yValues = list(map(f, xValues))
    plt.plot(xValues, yValues)

def fourier(f, xi, minimum=-np.inf, maximum=np.inf):
    func = lambda x: (f(x) * np.exp(-2j * np.pi * x * xi))

    return np.absolute(complex_quadrature(func, minimum, maximum)[0])

def list_to_function(lst):
    return lambda x: lst[int(x)] if 0 <= int(x) < len(lst) else 0

def function_to_list(f):
    return list(map(f, xValues))

def discrete_fourier(array):
    array = np.array(array)
    length = array.shape[0]
    n = np.arange(length)
    k = n.reshape((length, 1))
    M = np.exp(-2j * np.pi * k * n / length)

    return np.dot(M, array)

def fast_fourier_transform(array):
    x = np.asarray(array, dtype=float)
    N = x.shape[0]

    if N % 2 > 0:
        raise ValueError("size of x must be a power of 2")
    elif N <= 32:  # this cutoff should be optimized
        return discrete_fourier(x)
    else:
        X_even = fast_fourier_transform(x[::2])
        X_odd = fast_fourier_transform(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + factor[:N // 2] * X_odd,
                               X_even + factor[N // 2:] * X_odd])

def peak(x, size=1):
    if -size <= x <= 0:
        return x + size
    if 0 <= x <= size:
        return size - x
    return 0

if __name__ == "__main__":
    func = lambda x: peak(x, size=10)
    #hat = lambda xi: fourier(func, xi, min(xValues), max(xValues))
    hat = list_to_function(fast_fourier_transform(function_to_list(peak)))
    plot(func)
    plot(hat)
    plt.show()

