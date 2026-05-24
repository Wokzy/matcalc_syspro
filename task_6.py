import math
import numpy as np


def thomas(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm

    a - sub-diagonal
    b - main diagonal
    c - super-diagonal
    """

    n = b.shape[0]

    scratch = np.zeros(n, dtype=float)
    x = np.zeros(n, dtype=float)

    scratch[0] = c[0] / b[0]
    x[0] = d[0] / b[0]

    for i in range(1, n):
        if i < n - 1:
            scratch[i] = c[i] / (b[i] - a[i - 1] * scratch[i - 1]);
        x[i] = (d[i] - a[i - 1] * x[i - 1]) / (b[i] - a[i - 1] * scratch[i - 1])

    for i in range(n - 2, -1, -1):
        x[i] -= scratch[i] * x[i + 1]

    return x

def solution_1(t0, t1, f, conditions: np.ndarray, der_cond: tuple[bool] = (False, False), n_steps=100) -> tuple[np.ndarray, np.ndarray]:
    """
    Решение ур-я: y'' = f(t)

    der_con = False: y(t0), y(t1) = conditions
    der_con = True: y'(t0), y'(t1) = conditions
    """

    h = (t1 - t0) / n_steps
    t = np.linspace(t0, t1, n_steps + 1)

    # lec9, slide 8 - составляем СЛАУ

    a = np.ones(n_steps) / (h**2) 
    b = np.ones(n_steps + 1) * (-2 / h**2)
    c = np.ones(n_steps) / (h**2)
    d = f(t)

    # −b0u0 + c0u1 = d

    b[0] = 1.0
    c[0] = 0.0
    d[0] = conditions[0]

    b[n_steps] = -1.0
    a[n_steps - 1] = 0.0
    d[n_steps] = conditions[1]

    if der_cond[0]:
        # lec 9, slide 16

        # −2y0 + 2y1 = h**2 * f0 + 2h*cond[0]
        b[0] = -2.0
        c[0] = 2.0
        d[0] = h**2 * f(t0) + 2 * h * conditions[0]

    if der_cond[1]:
        # 2y_{n−1} - 2y_n = h**2 * f_n − 2h*cond[1]

        b[n_steps] = -2.0
        a[n_steps - 1] = 2.0
        d[n_steps] = h**2 * f(t0) - 2 * h * conditions[1]

    # print((np.abs(b[1:-1]) > np.abs(a[1:]) + np.abs(c[:-1])).any())
    # assert np.abs(b[0]) > np.abs(c[0])
    # assert np.abs(b[n_steps]) > np.abs(a[n_steps - 1])

    return thomas(a, b, c, d), t


def solution_dft(f, t1 = 2 * math.pi, n_steps=100):
    t = np.linspace(0.0, t1, n_steps, endpoint=False)

    f_dft = np.fft.fft(f(t))
    k = 2 * math.pi * np.fft.fftfreq(n_steps, d = t1 / n_steps)

    y_dft = np.zeros_like(f_dft, dtype=complex)
    mask = (k != 0)

    y_dft[mask] = -f_dft[mask] / (k[mask]**2)
    y_dft[0] = 0

    y = np.fft.ifft(y_dft)
    return y.real, t


def approximation_tomas():

    t0, t1 = 0.0, math.pi
    f = lambda x: np.sin(x)
    real = lambda x: -np.sin(x)
    real_der = lambda x: -np.cos(x)


    kwargs = [
        {"conditions": np.array([real(t0), real(t1)]), "der_cond": (False, False)},
        {"conditions": np.array([real_der(t0), real(t1)]), "der_cond": (True, False)},
        {"conditions": np.array([real(t0), real_der(t1)]), "der_cond": (False, True)},
        # {"conditions": np.array([-1.0, 0.0]), "der_cond": (True, True)},
    ]


    for args in kwargs:
        print(f'{args=}')

        k = 16
        for i in range(5):
            n = k * (1 << i)

            y_first, t_first = solution_1(t0, t1, f, **args, n_steps=n)
            real_first = real(t_first)

            y_second, t_second = solution_1(t0, t1, f, **args, n_steps=2*n)
            real_second = real(t_second)


            first = np.amax(y_first - real_first)
            second = np.amax(y_second - real_second)

            # print(second)
            print(math.log2(first / second))


def approximation_dft():
    t1 = 2 * math.pi
    f = lambda x: np.sin(x)
    real = lambda x: -np.sin(x)

    k = 16
    for i in range(5):
        n = k * (1 << i)

        y_first, t_first = solution_dft(f, t1, n_steps=n)
        real_first = real(t_first)

        y_second, t_second = solution_dft(f, t1, n_steps=2*n)
        real_second = real(t_second)


        first = np.amax(y_first - real_first)
        second = np.amax(y_second - real_second)

        # print(second)
        print(math.log2(first / second))


if __name__ == "__main__":

    """
    y'' = sin(x)
    y = -sin(x)
    """

    approximation_tomas()
    # approximation_dft()
