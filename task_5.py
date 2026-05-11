import math
import numpy as np
import matplotlib.pyplot as plt


def explicit_euler(f, t0, t1, y0, n_steps):
    h = (t1 - t0) / n_steps
    t = np.linspace(t0, t0 + n_steps * h, n_steps + 1)

    y = np.zeros((n_steps + 1, len(y0)))
    y[0] = y0

    for i in range(n_steps):
        y[i + 1] = y[i] + h * f(t[i], y[i])

    return t, y


def simple_iterations(
    leftside_func, x, correction_coeff=0.0, precision=1e-9, max_iter=128
):

    for i in range(max_iter):
        if np.linalg.norm(leftside_func(x) - x) <= precision:
            return x

        x = (1 - correction_coeff) * leftside_func(x) - correction_coeff * x
        # print(leftside_func(x) - x)

    return x


def jacobian(f, y0):
    eps = np.sqrt(np.finfo(float).eps)

    f0 = f(y0)
    n = len(f0)
    J = np.zeros((n, n))

    for j in range(n):
        y1 = y0.copy()
        y1[j] += eps
        J[:, j] = (f(y1) - f0) / eps

    return J

def newtons(f, jac, x, precision=1e-9, num_iters=128):
    for _ in range(num_iters):
        G = f(x)
        J = jac(x)
        x -= np.linalg.solve(J, G)

    return x


def implicit_euler(f, t0, t1, y0, n_steps, A = None):
    """
    y_n+1 = y_n + h * f(t_n+1, y_n+1)

    """
    h = (t1 - t0) / n_steps
    t = np.linspace(t0, t0 + n_steps * h, n_steps + 1)

    n = len(y0)
    y = np.zeros((n_steps + 1, n))
    y[0] = y0

    for i in range(n_steps):
        y_next = y[i].copy()

        if A is not None:
            # (I - h*A)*y_n+1 = yn - явное выражение при A const
            y[i + 1] =  np.linalg.inv(np.eye(n) - h * A) @ y[i]
        else:
            fn = lambda _y: _y - y[i] - h * f(t[i + 1], _y)
            jac = lambda _y: np.eye(n) - h * jacobian(fn, _y)

            y[i + 1] = newtons(fn, jac, y_next)

        # y[i + 1] = simple_iterations(
        #     leftside_func=lambda _y: y[i] + h * f(t[i + 1], _y),
        #     x=y_next,
        #     correction_coeff=0.001,
        # )

    return t, y


def rk4(f, t0, t1, y0, n_steps):
    h = (t1 - t0) / n_steps
    t = np.linspace(t0, t0 + n_steps * h, n_steps + 1)

    y = np.zeros((n_steps + 1, len(y0)))
    y[0] = y0

    for i in range(n_steps):
        k1 = f(t[i],  y[i])
        k2 = f(t[i] + h / 2, y[i] + k1 * h / 2)
        k3 = f(t[i] + h / 2, y[i] + k2 * h / 2)
        k4 = f(t[i] + h, y[i] + k3 * h)
        y[i + 1] =  y[i] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return t, y


def approximation(f, method, t0, t1, y0, real):
    k = 2
    for i in range(5):
        n = k * (1 << i)

        t_first, y_first = method(f, t0, t1, y0, n_steps=n)
        real_first = real(t_first)

        t_second, y_second = method(f, t0, t1, y0, n_steps=n*2)
        real_second = real(t_second)

        first = np.amax(y_first - real_first)
        second = np.amax(y_second - real_second)

        # print(first / second)
        print(math.log2(first / second))


def first():
    t0, t1 = (0.0, 1.0)
    y0 = np.array([1.0, 0.0])
    n_steps = 10

    """
    y1' =  y2
    y2' = -y1
    Точное решение при y0 = (1, 0): y1 = cos(t), y2 = -sin(t)
    """

    def f(t, y):
        return np.array([y[1], -y[0]])

    def real(t):
        return np.array([np.cos(t), -np.sin(t)]).T

    t_exp, y_exp = explicit_euler(f, t0, t1, y0, n_steps)
    _, y_imp = implicit_euler(f, t0, t1, y0, n_steps)
    _, y_rk4 = rk4(f, t0, t1, y0, n_steps)

    # print(y_exp)
    # print("===========")
    # print(y_imp)
    # print("===========")
    # print(y_rk4)
    # print("===========")
    # print(real(t_exp))

    approximation(f, explicit_euler, t0, t1, y0, real)
    print("===========")
    approximation(f, implicit_euler, t0, t1, y0, real)
    print("===========")
    approximation(f, rk4, t0, t1, y0, real)


def second():
    t0, t1 = (0.0, 1.0)
    y0 = np.array([1.0, -2.0])

    n_steps = 20

    def f(t, y):
        return np.array([998 * y[0] + 1998 * y[1], -999 * y[0] - 1999 * y[1]])

    A = np.array([[998.0, 1998.0], [-998.0, -1999]])
    _, y_exp = explicit_euler(f, t0, t1, y0, n_steps)
    _, y_imp = implicit_euler(f, t0, t1, y0, n_steps, A=A)

    # print(np.linalg.eig(A))

    print(y_exp)
    print("===========")
    print(y_imp)

def third(t0, t1, y0, n_steps = 10, line='-'):

    a, b, c, d = 10, 2, 2, 10
    def f(t, y):
        return np.array([a * y[0] - b * y[0] * y[1],
                         c * y[0] * y[1] - d * y[1]])

    t_rk4, y_rk4 = rk4(f, t0, t1, y0, n_steps)

    A = np.array([[a, -b], [c, -d]])

    # print(y_rk4)
    plt.plot(y_rk4.T[0], y_rk4.T[1], line, label=f'{t0=}, {t1=}, {y0=}')

    # for i in range(n_steps):
    #     print(np.linalg.eig(f(t_rk4[i], y_rk4[i])))

    # print(y_rk4)
    # print(f(t_rk4, y_rk4))
    # print(f(t_rk4, y_rk4))


if __name__ == "__main__":
    # first()
    second()
    # third(0, 1, np.array([1.0, 1.0]), n_steps=100)
    # third(0, 1, np.array([2.0, 2.0]), n_steps=100)
    # third(0, 1, np.array([3.0, 3.0]), n_steps=100)
    # third(0, 1, np.array([4.0, 4.0]), n_steps=100)
    # third(0, 1, np.array([4.95, 4.95]), n_steps=100)
    # plt.legend()
    # plt.show()
