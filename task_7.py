import numpy as np
import matplotlib.pyplot as plt


def build_hamiltonian(U, h):
    """

    (-0.5 * d^2x/dx^2) * psi + U(x)*psi = E * psi
    Используем формулу центральной разности для представления 2й производной

    """
    N = U.shape[0]
    a = np.zeros(N)
    b = np.zeros(N)
    c = np.zeros(N)

    for i in range(1, N-1):
        a[i] = -.5 / (h**2)
        b[i] = 1.0 / (h**2) + U[i]
        c[i] = -.5 / (h**2)

    b[0] = 1.0
    c[0] = 0.0
    a[0] = 0.0
    b[N-1] = 1.0
    a[N-1] = 0.0
    c[N-1] = 0.0

    return a, b, c


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


def mult_H(x, a, b, c):
    n = x.shape[0]
    Hx = np.zeros(n)

    for i in range(n):
        # Hx[i] = a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1]
        left = a[i] * (x[i-1] if i > 0 else 0.0)
        center = b[i] * x[i]
        right = c[i] * (x[i+1] if i < n-1 else 0.0)
        Hx[i] = left + center + right

    return Hx


def inverse_iteration(a, b, c, b0, precision=1e-12, max_iter=100):

    x = b0 / np.linalg.norm(b0)

    eigvalue = 0.0
    for i in range(max_iter):
        d = x.copy()
        x_new = thomas(a, b, c, d)

        # lec10, slide 9
        eigvalue_new = (x @ x_new) / (x_new @ x_new)
        x = x_new / np.linalg.norm(x_new)

        if abs(eigvalue - eigvalue_new) < precision:
            return eigvalue_new, x_new

        eigvalue = eigvalue_new

    return eigvalue, x


def main():
    L = 6.0
    N = 512
    x = np.linspace(-L, L, N)
    h = x[1] - x[0]

    U = x**2 / 2

    a, b, c = build_hamiltonian(U, h)
    E0, psi_num = inverse_iteration(a, b, c, np.random.rand(N), precision=1e-12)

    E_real = 0.5
    psi_real = (1.0 / np.pi)**0.25 * np.exp(-x**2 / 2)

    print(f"Error {abs(E0 - E_real):.2e}")

    plt.plot(x, psi_num, 'b-', label='psi_num')
    plt.plot(x, psi_real, 'r--', label='psi_real')
    plt.xlabel('x')
    plt.ylabel('psi(x)')
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()