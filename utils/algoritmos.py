import numpy as np
from scipy.optimize import approx_fprime


def bissecao(f, a, b, TOL, iter=100):
    """Método da bisseção para encontrar uma raiz de f em [a, b]."""
    if TOL <= 0:
        raise ValueError("A tolerância deve ser positiva.")

    if a > b:
        a, b = b, a

    fa, fb = f(a), f(b)
    if fa == 0:
        return a, 0
    if fb == 0:
        return b, 0
    if fa * fb > 0:
        raise ValueError("Nenhuma raiz encontrada no intervalo.")

    for i in range(1, iter + 1):
        c = (a + b) / 2.0
        fc = f(c)
        if fc == 0 or (b - a) / 2 <= TOL:
            return c, i

        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    return (a + b) / 2, iter


def pontofixo(a, g, TOL=1e-8):
    """Método de iteração de ponto fixo."""
    x = g(a)
    while abs(x - a) > TOL:
        a = x
        x = g(a)
    return x


def newton_raphson(a, f, TOL=1e-8, df=None):
    """Método de Newton-Raphson para funções escalares."""
    if df is None:
        def dfn(x):
            return approx_fprime(np.array([x]), lambda v: f(v[0]))[0]
    else:
        dfn = df

    g = lambda x: x - f(x) / dfn(x)
    return pontofixo(a, g, TOL)


def secante(a, b, f, TOL=1e-8):
    """Método da secante."""
    g = lambda a, b: (a * f(b) - b * f(a)) / (f(b) - f(a))
    x = g(a, b)
    while abs(x - b) > TOL:
        a, b = b, x
        x = g(a, b)
    return x


def newton_coef(x, y):
    """Coeficientes do polinômio interpolador de Newton."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    a = y.copy()
    for j in range(1, n):
        a[j:n] = (a[j:n] - a[j - 1 : n - 1]) / (x[j:n] - x[: n - j])
    return a


def newton(x, y):
    """Polinômio interpolador de Newton em forma de coeficientes do NumPy."""
    a = newton_coef(x, y)
    p = np.array([a[-1]], dtype=float)
    for k in range(len(a) - 2, -1, -1):
        p = np.convolve(p, np.array([1.0, -x[k]], dtype=float))
        if len(p) > 0:
            p[-1] += a[k]
    return p


def lagrange_add(p, q):
    """Adiciona dois polinômios em coeficientes de ordem decrescente."""
    if len(q) > len(p):
        p, q = q, p
    off = len(p) - len(q)
    res = p[:]
    for i in range(len(q)):
        res[off + i] += q[i]
    return res


def lagrange_mul(p, q):
    """Multiplica dois polinômios em coeficientes de ordem decrescente."""
    deg = (len(p) - 1) + (len(q) - 1)
    res = [0.0] * (deg + 1)
    for i, a in enumerate(p):
        for j, b in enumerate(q):
            res[i + j] += a * b
    return res


def lagrange_scale(p, s):
    """Escala os coeficientes de um polinômio."""
    return [s * c for c in p]


def lagrange(x, y):
    """Polinômio interpolador de Lagrange em coeficientes do NumPy."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    p = [0.0]
    for i in range(n):
        m = [1.0]
        d = 1.0
        xi = x[i]
        for j in range(n):
            if j == i:
                continue
            m = lagrange_mul(m, [1.0, -x[j]])
            d *= xi - x[j]
        pi = lagrange_scale(m, y[i] / d)
        p = lagrange_add(p, pi)
    return np.asarray(p, dtype=float)


def polinomial(x, y):
    """Polinômio interpolador obtido pela solução do sistema Vandermonde."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    V = np.vander(x, N=n, increasing=False)
    a = np.linalg.solve(V, y)
    return a


def lu_pivot(A: np.ndarray):
    """Fatoração LU com pivoteamento parcial."""
    A = A.astype(float).copy()
    n = A.shape[0]
    P = np.eye(n)
    L = np.zeros((n, n))
    U = A.copy()

    for k in range(n):
        pivot = np.argmax(np.abs(U[k:, k])) + k
        if np.isclose(U[pivot, k], 0.0):
            raise np.linalg.LinAlgError("Matriz singular ou quase singular.")
        if pivot != k:
            U[[k, pivot], k:] = U[[pivot, k], k:]
            P[[k, pivot], :] = P[[pivot, k], :]
            if k > 0:
                L[[k, pivot], :k] = L[[pivot, k], :k]

        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]

    np.fill_diagonal(L, 1.0)
    return P, L, U


def lb(L: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Resolução do sistema L*Y = B."""
    n = L.shape[0]
    Y = np.zeros(n)
    for i in range(n):
        Y[i] = B[i] - np.dot(L[i, :i], Y[:i])
    return Y


def uy(U: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Resolução do sistema U*X = Y."""
    n = U.shape[0]
    X = np.zeros(n)
    for i in reversed(range(n)):
        if np.isclose(U[i, i], 0.0):
            raise np.linalg.LinAlgError(
                "U possui pivô nulo; sistema sem solução única."
            )
        X[i] = (Y[i] - np.dot(U[i, i + 1 :], X[i + 1 :])) / U[i, i]
    return X


def lu(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Resolução de um sistema linear utilizando LU com pivoteamento."""
    P, L, U = lu_pivot(A)
    Pb = P @ B
    Y = lb(L, Pb)
    X = uy(U, Y)
    return X


def jacobi(A: np.ndarray, B: np.ndarray, k: int, TOL: float) -> np.ndarray:
    """Método iterativo de Jacobi."""
    A = A.astype(float)
    B = B.astype(float)
    n = B.shape[0]
    X = np.zeros(n)
    Xk = np.zeros(n)

    D = np.diag(A)
    if np.any(D == 0):
        raise ValueError("A possui elementos diagonais nulos; Jacobi pode falhar.")

    R = A - np.diagflat(D)

    for _ in range(k):
        Xk = (B - R @ X) / D
        if np.linalg.norm(Xk - X, ord=2) < TOL:
            return Xk
        X = Xk.copy()
    return X


def seidel(A: np.ndarray, B: np.ndarray, k: int, TOL: float) -> np.ndarray:
    """Método iterativo de Gauss-Seidel."""
    A = A.astype(float)
    B = B.astype(float)
    n = B.shape[0]
    X = np.zeros(n)

    for _ in range(k):
        Xk = X.copy()
        for i in range(n):
            s1 = np.dot(A[i, :i], X[:i])
            s2 = np.dot(A[i, i + 1 :], Xk[i + 1 :])
            X[i] = (B[i] - s1 - s2) / A[i, i]
        if np.linalg.norm(X - Xk, ord=2) < TOL:
            return X
    return X


def JN(x, F, eps=1e-8):
    """Matriz jacobiana numérica de F em x."""
    x = np.asarray(x, dtype=float)
    n = x.size
    Jnum = np.zeros((n, n), dtype=float)

    for i in range(n):
        def fi(v, i=i):
            return F(v)[i]

        Jnum[i, :] = approx_fprime(x, fi, epsilon=eps)

    return Jnum


def G(x, F, J):
    """Iteração de Newton para sistemas usando uma matriz Jacobiana dada."""
    x = np.asarray(x, dtype=float)
    return x - np.linalg.inv(J(x)) @ F(x)


def GN(x, F):
    """Iteração de Newton para sistemas usando Jacobiana numérica."""
    x = np.asarray(x, dtype=float)
    return x - np.linalg.inv(JN(x, F)) @ F(x)


def fixed_point(a, g, TOL=1e-8, iter=1000):
    """Método de ponto fixo para sistemas vetoriais."""
    x = np.asarray(g(a), dtype=float)
    a = np.asarray(a, dtype=float)
    i = 1
    while np.linalg.norm(x - a) > TOL and i < iter:
        a = x
        x = np.asarray(g(a), dtype=float)
        i += 1
    return x


__all__ = [
    "bissecao",
    "pontofixo",
    "newton_raphson",
    "secante",
    "newton_coef",
    "newton",
    "lagrange_add",
    "lagrange_mul",
    "lagrange_scale",
    "lagrange",
    "polinomial",
    "lu_pivot",
    "lb",
    "uy",
    "lu",
    "jacobi",
    "seidel",
    "JN",
    "G",
    "GN",
    "fixed_point",
]
