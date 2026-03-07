import math


def tgx_x(x: float):
	return math.tan(x) - x

def tgx_x_der(x: float):
	return (1 / math.cos(x) ** 2) - 1

def polinom(x: complex):
	return x**3 - 1

def polinom_der(x: complex):
	return 2 * x**2


def bisect(func, rng: tuple[float, float], precision: float = 1e-15, max_iter: int = 1024) -> tuple:
	a, b = rng

	assert func(a) < 0 and func(b) > 0, f"a: {func(a)}, b: {func(b)}"

	if func(a) == 0:
		return (a, a), 0

	if func(b) == 0:
		return (b, b), 0

	for i in range(max_iter):
		if func(b) - func(a) <= precision:
			return (a, b), i

		mid = (a + b) / 2

		if func(mid) == 0:
			return (mid, mid), i

		if func(a) * func(mid) < 0:
			b = mid
		else:
			a = mid

	else:
		raise RuntimeError("Method has not converged with given args")


def simple_iterations(leftside_func, x, correction_coeff=0.0, precision = 1e-15, max_iter=1024):

	for i in range(max_iter):
		if abs(leftside_func(x) - x) <= precision:
			return x, i

		x = (1 - correction_coeff) * leftside_func(x) - correction_coeff * x
	else:
		raise RuntimeError("Method has not converged with given args")

def newtons(func, x, precision, max_iter, func_der=None):
	for i in range(max_iter):
		if abs(func(x)) <= precision:
			return x, i

		x -= func(x) / func_der(x)
	else:
		raise RuntimeError("Method has not converged with given args")


def secant(func, rng, precision, max_iter):
	x1, x2 = rng

	for i in range(max_iter):
		if abs(func(x1)) <= precision:
			return x1, i

		x1, x2 = x1 - func(x1) * (x1 - x2) / (func(x1) - func(x2)), x1
	else:
		raise RuntimeError("Method has not converged with given args")


def solution(func, method, val: float | tuple, precision: float = 1e-12, max_iter: int = 1024, **kwargs):
	res, meta = method(func, val, precision=precision, max_iter=max_iter, **kwargs)

	print(f'Used {meta} iterations')

	return res



def main():
	# res = solution(tgx_x, bisect, (math.pi + .001, 3 * math.pi / 2 - 0.001))
	# res = solution(math.tan, simple_iterations, 4, max_iter=2048) # не решится, потому что производная всегда > 1
	# res = solution(math.tan, simple_iterations, 4.4, max_iter=1024, correction_coeff=.1)
	# res = solution(tgx_x, newtons, 4.4, max_iter=1024, func_der=tgx_x_der)
	# res = solution(tgx_x, secant, (4.4, 4.6), max_iter=1024)
	# print(res, tgx_x(res))


	# res = solution(polinom, newtons, complex(0.9, .0001), func_der=polinom_der)
	# res = solution(polinom, newtons, complex(math.cos(3*math.pi/4) + .0001, math.sin(3*math.pi/4) + .0001), func_der=polinom_der)
	res = solution(polinom, newtons, complex(math.cos(5*math.pi/4) + .0001, math.sin(5*math.pi/4) + .0001), func_der=polinom_der)
	print(res, polinom(res))


if __name__ == '__main__':
	main()
