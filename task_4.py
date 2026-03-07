
import math


def fn1(x):
	return math.sin(math.pi * x**5) / ((1 - x) * x**5)

def fn2(t):
	return math.exp(math.sin(t / (10 * (1 - t))) - math.sqrt(t / (1 - t))) / ((1 - t) ** 2)


def simpson(func, a, b, n = 2048) -> float:
	assert b > a
	assert not (n & 1)

	h = (b - a) / n

	res = 0
	for i in range(1, n, 2):
		res += func(a + (i - 1) * h) + 4 * func(a + i * h) + func(a + (i + 1) * h)

	return res * h / 3



def calc_polinom(x, n: int = 8):
	res = [1, x]

	for i in range(1, n):
		res.append(((2 * i + 1) * x * res[i] - i * res[i - 1]) / (i + 1))

	return res

def calc_polinom_der(x, n: int = 8):
	if n == 0:
		return .0

	r1, r2 = 1.0, .0

	polinom = calc_polinom(x, n)

	for i in range(1, n):
		r1, r2 = ((2 * i + 1) * (polinom[i] + x * r1) - i * r2) / (i + 1), r1

	return r1


def newtons(func, func_der, x, max_iter, n):
	for i in range(max_iter):
		x -= func(x, n)[-1] / func_der(x, n)

	return x


def get_gauss_values(a, b, i, n):
	yi = newtons(calc_polinom, calc_polinom_der, math.cos(math.pi * (4*i - 1) / (4 * n + 2)), 16, n = n)
	xi = (a + b) / 2 + (b - a) * yi / 2
	ci = (b - a) / ((1 - yi**2) * (calc_polinom_der(yi, n=n)**2)) # page 229

	return xi, ci

def gauss(func, a, b, n = 8) -> float:
	assert b > a

	res = 0

	for i in range(1, n + 1):
		xi, ci = get_gauss_values(a, b, i, n)
		# print(func(xi))
		res += ci * func(xi)

	return res


def main():
	print(simpson(fn1, 0.0001, .999, n = 65536))
	print(simpson(fn2, 0.0001, .999, n = 65536))
	print(gauss(fn2, 0, 1, n = 12))
	print(gauss(fn1, 0, 1, n = 5))

	# print(get_gauss_values(-1, 1, 3, 4))


if __name__ == '__main__':
	main()
