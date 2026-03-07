import matplotlib.pyplot as plt
import numpy as np

import math


def fn(x):
	return 1 / (1 + 25 * x**2)

def build_function(n: int) -> list:
	res = []
	for i in range(n + 1):
		xi = 2*i/n - 1
		yi = fn(xi)

		res.append((xi, yi))

	return res


def build_function_cheb(n: int, a, b) -> list:
	res = []
	for i in range(n + 1):
		xi = (b + a) / 2 + (b - a) * math.cos((2*i + 1) * math.pi / (2 * (n + 1))) / 2
		yi = fn(xi)

		res.append((xi, yi))

	return res


def build_diff_table(indices, func_table, final_res):

	indices_cache = {(i,): func_table[i][1] for i in indices}
	def build_diff_table_inner(indices_loc):
		if indices_loc in indices_cache:
			return indices_cache[indices_loc]

		res = ((build_diff_table_inner(indices_loc[:-1]) - 
				build_diff_table_inner(indices_loc[1:])) /
				(func_table[indices_loc[0]][0] - func_table[indices_loc[-1]][0]))

		if indices_loc == tuple(range(len(indices_loc))):
			final_res.append(res)

		indices_cache[indices_loc] = res

		return res

	return build_diff_table_inner(indices)


def newtons_polynom(x, func_table, diff_table):

	res = 0
	for i in range(len(diff_table) - 1, -1, -1):
		res += diff_table[i]
		res *= (x - func_table[i][0])

	return res + func_table[0][1]

	# res = func_table[0][1]
	# for i in range(len(diff_table)):
	# 	q = diff_table[i]
	# 	for j in range(i + 1):
	# 		q *= (x - func_table[j][0])

	# 	res += q

	# return res

def exp(n: int, a, b):
	func_table = build_function(n)
	diff_table = []

	build_diff_table(tuple(range(len(func_table))), func_table, diff_table)

	func_table_cheb = build_function_cheb(n, a, b)
	diff_table_cheb = []

	build_diff_table(tuple(range(len(func_table_cheb))), func_table_cheb, diff_table_cheb)

	# for i in range(len(func_table)):
	# 	print(newtons_polynom(func_table[i][0], func_table, diff_table), func_table[i][1])

	x_values = np.linspace(a, b, 200)
	y_values = np.abs(newtons_polynom(x_values, func_table, diff_table) - fn(x_values))
	y_values_cheb = np.abs(newtons_polynom(x_values, func_table_cheb, diff_table_cheb) - fn(x_values))

	plt.plot(x_values, y_values, color='green')
	plt.plot(x_values, y_values_cheb, color='red')

def main():


	exp(10, -1.0, 1.0)
	plt.grid(True)
	plt.show()

if __name__ == '__main__':
	main()
