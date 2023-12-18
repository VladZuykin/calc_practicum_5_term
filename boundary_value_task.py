# Решение уравнения y'' + p(x)y' + q(x)y = f(x), 0 < x < 0, y(0) = a, y(1) = b
from pprint import pprint

import numpy as np

from classes import Matrix


def p(x):
    return x ** 3


def q(x):
    return -np.log(1 + x)


def f(x):
    return x ** 1 / 3


def get_x_row(x_0: float, x_n_m_1: float, n: int):
    matrix = np.zeros(n)
    if n == 1:
        matrix[0] = x_0
        return matrix

    for i in range(n):
        matrix[i] = x_0 + i * (x_n_m_1 - x_0) / (n - 1)
    return matrix


def get_function_row(f: callable, x_row):
    rows = x_row.shape[0]
    res = np.zeros(rows)
    if rows == 1:
        res[0] = x_row[0]
        return res

    for i in range(rows):
        res[i] = f(x_row[i])
    return res


def get_boundary_value_problem_matrix(p_row, q_row, f_row, h: float, n: int, a, b):
    matrix = np.zeros((n + 1, n + 1))
    bias = np.zeros(n + 1)
    matrix[0, 0] = 1
    bias[0] = a
    matrix[n, n] = 1
    bias[n] = b
    for k in range(1, n):
        a_k = 1 / h ** 2 + (p_row[k] / 2 / h)
        c_k = 1 / h ** 2 - (p_row[k] / 2 / h)
        b_k = -2 / h ** 2 + q_row[k]
        matrix[k, k - 1] = c_k
        matrix[k, k] = b_k
        matrix[k, k + 1] = a_k
        bias[k] = f_row[k]
    return Matrix(matrix, bias)


def main():
    n = 16
    h = 1 / n
    a = 9
    b = 7
    x_row = get_x_row(0, 1, n + 1)
    p_row, q_row, f_row = get_function_row(p, x_row), get_function_row(q, x_row), get_function_row(f, x_row)
    tridiagonal_matrix = get_boundary_value_problem_matrix(p_row, q_row, f_row, h, n, a, b)
    # print("Получившаяся матрица")
    # print(tridiagonal_matrix)
    solution = tridiagonal_matrix.tridiagonal_method()
    if tridiagonal_matrix.predominant_main_diagonal():
        print("Решение существует и устойчиво.")
        # print(solution.value)
        z = zip(x_row, solution.value)
        pprint(tuple(z))
    elif solution.value is not None:
        print("Решение существует")
        print(solution.value)
    else:
        print("Решение не удаётся найти.")


if __name__ == '__main__':
    main()
