from pprint import pprint

import numpy as np
from classes import Matrix


def kernel(x, s):
    return x ** 3 + 4 * s


def f(x):
    return np.log(x)


# y(x_i) - lambda*h*\sum_{j=1}^n*w_j*K(x_i, x_j) * y(x_j) = f(x_i), i=1,...,n
def find_solution(quad_coeffs, kernel_matrix, f_i, h, lamb=1):
    if kernel_matrix.shape[0] != kernel_matrix.shape[1]:
        raise IndexError("Матрица прямоугольная.")
    rows = kernel_matrix.shape[0]
    matrix = np.zeros((rows, rows))
    for i in range(rows):
        for j in range(rows):
            if i != j:
                matrix[i][j] = -lamb * quad_coeffs[j] * h * kernel_matrix[i, j]
            else:
                matrix[i][j] = 1 - lamb * quad_coeffs[j] * h * kernel_matrix[i, j]
    ext_matrix = Matrix(matrix, f_i)
    solution = ext_matrix.solve()
    return solution


# k - число узлов
def trapezoid_coeffs(n: int):
    res = np.ones(n)
    res[0] = 1 / 2
    res[n - 1] = 1 / 2
    return res


def get_x_row(x_0: float, x_n_m_1: float, n: int):
    matrix = np.zeros(n)
    if n == 1:
        matrix[0] = x_0
        return matrix

    for i in range(n):
        matrix[i] = x_0 + i * (x_n_m_1 - x_0) / (n - 1)
    return matrix


def get_kernel_matrix(kernel: callable, x_row):
    rows = x_row.shape[0]
    kernel_matrix = np.zeros((rows, rows))
    if rows == 1:
        kernel_matrix[0, 0] = x_row[0]
        return kernel_matrix

    for i in range(rows):
        for j in range(rows):
            kernel_matrix[i, j] = kernel(x_row[i], x_row[j])
    return kernel_matrix


def get_function_row(f: callable, x_row):
    rows = x_row.shape[0]
    res = np.zeros(rows)
    if rows == 1:
        res[0] = x_row[0]
        return res

    for i in range(rows):
        res[i] = f(x_row[i])
    return res


def book_example():  # y(x) = 17/2 + (128/17) cos(2x)
    a = -np.pi
    b = np.pi
    lamb = 3 / (10 * np.pi)
    ker = lambda x, s: 1 / (0.64 * np.cos((x + s) / 2) ** 2 - 1)
    func = lambda x: 25 - 16 * np.sin(x) ** 2
    n = 37  # Число узлов
    h = (b - a) / (n - 1)

    x_row = get_x_row(a, b, n)
    kernel_matrix = get_kernel_matrix(ker, x_row)
    quad_coeffs = trapezoid_coeffs(n)
    f_rows = get_function_row(func, x_row)
    solution = find_solution(quad_coeffs, kernel_matrix, f_rows, h, lamb=lamb)
    z = zip(x_row, solution)
    pprint(tuple(z))


def main():
    x_0 = 1
    x_n_m_1 = 3
    n = 3
    h = (x_n_m_1 - x_0) / (n - 1)
    x_row = get_x_row(x_0, x_n_m_1, n)
    kernel_matrix = get_kernel_matrix(kernel, x_row)
    quad_coeffs = trapezoid_coeffs(n)
    f_rows = get_function_row(f, x_row)
    solution = find_solution(quad_coeffs, kernel_matrix, f_rows, h)
    z = zip(x_row, solution)
    pprint(tuple(z))


if __name__ == '__main__':
    book_example()
