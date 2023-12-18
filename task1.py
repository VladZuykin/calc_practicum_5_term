import numpy as np

from classes import Matrix

m = Matrix.from_file("data/variant13")
print("Исходная матрица.", m, sep="\n")
if not m.invertible():
    print("Матрица необратима.")
    exit(0)

print("Число обусловленности для матричной нормы, подчинённой первой норме Гёльдера вектора.")
print(m.condition_number(m.matrix_1_subordinate_norm))
print("Число обусловленности для матричной нормы, подчинённой второй норме Гёльдера вектора.")
print(m.condition_number(m.matrix_2_subordinate_norm))
print("Число обусловленности для матричной нормы, подчинённой max |x| норме вектора.")
print(m.condition_number(m.matrix_infty_subordinate_norm))
print()

print("Матрица после прямого хода Гаусса.", m.fwd_eliminated, sep="\n")
print("Определитель:", m.determinant)
print("Вектор решения, полученный методом Гаусса:", m.gaussian_solution)
print()
print("Обратная матрица.", m.inverse_matrix, sep="\n")
print()
print("Количество операций при прямом методе гаусса:", m.fwd_elimination_operations_num)
print("Количество операций при обратном методе гаусса:", m.back_substitution_operations_num)
print()

# Последний пункт перового занятия
n = 30
np_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        np_matrix[i][j] = 1 / (i + j + 1)
x = np.ones(n)

# Полученное b
b = np_matrix.dot(x)
print("Полученное b последнего пункта задания 1.0:", b)

matrix = Matrix(np_matrix, b)
print("Число обусловленности для матричной нормы, подчинённой первой норме Гёльдера вектора.")
print(matrix.condition_number(matrix.matrix_1_subordinate_norm))
print("Число обусловленности для матричной нормы, подчинённой второй норме Гёльдера вектора.")
print(matrix.condition_number(matrix.matrix_2_subordinate_norm))
print("Число обусловленности для матричной нормы, подчинённой max |x| норме вектора.")
print(matrix.condition_number(matrix.matrix_infty_subordinate_norm))
print("Полученное решение:", matrix.gaussian_solution)











