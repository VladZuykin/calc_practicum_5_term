from classes import Matrix
import numpy as np

m = Matrix.from_file("data/variant13")
print("Исходная матрица.", m, sep="\n")
print()
print("Полученные L и U матрицы:")
print(m.l_lup_matrix)
print(m.u_lup_matrix)
print()

print("Вектор решения, полученный LU методом:", m.lup_solution)
print(m.lup_determinant)
print("Число арифметических действий при разделении матриц:", m.lup_fwd_operations_num)
print("Число арифметических действий при решении двух уравнений:", m.lup_sub_operations_num)


print()
left_border = -10
right_border = 10
rows = 5

x = np.random.uniform(left_border, right_border, size=rows)
print("Сгенерированный x:", x)

b = m.get_matrix().dot(x)
m_new = Matrix(m.get_matrix(), b)
print("Новая матрица:", m_new, sep="\n")

print("Решение, полученное с помощью LU разложения:")
print(m_new.lup_solution)
