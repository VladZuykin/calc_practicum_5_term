from classes import Matrix
import numpy as np

m = Matrix.from_file("data/variant13_2")

print("Исходная матрица.", m, sep="\n")
print()

gaussian_solution = m.solve(Matrix.CalculationTypes.gaussian)
print("Решение методом Гаусса:", gaussian_solution)

ans = m.get_simple_iteration_method_matrices()
print("Матрица B.")
print(ans.B)
print()

print("Вектор c.")
print(ans.c)
print()

x_0 = np.zeros(m.b.shape)
print(f"Проверка решением из метода Гаусса: Bx+c={ans.B.dot(gaussian_solution) + ans.c}")
print(f"||B|| = {Matrix.matrix_infty_subordinate_norm(ans.B)}")

k = 10
ord = 2
print(f"Пусть k={k}, ord нормы={ord}")
print("Априорная оценка x_k:", m.sim_a_priori_estimate(ans.B, x_0, k, ans.c, ord=ord))
sim_ans = m.simple_iteration_method(k)
print("Полученное значение x_k:", sim_ans.value)
print("Фактическая погрешность:", m.sim_actual_error(sim_ans.value, gaussian_solution, ord=ord))
print("Апостериорная погрешность:", sim_ans.aposteriori_estimation)
print()

seidel_ans = m.seidel_method(k=k)
print("Решение методом Зейделя:", seidel_ans.value)
print("Фактическая погрешность:", m.sim_actual_error(seidel_ans.value, gaussian_solution, ord=ord))
