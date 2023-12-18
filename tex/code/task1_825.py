from classes import Matrix

m = Matrix.from_file("data/variant13_1825")

print("Исходная матрица.", m, sep="\n")
print()

is_tridiagonal, is_tridiagonal_alg_processable = m.is_tridiagonal(), m.tridiagonal_algorithm_processable()

if is_tridiagonal:
    print("Матрица трёхдиагональная.")
    if is_tridiagonal_alg_processable:
        print("Матрица разрешима методом прогонки.")
    else:
        print("Матрица может быть не разрешима методом прогонки.")
else:
    print("Матрица не является трёхдиагональной.")

answer = m.tridiagonal_method()

print("Решение методом Гаусса:", m.solve(Matrix.CalculationTypes.gaussian))
print("Полученный ответ:", answer.value)
print("Количество арифметических операций:", answer.operations_num)

