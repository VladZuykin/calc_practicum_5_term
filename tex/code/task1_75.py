from classes import Matrix

m = Matrix.from_file("data/cholesky_matrix")

print("Исходная матрица.", m, sep="\n")
print()

sym, positive = m.is_symmetric(), m.is_positive_determined()

if sym:
    print("Матрица симметрична.")
    if positive:
        print("Матрица положительно определена.")
    else:
        print("Матрица не является положительно определенной.")
else:
    print("Матриц несимметрична.")

answer = m.solve_cholesky()
print("Вектор решения, полученный методом Холецкого:", answer.value)
print("Количество операций:", answer.operations_num)
