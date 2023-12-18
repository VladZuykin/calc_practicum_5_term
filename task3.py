from classes import Matrix
import numpy as np

m = Matrix.from_file("data/3tasktest")
print("Исходная матрица.", m, sep="\n")
print()
ans = Matrix.power_method(m.matrix, eps=1e-10)

print("Мат. пакетные собственные числа и векторы")
print(np.linalg.eig(m.matrix))

print("Степенной метод")
print("Собственное число:", ans.eigenvalue)
print("Найденный единичный собственный вектор x_k:", ans.eigenvector)
print("Потребовалось шагов:", ans.k)
x_new = m.matrix.dot(ans.eigenvector)
print("Ax_k=", x_new)
print("Ax_k/x_k", x_new / ans.eigenvector)
print()


# Количество итераций QR-алгоритма
k = 100
print("QR-алгоритм")
qr_alg_ans = Matrix.qr_algorithm(m.matrix, k=k)
print("Собственные значения, найденные QR-разложением:", " ".join(map(str, qr_alg_ans.eigenvalues)), sep="\n")
print(
    "Соответствующие собственные векторы, найденные QR-разложением:",
    "\n".join(
        " ".join(map(lambda el: f"({el:+.5f})", qr_alg_ans.eigenvectors[row, :])) for row in range(m.matrix.shape[0])
    ),
    sep="\n"
)

# Номер вектора
i = 1
print(f"Проверка для {i + 1}-го с.ч. и с.в.")
print("Ax =", m.matrix.dot(qr_alg_ans.eigenvectors[:, i]))
print("lambda * x =", qr_alg_ans.eigenvalues[i] * qr_alg_ans.eigenvectors[:, i])
print()

# Количество итераций
k = 2
eig_v = Matrix.inverse_iteration_method(m.matrix, qr_alg_ans.eigenvalues[i], k=k)
print("Метод обратных итераций")
print(f"Этот же вектор, полученный для {i + 1}-го с.ч.:",
      eig_v)
print("Для проверки поделим поэлементно вектор, полученный QR-разложением, на этот вектор:",
      qr_alg_ans.eigenvectors[:, i] / eig_v)
