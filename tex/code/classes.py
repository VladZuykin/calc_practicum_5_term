import enum
import math
from collections import namedtuple

import numpy as np
import numpy.typing as npt
from typing import final, Optional, Union
from exceptions import IncorrectFileError, IsNotSymmetric, IsNotPositiveDetermined, IsNotTridiagonal, IsNotSquare

ans_fields = ("value", "operations_num", "error")
Answer = namedtuple('SubstitutionAnswer', ans_fields, defaults=[None] * len(ans_fields))
simple_it_m_fields = ("B", "c", "D", "D_inv")
SimpleIterationMatricesAnswer = namedtuple("SimpleIterationMatricesAnswer", simple_it_m_fields)
simple_it_fields = ("value", "apriori_estimation", "aposteriori_estimation")
SimpleIterationAnswer = namedtuple("SimpleIterationAnswer", simple_it_fields,
                                   defaults=[None] * len(simple_it_fields))
seidel_fields = ("value", "prev_value")
SeidelMethodAnswer = namedtuple("SeidelMethodAnswer", seidel_fields,
                                defaults=[None] * len(seidel_fields))

power_fields = ("eigenvalue", "eigenvector", "k")
PowerAnswer = namedtuple("PowerAnswer", power_fields)

qr_dec_fields = ("Q", "R")
QRDecAnswer = namedtuple("QRDecAnswer", qr_dec_fields)

qr_alg_fields = ("A_k", "eigenvalues", "eigenvectors")
QRAlgAnswer = namedtuple("QRAlgAnswer", qr_alg_fields)


class Matrix:
    class CalculationTypes(enum.Enum):
        gaussian = 1
        lup = 2

    def __init__(self, matrix, b):
        self._l_lup_matrix = None
        self._u_lup_matrix = None
        self.matrix: final(Optional[npt.ArrayLike]) = np.copy(matrix)
        self.b: final(Optional[npt.ArrayLike]) = np.copy(b)

        # Матрица и порядок колонок после прямого хода Гаусса
        self._fwd_eliminated_matrix = None
        self._fwd_elimination_operations_num = None

        self._lup_fwd_operations_num = None
        self._back_substitution_operations_num = None

        self._gaussian_solution: Optional[npt.ArrayLike] = None
        self._lup_solution: Optional[npt.ArrayLike] = None

        self._lup_sub_operations_num = None

        # Обратная к нашей матрица
        self._inv_matrix: Optional[npt.ArrayLike] = None

        # Матрицы метода простой итерации
        self.sim_b_matrix, self.sim_c_vector, self.sim_d, self.sim_d_inv = [None] * 4

    def invertible(self):
        return np.linalg.det(self.matrix) != 0

    def get_matrix(self):
        return self.matrix.copy()

    @classmethod
    def from_file(cls, filename):
        try:
            with open(filename) as topo_file:
                assumed_cols = None
                line_num = 0
                matrix = None
                b = None
                for line in topo_file:
                    line = line.strip()
                    line_numbers = list(map(float, line.split()))
                    line_cols = len(line_numbers)
                    if assumed_cols is None:
                        assumed_cols = len(line_numbers)
                        matrix = np.zeros((assumed_cols, assumed_cols), dtype=float)
                        matrix[0][:] = line_numbers
                    elif line_cols == 1 and assumed_cols < line_num <= 2 * assumed_cols:
                        b[line_num - assumed_cols - 1], *_ = line_numbers
                    elif line_cols != assumed_cols and line_num == assumed_cols:  # Достигли пробела между матрицей и b
                        b = np.zeros(assumed_cols, dtype=float)
                    elif line_cols == assumed_cols:
                        matrix[line_num][:] = line_numbers
                    elif not line and line_num > 2 * assumed_cols:  # Пустые строчки в конце
                        pass
                    else:
                        raise IncorrectFileError("В матрице ошибка.")
                    line_num += 1
        except Exception as e:
            raise IncorrectFileError(e)
        else:
            return cls(matrix, b)

    def __repr__(self):
        shape = self.matrix.shape
        res = ""

        matrix_cols = getattr(self, "_fwd_eliminated_matrix_cols", None)
        if matrix_cols:
            res += ("[!! " + " ".join(f'{matrix_cols.index(j) + 1: >25}' for j in range(shape[1])) +
                    " | " + f"{'b': >25}" + " !!]" + "\n")
        for i in range(shape[0]):
            res += ("[   " + " ".join(f'{self.matrix[i][j]: >25}' for j in range(shape[1])) + " | " +
                    f'{self.b[i]: >25}' + "   ]\n")
        return res

    def _inner_fwd_elimination(self):
        shape = self.matrix.shape
        not_used_leading_cols = set(range(shape[0]))
        used_leading_cols = []
        self._fwd_elimination_operations_num = 0
        for main_row in range(shape[0]):
            leading_col, leading_value = max(
                enumerate(el for el in self.matrix[main_row][:]),
                key=lambda elem: abs(elem[1]) if elem[0] in not_used_leading_cols else -1
            )  # Находим следующий гл. элемент

            # Убираем столбец
            used_leading_cols.append(leading_col)
            not_used_leading_cols.discard(leading_col)

            for row in range(main_row + 1, shape[0]):
                # Вычитаем l * ряд из нижних рядов
                numerator = self.matrix[row][leading_col] / leading_value
                self.matrix[row][leading_col] = 0
                self._fwd_elimination_operations_num += 1
                for col in not_used_leading_cols:
                    self.matrix[row][col] -= self.matrix[main_row][col] * numerator
                    self._fwd_elimination_operations_num += 2
                self.b[row] -= self.b[main_row] * numerator
                self._fwd_elimination_operations_num += 2

        # Устанавливаем истинный порядок столбцов
        self._fwd_eliminated_matrix_cols = tuple(used_leading_cols)

    @property
    def fwd_elimination_operations_num(self):
        return self.fwd_eliminated._fwd_elimination_operations_num

    @property
    def fwd_eliminated(self):
        if self._fwd_eliminated_matrix is None:
            self._fwd_eliminated_matrix = Matrix(self.matrix, self.b)
            self._fwd_eliminated_matrix._inner_fwd_elimination()
        return self._fwd_eliminated_matrix

    @property
    def inverse_matrix(self):
        if self._inv_matrix is not None:
            return self._inv_matrix
        matrix = np.copy(self.matrix)
        shape = matrix.shape
        self._inv_matrix = np.eye(shape[0])

        # Убираем элементы под диагональю
        for main_row in range(shape[0]):
            for row in range(main_row + 1, shape[0]):
                # Вычитаем l * ряд из нижних рядов
                multiplier = matrix[row][main_row] / matrix[main_row][main_row]
                for col in range(0, shape[1]):
                    matrix[row][col] -= matrix[main_row][col] * multiplier
                    self._inv_matrix[row][col] -= self._inv_matrix[main_row][col] * multiplier
        # Убираем элементы над диагональю
        for main_row in range(shape[0] - 1, 0, -1):
            for row in range(main_row - 1, -1, -1):
                # Вычитаем l * ряд из верхних рядов
                multiplier = matrix[row][main_row] / matrix[main_row][main_row]
                for col in range(0, shape[1]):
                    matrix[row][col] -= matrix[main_row][col] * multiplier
                    self._inv_matrix[row][col] -= self._inv_matrix[main_row][col] * multiplier

        # Умножаем строки на элементы диагонали
        for row in range(shape[0]):
            value = matrix[row][row]
            for col in range(shape[1]):
                matrix[row][col] /= value
                self._inv_matrix[row][col] /= value
        return self._inv_matrix

    def solve(self, method=CalculationTypes.gaussian):
        if method == self.CalculationTypes.gaussian:
            return self.gaussian_solution
        if method == self.CalculationTypes.lup:
            return self.lup_solution

    @property
    def lup_solution(self):
        if self._lup_solution is not None:
            return self._lup_solution
        rows, *_ = self.matrix.shape
        # Ищем U и L
        l_matrix = np.eye(rows)
        u_matrix = np.copy(self.matrix)
        b = np.copy(self.b)

        self._lup_fwd_operations_num = 0
        # Находим матрицы L и U
        for main_row in range(rows):
            for i in range(main_row + 1, rows):
                l_matrix[i][main_row] = u_matrix[i][main_row] / u_matrix[main_row][main_row]
                self._lup_fwd_operations_num += 1
            for i in range(main_row + 1, rows):
                for j in range(main_row + 1, rows):
                    u_matrix[i][j] -= l_matrix[i][main_row] * u_matrix[main_row][j]
                    self._lup_fwd_operations_num += 2
                u_matrix[i][main_row] = 0
        self._l_lup_matrix, self._u_lup_matrix = l_matrix, u_matrix

        # Подстановка L и U матриц
        ans_y = self.back_substitution_lower(l_matrix, b, diagonal_ones=True)
        y = ans_y.value
        ans_x = self.back_substitution_upper(u_matrix, y)
        self._lup_solution = ans_x.value

        self._lup_sub_operations_num = ans_y.operations_num + ans_x.operations_num
        return self._lup_solution

    @property
    def lup_fwd_operations_num(self):
        return self._lup_fwd_operations_num

    @property
    def lup_sub_operations_num(self):
        return self._lup_sub_operations_num

    @property
    def l_lup_matrix(self):
        self.lup_solution
        return self._l_lup_matrix

    @property
    def u_lup_matrix(self):
        self.lup_solution
        return self._u_lup_matrix

    @property
    def gaussian_solution(self):
        if self._gaussian_solution is not None:
            return self._gaussian_solution

        ans = self.back_substitution_upper(
            self.fwd_eliminated.matrix, self.fwd_eliminated.b,
            cols=self.fwd_eliminated._fwd_eliminated_matrix_cols
        )
        self._gaussian_solution = ans.value
        self._back_substitution_operations_num = ans.operations_num

        return self._gaussian_solution

    @property
    def back_substitution_operations_num(self):
        return self._back_substitution_operations_num

    @property
    def determinant(self):
        matrix = self.fwd_eliminated.matrix
        cols = self.fwd_eliminated._fwd_eliminated_matrix_cols
        return math.prod(matrix[i][cols[i]] for i in range(matrix.shape[0])) * self.get_permutation_sign(cols)

    @property
    def lup_determinant(self):
        solution = self.lup_solution
        matrix = self._u_lup_matrix
        return math.prod(matrix[i][i] for i in range(matrix.shape[0]))

    def condition_number(self, norm):
        return norm(self.inverse_matrix) * norm(self.matrix)

    @staticmethod
    def get_permutation_sign(permutation):
        inv = 0
        for i in range(len(permutation)):
            for j in range(i + 1, len(permutation)):
                if permutation[j] < permutation[i]:
                    inv += 1
        if inv % 2 == 0:
            return 1
        return -1

    def solve_cholesky(self, definiteness_accuracy=1e-6, symmetric_accuracy=1e-9):
        if not self.is_symmetric(symmetric_accuracy):
            return Answer(value=None, operations_num=None, error=IsNotSymmetric)
        if not self.is_positive_determined(accuracy=definiteness_accuracy, symmetry_accuracy=symmetric_accuracy):
            return Answer(value=None, operations_num=None, error=IsNotPositiveDetermined)
        operations_num = 0
        l_matrix = np.zeros(self.matrix.shape)
        for l_col in range(self.matrix.shape[0]):
            diag_elem = self.matrix[l_col, l_col]
            for col in range(l_col):
                diag_elem -= l_matrix[l_col, col] ** 2
                operations_num += 2
            l_matrix[l_col, l_col] = np.sqrt(diag_elem)
            operations_num += 1

            for row in range(l_col + 1, self.matrix.shape[1]):
                elem = self.matrix[row, l_col]
                for i in range(l_col):
                    elem -= l_matrix[row, i] * l_matrix[l_col, i]
                    operations_num += 2
                elem /= l_matrix[l_col, l_col]
                operations_num += 1
                l_matrix[row, l_col] = elem

        y_answer = self.back_substitution_lower(l_matrix, self.b)
        y = y_answer.value
        x_answer = self.back_substitution_upper(l_matrix.T, y)
        x = x_answer.value
        operations_num += y_answer.operations_num + x_answer.operations_num
        return Answer(x, operations_num)

    @staticmethod
    def back_substitution_lower(lower_matrix, b, cols=None, diagonal_ones=False):
        operations_num = 0
        rows = lower_matrix.shape[0]
        x = np.zeros(rows)
        for row in range(rows):
            x[row] = b[row]
            for col in range(row):
                if not cols:
                    x[row] -= lower_matrix[row][col] * x[col]
                else:
                    x[row] -= lower_matrix[row][cols[col]] * x[col]
                operations_num += 2
            if not diagonal_ones:
                if not cols:
                    x[row] /= lower_matrix[row][row]
                else:
                    x[row] /= lower_matrix[row][cols[row]]
                operations_num += 1
        if cols:
            x_old = x
            x = np.zeros(rows)
            for i in range(rows):
                x[i] = x_old[cols.index(i)]
        return Answer(x, operations_num)

    @staticmethod
    def back_substitution_upper(upper_matrix, b, cols=None, diagonal_ones=False):
        operations_num = 0
        rows = upper_matrix.shape[0]
        x = np.zeros(rows)
        for row in range(rows - 1, -1, -1):
            x[row] = b[row]
            for col in range(rows - 1, row, -1):
                if not cols:
                    x[row] -= upper_matrix[row][col] * x[col]
                else:
                    x[row] -= upper_matrix[row][cols[col]] * x[col]
                operations_num += 2
            if not diagonal_ones:
                if not cols:
                    x[row] /= upper_matrix[row][row]
                else:
                    x[row] /= upper_matrix[row][cols[row]]
                operations_num += 1
        if cols:
            x_old = x
            x = np.zeros(rows)
            for i in range(rows):
                x[i] = x_old[cols.index(i)]
        return Answer(x, operations_num)

    def is_symmetric(self, accuracy=1e-9):
        if not self.is_square():
            return False
        for row in range(self.matrix.shape[0]):
            for col in range(row + 1, self.matrix.shape[1]):
                if abs(self.matrix[row, col] - self.matrix[col][row]) > accuracy:
                    return False
        return True

    def is_positive_determined(self, accuracy=1e-6, symmetry_accuracy=1e-9):
        if not self.is_symmetric(accuracy=symmetry_accuracy):
            return False
        for row in range(1, self.matrix.shape[0] + 1):
            tmp_m = self.matrix[:row, :row]
            tmp_b = self.b[:row]
            tmp_matrix = Matrix(tmp_m, tmp_b)
            if tmp_matrix.determinant < 0 or abs(tmp_matrix.determinant) <= accuracy:
                return False
        return True

    def is_tridiagonal(self):
        # Проверка первой строки
        for j in range(2, self.matrix.shape[1]):
            if self.matrix[0, j] != 0:
                return False

        # Проверка последней строки
        for j in range(0, self.matrix.shape[1] - 2):
            if self.matrix[self.matrix.shape[0] - 1, j] != 0:
                return False

        # Проверка промежуточных строк
        for i in range(1, self.matrix.shape[0] - 1):
            for j in range(i - 1):
                if self.matrix[i, j] != 0:
                    return False
            for j in range(i + 3, self.matrix.shape[1]):
                if self.matrix[i, j] != 0:
                    return False
        return True

    def tridiagonal_algorithm_processable(self):
        rows = self.matrix.shape[0]
        if not self.is_tridiagonal():
            return False

        if rows == 1:
            return True

        if abs(self.matrix[0, 0]) < abs(self.matrix[0, 1]):
            return False

        for i in range(1, self.matrix.shape[0] - 1):
            if (abs(self.matrix[i, i - 1]) >= abs(self.matrix[i, i]) or
                    abs(self.matrix[i, i]) < abs(self.matrix[i, i - 1]) + abs(self.matrix[i, i + 1])):
                return False

        if abs(self.matrix[rows - 1, rows - 1]) < abs(self.matrix[rows - 1, rows - 2]):
            return False
        return True

    def predominant_main_diagonal(self):
        rows = self.matrix.shape[0]
        if not self.is_tridiagonal():
            return False

        if rows == 1:
            return True

        if abs(self.matrix[0, 0]) < abs(self.matrix[0, 1]):
            return False

        for i in range(1, self.matrix.shape[0] - 1):
            if abs(self.matrix[i, i]) <= abs(self.matrix[i, i - 1]) + abs(self.matrix[i, i + 1]):
                return False

        if abs(self.matrix[rows - 1, rows - 1]) < abs(self.matrix[rows - 1, rows - 2]):
            return False
        return True

    def tridiagonal_method(self):
        if not self.is_square():
            return Answer(None, None, IsNotSquare)
        if not self.is_tridiagonal():
            return Answer(None, None, IsNotTridiagonal)

        # Прямой ход
        operations_num = 0
        rows = self.matrix.shape[0]
        alpha, beta, gamma = np.zeros(self.matrix.shape[1]), np.zeros(self.matrix.shape[1]), np.zeros(
            self.matrix.shape[1])

        gamma[0] = self.matrix[0, 0]
        alpha[0] = -self.matrix[0, 1] / gamma[0]
        beta[0] = self.b[0] / gamma[0]

        operations_num += 3

        for row in range(1, rows - 1):
            gamma[row] = self.matrix[row, row] + self.matrix[row, row - 1] * alpha[row - 1]
            operations_num += 2
            alpha[row] = -self.matrix[row, row + 1] / gamma[row]
            operations_num += 2
            beta[row] = (self.b[row] - self.matrix[row, row - 1] * beta[row - 1]) / gamma[row]
            operations_num += 3

        gamma[rows - 1] = self.matrix[rows - 1, rows - 1] + self.matrix[rows - 1, rows - 2] * alpha[rows - 2]
        operations_num += 2
        beta[rows - 1] = (self.b[rows - 1] - self.matrix[rows - 1, rows - 2] * beta[rows - 2]) / gamma[rows - 1]
        operations_num += 3

        # Обратный ход

        x = np.zeros(rows)
        x[rows - 1] = beta[rows - 1]
        for row in range(rows - 2, -1, -1):
            x[row] = alpha[row] * x[row + 1] + beta[row]
            operations_num += 2

        return Answer(x, operations_num)

    def is_square(self):
        return self.matrix.shape[0] == self.matrix.shape[1]

    @staticmethod
    def sim_a_priori_estimate(matrix: np.ndarray, x_0: np.ndarray,
                              k: int, bias: np.ndarray, ord: Union[int] = 2):
        matrix_norm = Matrix.matrix_subordinate_norm(matrix, ord=ord)
        matrix_norm_in_power = matrix_norm ** k
        x_0_norm = Matrix.holders_vector_norm(x_0, ord=ord)
        bias_norm = Matrix.holders_vector_norm(bias, ord=ord)
        return matrix_norm_in_power * x_0_norm + matrix_norm_in_power / (1 - matrix_norm) * bias_norm

    @staticmethod
    def sim_a_posteriori_estimate(matrix: np.ndarray, x_k: np.ndarray,
                                  x_k_m_1: np.ndarray, ord: Union[int] = 2):
        matrix_norm = Matrix.matrix_subordinate_norm(matrix, ord=ord)
        return matrix_norm / (1 - matrix_norm) * Matrix.holders_vector_norm(x_k - x_k_m_1, ord=ord)

    @staticmethod
    def sim_actual_error(x_k: np.ndarray, x_actual: np.ndarray, ord: Union[int] = 2):
        return Matrix.holders_vector_norm(x_k - x_actual, ord=ord)

    def get_simple_iteration_method_matrices(self):
        if self.sim_b_matrix is not None:
            return SimpleIterationMatricesAnswer(self.sim_b_matrix, self.sim_c_vector, self.sim_d, self.sim_d_inv)
        # Создаём матрицу D
        rows = self.matrix.shape[0]
        d_matrix = np.zeros((rows, rows))
        for row in range(rows):
            d_matrix[row, row] = self.matrix[row, row]

        # Находим обратную к ней
        d_inv = np.eye(rows)
        for row in range(rows):
            d_inv[row, row] /= d_matrix[row, row]

        # Получаем матрицу B и вектор c
        b_matrix = np.zeros((rows, rows))
        c_vector = np.zeros(rows)
        for row in range(rows):
            for col in range(rows):
                if row == col:
                    b_matrix[row, col] = 1
                b_matrix[row, col] -= d_inv[row, row] * self.matrix[row, col]

            c_vector[row] = d_inv[row][row] * self.b[row]

        self.sim_b_matrix, self.sim_c_vector, self.sim_d, self.sim_d_inv = b_matrix, c_vector, d_matrix, d_inv
        return SimpleIterationMatricesAnswer(b_matrix, c_vector, d_matrix, d_inv)

    def simple_iteration_method(self, k: int, x_0=None, ord: Union[int,] = 2):
        if k < 0:
            raise ValueError("k должно быть натуральным")
        if not x_0:
            x_0 = np.zeros(self.b.shape)
        if k == 0:
            return SimpleIterationAnswer(x_0.copy())
        self.get_simple_iteration_method_matrices()

        x_k_m_1 = None
        x_k = x_0.copy()
        for i in range(k):
            x_k_m_1 = x_k
            x_k = self.sim_b_matrix.dot(x_k_m_1) + self.sim_c_vector

        return SimpleIterationAnswer(x_k,
                                     self.sim_a_priori_estimate(self.matrix, x_0, k, self.sim_c_vector, ord=ord),
                                     self.sim_a_posteriori_estimate(self.matrix, x_k, x_k_m_1, ord=ord))

    def seidel_method(self, k: int, x_0=None):
        ans = self.get_simple_iteration_method_matrices()
        if k < 1:
            raise ValueError("ord должно быть натуральным")
        if not x_0:
            x_0 = np.zeros(self.b.shape)
        rows = self.b.shape[0]
        b_matrix, c_vector = ans.B, ans.c
        if k == 0:
            return x_0

        x_k_m_1 = None
        x_k = x_0.copy()
        for it in range(k):
            x_k_m_1 = x_k.copy()
            for i in range(rows):
                element = 0
                for j in range(i):
                    element += b_matrix[i, j] * x_k[j]
                for j in range(i, rows):
                    element += b_matrix[i, j] * x_k_m_1[j]
                element += c_vector[i]
                x_k[i] = element

        return SeidelMethodAnswer(x_k, x_k_m_1)

    @staticmethod
    def power_method(matrix: np.ndarray, eps=1e-6, x_0=None):
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Матрица должна быть квадратной.")
        rows = matrix.shape[1]
        # Создание x_0, если не передано
        if x_0 is None:
            x_0 = np.random.uniform(-1, 1, rows)
            while not any(x_0):
                x_0 = np.random.uniform(-1, 1, rows)
        # Нормирование x_0
        x_0 = x_0 * 1 / Matrix.holders_vector_norm(x_0, ord=2)

        # Нахождение k-го приближения собственного вектора x_k
        k = 0
        x_k = x_0
        # x_k_m_1 = None
        lam_k_m_1 = None
        lam_k = matrix.dot(x_k).T.dot(x_k)
        while lam_k_m_1 is None or abs(lam_k_m_1 - lam_k) > eps:
            lam_k_m_1 = lam_k
            # x_k_m_1 = x_k
            y_k = matrix.dot(x_k)
            x_k = y_k * 1 / Matrix.holders_vector_norm(y_k, ord=2)
            lam_k = matrix.dot(x_k).T.dot(x_k)
            k += 1
        # Нахождение k-го приближения собственного значения
        eigenvalue = matrix.dot(x_k).T.dot(x_k)
        return PowerAnswer(eigenvalue, x_k, k)

    @staticmethod
    def rotation_method_qr_decomposition(matrix: np.ndarray):
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Матрица должна быть квадратной.")
        rows = matrix.shape[0]
        r_matrix = matrix.copy()
        q_matrix = np.eye(rows)
        for primary_row in range(rows):
            for temp_row in range(primary_row + 1, rows):
                if r_matrix[temp_row, primary_row] == 0:
                    continue  # Пропускаем строку, если целевой элемент матрицы уже равен 0.
                # Находим c и s
                denum = np.sqrt(r_matrix[primary_row, primary_row] ** 2 + r_matrix[temp_row, primary_row] ** 2)
                c = r_matrix[primary_row, primary_row] / denum
                s = r_matrix[temp_row, primary_row] / denum
                # Сохраняем нужные строки R
                primary_row_copied = r_matrix[primary_row, primary_row:].copy()
                temp_row_copied = r_matrix[temp_row, primary_row:].copy()
                # Изменяем R
                r_matrix[primary_row, primary_row:] = c * primary_row_copied + s * temp_row_copied
                r_matrix[temp_row, primary_row:] = -s * primary_row_copied + c * temp_row_copied
                r_matrix[temp_row, primary_row] = 0
                # Сохраняем нужные строки Q
                primary_row_copied = q_matrix[primary_row, :].copy()
                temp_row_copied = q_matrix[temp_row, :].copy()
                # Изменяем Q
                q_matrix[primary_row, :] = c * primary_row_copied + s * temp_row_copied
                q_matrix[temp_row, :] = -s * primary_row_copied + c * temp_row_copied
        # Транспонируем, чтобы получить искомый Q
        q_matrix = q_matrix.T
        return QRDecAnswer(q_matrix, r_matrix)

    @staticmethod
    def qr_algorithm(matrix: np.ndarray, k: int):
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Матрица должна быть квадратной.")
        rows = matrix.shape[0]
        a_k_matrix: np.ndarray = matrix.copy()
        eigenvectors_k: np.ndarray = np.eye(rows)
        for _ in range(k):
            ans = Matrix.rotation_method_qr_decomposition(a_k_matrix)
            q_matrix, r_matrix = ans.Q, ans.R
            a_k_matrix = r_matrix.dot(q_matrix)
            # Получаем приближения с.в.
            eigenvectors_k = eigenvectors_k.dot(q_matrix)
        # Получаем приближения с.ч.
        eigenvalues_k = a_k_matrix.diagonal().copy()
        return QRAlgAnswer(a_k_matrix, eigenvalues_k, eigenvectors_k)

    @staticmethod
    def inverse_iteration_method(matrix: np.ndarray, eigenvalue_estimation: float, k: int, x_0=None):
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Матрица должна быть квадратной.")
        rows = matrix.shape[0]
        if x_0 is None:
            x_0 = np.ones(rows)
        modified_matrix = matrix - eigenvalue_estimation * np.eye(rows)
        x_k = x_0
        for _ in range(k):
            m = Matrix(modified_matrix, x_k)
            y_k_next = m.solve()
            x_k = y_k_next / Matrix.holders_vector_norm(y_k_next, ord=2)
        return x_k

    @staticmethod
    def holders_vector_norm(x: np.ndarray, ord: Union[int,] = 2):
        if ord is np.infty:
            return np.max(np.abs(x))
        if ord < 1:
            raise ValueError("ord должно быть натуральным")
        rows_nums = x.shape[0]
        value = 0
        for row in range(rows_nums):
            value += abs(x[row]) ** ord
        return value ** (1 / ord)

    @staticmethod
    def matrix_subordinate_norm(matrix: np.ndarray, ord: Union[int,] = 2):
        if ord is np.infty:
            return Matrix.matrix_infty_subordinate_norm(matrix)
        if ord == 1:
            return Matrix.matrix_1_subordinate_norm(matrix)
        if ord == 2:
            return Matrix.matrix_2_subordinate_norm(matrix)
        raise ValueError("ord должно быть 1, 2 или numpy.infty.")

    @staticmethod
    def matrix_1_subordinate_norm(matrix: np.ndarray):
        maximum = 0
        for col in range(matrix.shape[1]):
            value = 0
            for row in range(matrix.shape[0]):
                value += abs(matrix[row, col])
            if value > maximum:
                maximum = value
        return maximum

    @staticmethod
    def matrix_infty_subordinate_norm(matrix: np.ndarray):
        maximum = 0
        for row in range(matrix.shape[0]):
            value = 0
            for col in range(matrix.shape[1]):
                value += abs(matrix[row, col])
            if value > maximum:
                maximum = value
        return maximum

    @staticmethod
    def matrix_2_subordinate_norm(matrix: np.ndarray):
        return np.linalg.norm(matrix, ord=2)
