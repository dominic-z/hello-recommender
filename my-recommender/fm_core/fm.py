"""
Sorry that my English is poor.
use scipy.sparse.csr_matrix to store data
算了直接写中文吧，反正也没人看
"""

import numpy as np
from scipy.sparse import csr_matrix
from typing import List
from itertools import combinations


# noinspection PyPep8Naming,PyAttributeOutsideInit
class FMClassifier:
    """
    Factorization Machine only support Binary classification method when it's used to be a classification method
    """
    _legal_loss_function_types = {'logit loss', 'hinge loss', 'exponential loss'}
    _legal_learning_rate_schedule = {'constant', 'exponential decay', 'natural exp decay'}

    def __init__(self,
                 d: int,
                 factor_nums: List[int],
                 init_stdev=0.1,
                 loss_function_type='logit loss'):
        """

        :param d: d-way 分解机
        :param factor_nums: 一个列表，第i个元素为第i阶v向量的元素个数
        :param init_stdev: 权重初始化标准差
        :param loss_function_type: 损失函数，支持三种，log损失，hinge损失，指数损失，合法的损失名见_legal_loss_function_types
        """
        if d < 2:
            raise ValueError('d should be lager than 2')
        self.d = d
        if d != len(factor_nums) + 1:
            raise ValueError('d should be equal with len(factor_nums)+1')
        self.factor_nums = factor_nums
        self.init_stdev = init_stdev

        if loss_function_type not in self._legal_loss_function_types:
            raise ValueError('%s is an illegal loss_function' % loss_function_type)
        self.loss_function_type = loss_function_type

        self._fit = False

    def predict(self, X: csr_matrix):
        if not self._fit:
            raise ValueError('not fit yet')
        pre_y = self.decision_function(X)
        pre_y[pre_y > 0] = 1
        pre_y[pre_y <= 0] = -1
        return pre_y

    def decision_function(self, X: csr_matrix):
        if not self._fit:
            raise ValueError('not fit yet')
        distances = list()

        for sample_index in range(X.shape[0]):
            distance = self.weight_0_
            c_start, c_end = X.indptr[sample_index], X.indptr[sample_index + 1]
            for feature_index in X.indices[c_start:c_end]:
                distance += self.weights_[feature_index] * X[sample_index, feature_index]
            for d in range(2, self.d + 1):
                v_matrix = self.v_matrices_[d - 2]
                if d > len(X.indices[c_start:c_end]):
                    break
                for feature_indices in combinations(X.indices[c_start:c_end], d):
                    feature_prod, vector_prod = 1, np.ones(self.factor_nums[d - 2])
                    for feature_index in feature_indices:
                        feature_prod *= X[sample_index, feature_index]
                        vector_prod *= v_matrix[feature_index, :]
                    distance += feature_prod * np.sum(vector_prod)
            # clip
            # 这一步是为了防止有些向量算出的distance太大导致exp计算溢出
            # distance = -10 if distance < -10 else distance
            # distance = 10 if distance > 10 else distance
            distances.append(distance)
        distances = np.array(distances)
        return distances

    def sgd_fit(self, X: csr_matrix, y: np.ndarray,
                learning_rate=0.05,
                learning_rate_schedule='constant',
                decay_rate=0.9,
                decay_step=10,
                gradient_clip=None,
                batch_size=None, num_iter=10,
                w0_reg=0, w1_regs: np.ndarray = None, v_regs: np.ndarray = None, ):
        """

        :param X: 特征稀疏矩阵
        :param y: 标签
        :param learning_rate: 学习率，建议设置的小一点，感觉这个算法很容易有悬崖啊
        :param learning_rate_schedule: 学习率更新方式，包括恒定学习率和两种衰减学习率
        :param decay_rate: 衰减系数
        :param decay_step: 衰减步幅
        :param gradient_clip: 对梯度进行剪切，防止梯度过大导致震荡，请善用这个参数，感觉一不小心梯度就上去了
        :param batch_size:
        :param num_iter:
        :param w0_reg: w0的正则化系数
        :param w1_regs: w1的正则化系数，是一个(X.shape[1],)的一维向量
        :param v_regs: 二维向量，其中第i行第j列代表着j阶第i个向量的正则化系数，因此形状应该是[X.shape[1],d-1]
        :return:
        """
        self.weight_0_: float = np.random.normal(loc=0, scale=self.init_stdev)
        # self.weight_0_: float = 0
        self.weights_: np.ndarray = np.random.normal(loc=0, scale=self.init_stdev, size=X.shape[1])
        # self.weights_: np.ndarray = np.ones((X.shape[1]))
        self.v_matrices_: List[np.ndarray] = list()
        for d in range(self.d - 1):
            self.v_matrices_.append(
                np.random.normal(loc=0, scale=self.init_stdev, size=[X.shape[1], self.factor_nums[d]]))
            # self.v_matrices_.append(np.ones((X.shape[1], self.factor_nums[d])))

        if learning_rate_schedule not in self._legal_learning_rate_schedule:
            raise ValueError('%s is an illegal learning_rate_schedule' % learning_rate_schedule)

        if w1_regs is None:
            w1_regs = np.zeros(X.shape[1])
        elif w1_regs.shape[0] != X.shape[1]:
            raise ValueError('the shape of w1_reg should be equal with %d' % X.shape[1])

        if v_regs is None:
            v_regs = np.zeros((X.shape[1], self.d - 1))
        elif v_regs.shape != (X.shape[1], self.d - 1):
            raise ValueError('the shape of v_regs should be equal with (%d,%d)' % (X.shape[1], self.d - 1))

        if batch_size is None or batch_size > X.shape[0]:
            batch_size = X.shape[0]

        next_index = 0
        self._fit = True
        for i in range(num_iter * X.shape[0] // batch_size):
            indices, next_index = self.__next_batch(X, cur_index=next_index, batch_size=batch_size)
            X_batch, y_batch = X[indices], y[indices]
            loss = self.__get_loss(X_batch, y_batch, w0_reg, w1_regs, v_regs)
            print('optimize for %d time, the loss is %.5f' % (i, loss))
            self._sgd_optimizer(X_batch, y_batch, gradient_clip, learning_rate, w0_reg, w1_regs, v_regs)
            # self._sgd_optimizer_slow(X_batch, y_batch, gradient_clip, learning_rate, w0_reg, w1_regs, v_regs)
            learning_rate = self._update_learning_rate(learning_rate, learning_rate_schedule, decay_rate, decay_step)

    def _update_learning_rate(self, learning_rate,
                              learning_rate_schedule,
                              decay_rate,
                              decay_step):
        if learning_rate_schedule == 'exponential decay':
            learning_rate = learning_rate * decay_rate ** (1 / decay_step)
        elif learning_rate_schedule == 'natural exp decay':
            learning_rate = learning_rate * np.exp(-decay_step)
        return learning_rate

    def _sgd_optimizer(self, X: csr_matrix, y: np.ndarray, gradient_clip, learning_rate,
                       w0_reg, w1_regs: np.ndarray, v_regs: np.ndarray):
        distances = self.decision_function(X)
        if self.loss_function_type == 'logit loss':
            gradients_of_loss_and_distance = (1 / (1 + np.exp(-distances * y))) * \
                                             np.exp(-distances * y) * \
                                             (-y)
        elif self.loss_function_type == 'hinge loss':
            gradients_of_loss_and_distance = np.zeros(X.shape[0])
            for i in range(X.shape[0]):
                if y[i] == 1 and distances[i] < 0:
                    gradients_of_loss_and_distance[i] = -1
                if y[i] == -1 and distances[i] > 0:
                    gradients_of_loss_and_distance[i] = 1
        else:
            gradients_of_loss_and_distance = np.exp(-distances * y) * \
                                             (-y)
        weight_0_gradient = np.mean(gradients_of_loss_and_distance) + 2 * w0_reg * self.weight_0_
        if gradient_clip is not None:
            weight_0_gradient = gradient_clip if weight_0_gradient > gradient_clip else weight_0_gradient
            weight_0_gradient = -gradient_clip if weight_0_gradient < -gradient_clip else weight_0_gradient
        self.weight_0_ = self.weight_0_ - learning_rate * weight_0_gradient

        weights_gradients = np.zeros(self.weights_.shape[0])
        for sample_index in range(X.shape[0]):
            weights_gradients = weights_gradients + \
                                X[sample_index, :].toarray()[0] * gradients_of_loss_and_distance[sample_index]
        weights_gradients = weights_gradients / X.shape[0] + 2 * w1_regs * self.weights_
        if gradient_clip is not None:
            weights_gradients[weights_gradients > gradient_clip] = gradient_clip
            weights_gradients[weights_gradients < -gradient_clip] = -gradient_clip
        self.weights_ = self.weights_ - learning_rate * weights_gradients

        for d in range(2, self.d + 1):
            factor_num = self.factor_nums[d - 2]
            v_matrix_gradients = np.zeros((X.shape[1], factor_num))
            v_matrix = self.v_matrices_[d - 2]
            v_reg = v_regs[:, d - 2:d - 1]
            for l in range(X.shape[1]):
                v_l_gradients = np.zeros(factor_num)
                for sample_index in range(X.shape[0]):
                    if X[sample_index, l] == 0:
                        continue
                    sum_of_prods = np.zeros(factor_num)
                    for feature_indices in combinations(sorted(set(X[sample_index, :].indices) - {l}), d - 1):
                        for m in range(factor_num):
                            prod = 1
                            for feature_index in feature_indices:
                                prod *= X[sample_index, feature_index] * v_matrix[feature_index, m]
                            sum_of_prods[m] += prod
                    sum_of_prods = sum_of_prods * X[sample_index, l]
                    v_l_gradients = v_l_gradients + gradients_of_loss_and_distance[sample_index] * sum_of_prods
                v_matrix_gradients[l] = v_l_gradients / X.shape[0]
            v_matrix_gradients = v_matrix_gradients + 2 * v_reg * v_matrix
            if gradient_clip is not None:
                v_matrix_gradients[v_matrix_gradients > gradient_clip] = gradient_clip
                v_matrix_gradients[v_matrix_gradients < -gradient_clip] = -gradient_clip
            self.v_matrices_[d - 2] = v_matrix - learning_rate * v_matrix_gradients

    def _sgd_optimizer_slow(self, X: csr_matrix, y: np.ndarray, gradient_clip, learning_rate,
                            w0_reg, w1_regs: np.ndarray, v_regs: np.ndarray):
        distances = self.decision_function(X)
        if self.loss_function_type == 'logit loss':
            gradients_of_loss_and_distance = (1 / (1 + np.exp(-distances * y))) * \
                                             np.exp(-distances * y) * \
                                             (-y)
        elif self.loss_function_type == 'hinge loss':
            gradients_of_loss_and_distance = np.zeros(X.shape[0])
            for i in range(X.shape[0]):
                if y[i] == 1 and distances[i] < 1:
                    gradients_of_loss_and_distance[i] = -1
                if y[i] == -1 and distances[i] > -1:
                    gradients_of_loss_and_distance[i] = 1
        else:
            gradients_of_loss_and_distance = np.exp(-distances * y) * \
                                             (-y)

        weight_0_gradient = np.mean(gradients_of_loss_and_distance) + 2 * w0_reg * self.weight_0_
        if gradient_clip is not None:
            weight_0_gradient = gradient_clip if weight_0_gradient > gradient_clip else weight_0_gradient
            weight_0_gradient = -gradient_clip if weight_0_gradient < -gradient_clip else weight_0_gradient
        self.weight_0_ = self.weight_0_ - learning_rate * weight_0_gradient

        for l in range(X.shape[1]):
            gradient = 0
            for s_index in range(X.shape[0]):
                gradient += gradients_of_loss_and_distance[s_index] * X[s_index, l]
            gradient = gradient / X.shape[0] + 2 * w1_regs[l] * self.weights_[l]
            if gradient_clip is not None:
                if gradient > gradient_clip:
                    gradient = gradient_clip
                elif gradient < -gradient_clip:
                    gradient = -gradient_clip
            self.weights_[l] = self.weights_[l] - learning_rate * gradient

        for d in range(2, self.d + 1):
            factor_num = self.factor_nums[d - 2]
            v_matrix = self.v_matrices_[d - 2].copy()
            v_reg = v_regs[:, d - 2]

            for l in range(X.shape[1]):
                for m in range(factor_num):
                    gradient = 0
                    for s_index in range(X.shape[0]):
                        sum = 0
                        for f_indices in combinations(set(range(X.shape[1])) - {l}, d - 1):
                            prod = 1
                            for f_index in f_indices:
                                prod *= X[s_index, f_index] * v_matrix[f_index, m]
                            sum += prod
                        gradient += gradients_of_loss_and_distance[s_index] * X[s_index, l] * sum
                    gradient = gradient / X.shape[0] + 2 * v_reg[l] * v_matrix[l, m]
                    if gradient_clip is not None:
                        if gradient > gradient_clip:
                            gradient = gradient_clip
                        elif gradient < -gradient_clip:
                            gradient = -gradient_clip
                    self.v_matrices_[d - 2][l, m] = v_matrix[l, m] - learning_rate * gradient

    def _als_optimizer(self, X: csr_matrix, y: np.ndarray,
                       w0_reg, w1_regs: np.ndarray, v_regs: np.ndarray):
        pass

    def __next_batch(self, data, cur_index, batch_size):
        if cur_index + batch_size <= data.shape[0]:
            next_index = cur_index + batch_size
            indices = list(range(cur_index, next_index))
        else:
            next_index = cur_index + batch_size - data.shape[0]
            indices = list(range(cur_index, data.shape[0])) + list(range(next_index))
        return indices, next_index

    def __get_loss(self, X: csr_matrix, y: np.ndarray, w0_reg, w1_regs: np.ndarray, v_regs: np.ndarray):
        pre_y = self.predict(X)
        if self.loss_function_type == 'logit loss':
            loss = np.mean(-np.log(1 / (1 + np.exp(-pre_y * y))))

        elif self.loss_function_type == 'hinge loss':
            loss = 0
            for i in range(len(y)):
                if y[i] == 1:
                    loss += np.max([0, 1 - pre_y[i]])
                else:
                    loss += np.max([0, 1 + pre_y[i]])
            loss = loss / len(y)
        else:
            loss = np.exp(-pre_y * y)
        loss += w0_reg * self.weight_0_ ** 2 + np.sum(w1_regs * (self.weights_ ** 2))
        for d in range(len(self.v_matrices_)):
            v_matrix = self.v_matrices_[d]
            v_reg = v_regs[:, d:d + 1]
            loss += np.sum(v_matrix * v_matrix * v_reg)

        return loss


# noinspection PyPep8Naming,PyAttributeOutsideInit
class FMRegression:
    """
    和分类函数差不多，只不过回归的时候只用了均方误差
    Factorization Machine only support Binary classification method when it's used to be a classification method
    """
    _legal_learning_rate_schedule = {'constant', 'exponential decay', 'natural exp decay'}

    def __init__(self,
                 d: int,
                 factor_nums: List[int],
                 init_stdev=0.1):
        if d < 2:
            raise ValueError('d should be lager than 2')
        self.d = d
        if d != len(factor_nums) + 1:
            raise ValueError('d should be equal with len(factor_nums)+1')
        self.factor_nums = factor_nums
        self.init_stdev = init_stdev

        self._fit = False

    def predict(self, X: csr_matrix):
        if not self._fit:
            raise ValueError('not fit yet')
        pre_y = self.decision_function(X)
        return pre_y

    def decision_function(self, X: csr_matrix):
        if not self._fit:
            raise ValueError('not fit yet')
        distances = list()

        for sample_index in range(X.shape[0]):
            distance = self.weight_0_
            c_start, c_end = X.indptr[sample_index], X.indptr[sample_index + 1]
            for feature_index in X.indices[c_start:c_end]:
                distance += self.weights_[feature_index] * X[sample_index, feature_index]
            for d in range(2, self.d + 1):
                v_matrix = self.v_matrices_[d - 2]
                if d > len(X.indices[c_start:c_end]):
                    break
                for feature_indices in combinations(X.indices[c_start:c_end], d):
                    feature_prod, vector_prod = 1, np.ones(self.factor_nums[d - 2])
                    for feature_index in feature_indices:
                        feature_prod *= X[sample_index, feature_index]
                        vector_prod *= v_matrix[feature_index, :]
                    distance += feature_prod * np.sum(vector_prod)
            # clip
            distance = self.min_target_ if distance < self.min_target_ else distance
            distance = self.max_target_ if distance > self.max_target_ else distance
            distances.append(distance)
        distances = np.array(distances)
        return distances

    def sgd_fit(self, X: csr_matrix, y: np.ndarray,
                learning_rate=0.05,
                learning_rate_schedule='constant',
                decay_rate=0.9,
                decay_step=10,
                gradient_clip=None,
                batch_size=None, num_iter=10,
                w0_reg=0, w1_regs: np.ndarray = None, v_regs: np.ndarray = None):
        self.weight_0_: float = np.random.normal(loc=0, scale=self.init_stdev)
        # self.weight_0_: float = 0
        self.weights_: np.ndarray = np.random.normal(loc=0, scale=self.init_stdev, size=X.shape[1])
        # self.weights_: np.ndarray = np.zeros((X.shape[1]))
        self.v_matrices_: List[np.ndarray] = list()

        self.max_target_ = np.max(y)
        self.min_target_ = np.min(y)

        for d in range(self.d - 1):
            self.v_matrices_.append(
                np.random.normal(loc=0, scale=self.init_stdev, size=[X.shape[1], self.factor_nums[d]]))
            # self.v_matrices_.append(np.ones((X.shape[1], self.factor_nums[d])))

        if learning_rate_schedule not in self._legal_learning_rate_schedule:
            raise ValueError('%s is an illegal learning_rate_schedule' % learning_rate_schedule)

        if w1_regs is None:
            w1_regs = np.zeros(X.shape[1])
        elif w1_regs.shape[0] != X.shape[1]:
            raise ValueError('the shape of w1_reg should be equal with %d' % X.shape[1])

        if v_regs is None:
            v_regs = np.zeros((X.shape[1], self.d - 1))
        elif v_regs.shape != (X.shape[1], self.d - 1):
            raise ValueError('the shape of v_regs should be equal with (%d,%d)' % (X.shape[1], self.d - 1))

        if batch_size is None or batch_size > X.shape[0]:
            batch_size = X.shape[0]

        next_index = 0
        self._fit = True
        for i in range(num_iter * X.shape[0] // batch_size):
            indices, next_index = self.__next_batch(X, cur_index=next_index, batch_size=batch_size)
            X_batch, y_batch = X[indices], y[indices]
            loss = self.__get_loss(X_batch, y_batch, w0_reg, w1_regs, v_regs)
            print('optimize for %d time, the loss is %.5f' % (i, loss))
            self._sgd_optimizer(X_batch, y_batch, gradient_clip, learning_rate, w0_reg, w1_regs, v_regs)
            # self._sgd_optimizer_slow(X_batch, y_batch, gradient_clip, learning_rate, w0_reg, w1_regs, v_regs)
            learning_rate = self._update_learning_rate(learning_rate, learning_rate_schedule, decay_rate, decay_step)

    def _update_learning_rate(self, learning_rate,
                              learning_rate_schedule,
                              decay_rate,
                              decay_step):
        if learning_rate_schedule == 'exponential decay':
            learning_rate = learning_rate * decay_rate ** (1 / decay_step)
        elif learning_rate_schedule == 'natural exp decay':
            learning_rate = learning_rate * np.exp(-decay_step)
        return learning_rate

    def _sgd_optimizer(self, X: csr_matrix, y: np.ndarray, gradient_clip, learning_rate,
                       w0_reg, w1_regs: np.ndarray, v_regs: np.ndarray):
        distances = self.decision_function(X)
        gradients_of_loss_and_distance = 2 * (distances - y)

        weight_0_gradient = np.mean(gradients_of_loss_and_distance) + 2 * w0_reg * self.weight_0_
        if gradient_clip is not None:
            weight_0_gradient = gradient_clip if weight_0_gradient > gradient_clip else weight_0_gradient
            weight_0_gradient = -gradient_clip if weight_0_gradient < -gradient_clip else weight_0_gradient
        self.weight_0_ = self.weight_0_ - learning_rate * weight_0_gradient

        weights_gradients = np.zeros(self.weights_.shape[0])
        for sample_index in range(X.shape[0]):
            weights_gradients = weights_gradients + \
                                X[sample_index, :].toarray()[0] * gradients_of_loss_and_distance[sample_index]
        weights_gradients = weights_gradients / X.shape[0] + 2 * w1_regs * self.weights_
        if gradient_clip is not None:
            weights_gradients[weights_gradients > gradient_clip] = gradient_clip
            weights_gradients[weights_gradients < -gradient_clip] = -gradient_clip
        self.weights_ = self.weights_ - learning_rate * weights_gradients

        for d in range(2, self.d + 1):
            factor_num = self.factor_nums[d - 2]
            v_matrix_gradients = np.zeros((X.shape[1], factor_num))
            v_matrix = self.v_matrices_[d - 2]
            v_reg = v_regs[:, d - 2:d - 1]
            for l in range(X.shape[1]):
                v_l_gradients = np.zeros(factor_num)
                for sample_index in range(X.shape[0]):
                    if X[sample_index, l] == 0:
                        continue
                    sum_of_prods = np.zeros(factor_num)
                    for feature_indices in combinations(sorted(set(X[sample_index, :].indices) - {l}), d - 1):
                        for m in range(factor_num):
                            prod = 1
                            for feature_index in feature_indices:
                                prod *= X[sample_index, feature_index] * v_matrix[feature_index, m]
                            sum_of_prods[m] += prod
                    sum_of_prods = sum_of_prods * X[sample_index, l]
                    v_l_gradients = v_l_gradients + gradients_of_loss_and_distance[sample_index] * sum_of_prods
                v_matrix_gradients[l] = v_l_gradients / X.shape[0]
            v_matrix_gradients = v_matrix_gradients + 2 * v_reg * v_matrix
            if gradient_clip is not None:
                v_matrix_gradients[v_matrix_gradients > gradient_clip] = gradient_clip
                v_matrix_gradients[v_matrix_gradients < -gradient_clip] = -gradient_clip
            self.v_matrices_[d - 2] = v_matrix - learning_rate * v_matrix_gradients

    def _sgd_optimizer_slow(self, X: csr_matrix, y: np.ndarray, gradient_clip, learning_rate,
                            w0_reg, w1_regs: np.ndarray, v_regs: np.ndarray):
        distances = self.decision_function(X)
        gradients_of_loss_and_distance = 2 * (distances - y)

        weight_0_gradient = np.mean(gradients_of_loss_and_distance) + 2 * w0_reg * self.weight_0_
        if gradient_clip is not None:
            weight_0_gradient = gradient_clip if weight_0_gradient > gradient_clip else weight_0_gradient
            weight_0_gradient = -gradient_clip if weight_0_gradient < -gradient_clip else weight_0_gradient
        self.weight_0_ = self.weight_0_ - learning_rate * weight_0_gradient

        for l in range(X.shape[1]):
            gradient = 0
            for s_index in range(X.shape[0]):
                gradient += gradients_of_loss_and_distance[s_index] * X[s_index, l]
            gradient = gradient / X.shape[0] + 2 * w1_regs[l] * self.weights_[l]
            if gradient_clip is not None:
                if gradient > gradient_clip:
                    gradient = gradient_clip
                elif gradient < -gradient_clip:
                    gradient = -gradient_clip
            self.weights_[l] = self.weights_[l] - learning_rate * gradient

        for d in range(2, self.d + 1):
            factor_num = self.factor_nums[d - 2]
            v_matrix = self.v_matrices_[d - 2].copy()
            v_reg = v_regs[:, d - 2]

            for l in range(X.shape[1]):
                for m in range(factor_num):
                    gradient = 0
                    for s_index in range(X.shape[0]):
                        sum_ = 0
                        for f_indices in combinations(set(range(X.shape[1])) - {l}, d - 1):
                            prod = 1
                            for f_index in f_indices:
                                prod *= X[s_index, f_index] * v_matrix[f_index, m]
                            sum_ += prod
                        gradient += gradients_of_loss_and_distance[s_index] * X[s_index, l] * sum_
                    gradient = gradient / X.shape[0] + 2 * v_reg[l] * v_matrix[l, m]
                    if gradient_clip is not None:
                        if gradient > gradient_clip:
                            gradient = gradient_clip
                        elif gradient < -gradient_clip:
                            gradient = -gradient_clip
                    self.v_matrices_[d - 2][l, m] = v_matrix[l, m] - learning_rate * gradient

    def _als_optimizer(self, X: csr_matrix, y: np.ndarray,
                       w0_reg, w1_regs: np.ndarray, v_regs: np.ndarray):
        pass

    def __next_batch(self, data, cur_index, batch_size):
        if cur_index + batch_size <= data.shape[0]:
            next_index = cur_index + batch_size
            indices = list(range(cur_index, next_index))
        else:
            next_index = cur_index + batch_size - data.shape[0]
            indices = list(range(cur_index, data.shape[0])) + list(range(next_index))
        return indices, next_index

    def __get_loss(self, X: csr_matrix, y: np.ndarray, w0_reg, w1_regs: np.ndarray, v_regs: np.ndarray):
        pre_y = self.predict(X)
        loss = np.mean(np.power(pre_y - y, 2))
        loss += w0_reg * self.weight_0_ ** 2 + np.sum(w1_regs * (self.weights_ ** 2))
        for d in range(len(self.v_matrices_)):
            v_matrix = self.v_matrices_[d]
            v_reg = v_regs[:, d:d + 1]
            loss += np.sum(v_matrix * v_matrix * v_reg)

        return loss
