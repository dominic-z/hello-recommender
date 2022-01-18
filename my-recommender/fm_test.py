from sklearn.feature_extraction import DictVectorizer
import numpy as np
from fm_core.fm import FMClassifier, FMRegression
from scipy.sparse import csr_matrix


def loadData(filename):
    data = []
    y = []
    users = set()
    items = set()
    with open(filename) as f:
        for line in f:
            (user, movieid, rating, ts) = line.split('\t')
            data.append({"user_id": str(user), "movie_id": str(movieid)})
            y.append(float(rating))
            users.add(user)
            items.add(movieid)

    return (data, np.array(y), users, items)


if __name__ == '__main__':
    train = [
        {"user": "1", "item": "5", "age": 19},
        {"user": "2", "item": "43", "age": 33},
        {"user": "3", "item": "20", "age": 55},
        {"user": "4", "item": "10", "age": 20},
    ]
    v = DictVectorizer()
    X = v.fit_transform(train)
    y = np.array([1, 1, -1, -1])
    fm = FMClassifier(3, [2,2], loss_function_type='hinge loss')
    fm.sgd_fit(X,y,learning_rate=1, gradient_clip=None,
               learning_rate_schedule='exponential decay',
               v_regs=None, batch_size=4, num_iter=5)
    print(fm.predict(X))
    print(fm.weight_0_)
    print(fm.weights_)
    print(fm.v_matrices_)


    # fm = FMRegression(2, [10])
    # (train_data, y_train, train_users, train_items) = loadData("data/ml-100k/ua.base")
    # (test_data, y_test, test_users, test_items) = loadData("data/ml-100k/ua.test")
    # v = DictVectorizer()
    # X_train = v.fit_transform(train_data)
    # X_test = v.transform(test_data)
    #
    # fm.sgd_fit(X_train[:10000, :], y_train[:10000], learning_rate=0.05, gradient_clip=None,
    #            learning_rate_schedule='exponential decay',
    #            v_regs=None, batch_size=1000, num_iter=5)
    # res = fm.predict(X_test)
    # with open('result.csv', 'w') as f:
    #     f.write(str(fm.weight_0_))
    #     f.write('\n')
    #     f.write(str(fm.weights_))
    #     f.write('\n')
    #     f.write(str(fm.v_matrices_))
    #     f.write('\n')
    #     for r in res:
    #         f.write(str(r))
    #         f.write('\n')
