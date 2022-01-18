from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb


class GDBTAndLR:
    def __init__(self):
        self.gdbt = GradientBoostingClassifier()
        self.lr = LogisticRegression()

    def set_param_for_gdbt(self, **gdbt_params):
        self.gdbt.set_params(**gdbt_params)

    def set_param_for_lr(self, **lr_params):
        self.lr.set_params(**lr_params)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.gdbt.fit(X, y)
        # 可以设置为auto，不会漏叶子，因为gbc的每个叶子节点一定是有训练样本的
        self.one_hot_encoder = OneHotEncoder(categories='auto')
        # GDBT的api中提到过
        # GB builds an additive model in a forward stage-wise fashion;
        # it allows for the optimization of arbitrary differentiable loss functions.
        # In each stage n_classes_ regression trees are fit on the negative gradient of the binomial or multinomial deviance loss function.
        # Binary classification is a special case where only a single regression tree is induced.
        # 就是说每个阶段都会训练n_classes_个分类树，因此apply的返回值的形状是[samples_num,estimator_num,class_]
        # 因为每个estimator_num有classes_个分类树
        pre_y = self.gdbt.apply(X).reshape([X.shape[0], -1])
        sparse_X_matrix = self.one_hot_encoder.fit_transform(pre_y)

        self.lr.fit(sparse_X_matrix, y)

    def predict(self, X: np.ndarray):
        sparse_X_matrix = self.one_hot_encoder.transform(self.gdbt.apply(X).reshape([X.shape[0], -1]))
        return self.lr.predict(sparse_X_matrix)

    def predict_proba(self, X: np.ndarray):
        sparse_X_matrix = self.one_hot_encoder.transform(self.gdbt.apply(X).reshape([X.shape[0], -1]))
        return self.lr.predict_proba(sparse_X_matrix)


class RFAndLR:
    def __init__(self):
        self.rf = RandomForestClassifier()
        self.lr = LogisticRegression()

    def set_param_for_rf(self, **rf_params):
        self.rf.set_params(**rf_params)

    def set_param_for_lr(self, **lr_params):
        self.lr.set_params(**lr_params)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.rf.fit(X, y)
        self.one_hot_encoder = OneHotEncoder(categories='auto')
        pre_y = self.rf.apply(X)
        sparse_X_matrix = self.one_hot_encoder.fit_transform(pre_y)

        self.lr.fit(sparse_X_matrix, y)

    def predict(self, X: np.ndarray):
        sparse_X_matrix = self.one_hot_encoder.transform(self.rf.apply(X))
        return self.lr.predict(sparse_X_matrix)

    def predict_proba(self, X: np.ndarray):
        sparse_X_matrix = self.one_hot_encoder.transform(self.rf.apply(X))
        return self.lr.predict_proba(sparse_X_matrix)


class XgboostAndLR:
    def __init__(self):
        self.xgb_dt = xgb.XGBClassifier()
        self.lr = LogisticRegression()

    def set_param_for_xgb_dt(self, **xgb_params):
        self.xgb_dt.set_params(**xgb_params)

    def set_param_for_lr(self, **lr_params):
        self.lr.set_params(**lr_params)

    def fit(self, X: np.ndarray, y: np.ndarray, **xgb_dt_fit_param):
        self.xgb_dt.fit(X, y, **xgb_dt_fit_param)
        self.one_hot_encoder = OneHotEncoder(categories='auto')
        sparse_X_matrix = self.one_hot_encoder.fit_transform(self.xgb_dt.apply(X))

        self.lr.fit(sparse_X_matrix, y)

    def predict(self, X: np.ndarray):
        sparse_X_matrix = self.one_hot_encoder.transform(self.xgb_dt.apply(X))
        return self.lr.predict(sparse_X_matrix)

    def predict_proba(self, X: np.ndarray):
        sparse_X_matrix = self.one_hot_encoder.transform(self.xgb_dt.apply(X))
        return self.lr.predict_proba(sparse_X_matrix)
