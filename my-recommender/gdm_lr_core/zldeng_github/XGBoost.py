#!/usr/bin/env python
# encoding=utf8
'''
  Author: zldeng
  create@2017-09-21 11:20:15
'''

import sys, os

# reload(sys)
# sys.setdefaultencoding('utf8')

import xgboost as xgb
import numpy as np
import pickle

from xgboost import XGBClassifier, DMatrix


class XGBoost(object):
    '''
    xgboost model for text classification
    '''

    def __init__(self, xgb_model_name, n_jobs=1,
                 eval_metric='mlogloss'):
        self.n_jobs = n_jobs
        self.eval_metric = eval_metric
        self.xgb_model_name = xgb_model_name
        self.init_flag = False

    def trainModel(self, train_x, train_y):
        '''
        train_x: darray [samples feature_cnt] 使用FeatureModel处理后的特征向量
        '''
        self.clf = xgb.XGBClassifier(n_jobs=self.n_jobs)

        self.clf.fit(train_x, train_y, eval_metric=self.eval_metric,
                     eval_set=[(train_x, train_y)])

        self.init_flag = True

        evals_result = self.clf.evals_result()

        print('evals_result: ', evals_result)

        with open(self.xgb_model_name, 'wb') as f:
            pickle.dump(self.clf, f, True)

    def loadModel(self, xgb_model_name):
        with open(xgb_model_name, 'rb') as f:
            self.clf = pickle.load(f)
            self.init_flag = True

    def testModel(self, test_x, test_y):
        '''
        test_x: darray [samples feature_cnt]
        '''
        if not self.init_flag:
            print('Not init xgb_clf. load model now...')
            self.loadModel(self.xgb_model_name)
            print('Load xgbModel done!')

        # 如果不需要使用预测的概率信息，可直接调用predict方法
        # 折腾这一步干啥
        predict_res = self.clf.predict_proba(test_x)

        pred_idx_arr = np.argmax(predict_res, axis=1)
        pred_label = [self.clf.classes_[idx] for idx in pred_idx_arr]

        total = len(test_y)
        correct = 0
        for idx in range(total):
            if pred_label[idx] == test_y[idx]:
                correct += 1
        print('Xgb test: ', total, correct, correct * 1.0 / total)
