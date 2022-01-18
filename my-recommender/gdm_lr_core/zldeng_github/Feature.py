#!/usr/bin/env python
# encoding=utf8
'''
  Author: zldeng
  create@2017-09-21 11:18:03
'''
import sys, os
# reload(sys)
# sys.setdefaultencoding('utf8')

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np
import pickle


class FeatureModel(object):
    '''
    this class is used to select feature from the raw text
    '''

    def __init__(self, feature_vec_model_name, best_feature_model_name):
        self.feature_vec_model_name = feature_vec_model_name
        self.best_feature_model_name = best_feature_model_name

        self.init_flag = False

    def fitModelByData(self, x_train, y_train):
        best_k = self.max_feature_cnt
        # 创建TfidfVectorizer词典的时候，只对语料库里出现次数在min max之间的词构建
        vec_max_df = self.feature_max_df
        vec_min_df = self.feature_min_df
        # ngram_range(min,max)是指将text分成min，min+1，min+2,.........max 个不同的词组比如'Python is useful'中ngram_range(1,3)之后可得到'Python'  'is'  'useful'  'Python is'  'is useful' 和'Python is useful'如果是ngram_range (1,1) 则只能得到单个单词'Python'  'is'和'useful'
        #
        # 作者：Ten_Minutes
        # 链接：https://www.jianshu.com/p/c39feaf0d62f
        # 来源：简书
        # 简书著作权归作者所有，任何形式的转载都请联系作者获得授权并注明出处。
        vec_ngram_range = self.ngram_range
        self.tf_vec_ = TfidfVectorizer(ngram_range=vec_ngram_range,
                                       min_df=vec_min_df, max_df=vec_max_df)
        self.best_ = SelectKBest(chi2, k=best_k)
        train_tf_vec = self.tf_vec_.fit_transform(x_train)
        train_best = self.best_.fit_transform(train_tf_vec, y_train)

    def setFeatureModelPara(self, max_feature_cnt, feature_max_df,
                            feature_min_df, ngram_range):
        self.max_feature_cnt = max_feature_cnt
        self.feature_max_df = feature_max_df
        self.feature_min_df = feature_min_df
        self.ngram_range = ngram_range

    def fit(self, max_feature_cnt, feature_max_df,
            feature_min_df, ngram_range, x_train, y_train):
        self.setFeatureModelPara(max_feature_cnt, feature_max_df,
                                 feature_min_df, ngram_range)

        self.fitModelByData(x_train, y_train)

        with open(self.feature_vec_model_name, 'wb') as f:
            pickle.dump(self.tf_vec_, f, True)
        with open(self.best_feature_model_name, 'wb') as f:
            pickle.dump(self.best_, f, True)

    def loadModel(self):
        with open(self.feature_vec_model_name, 'rb') as f:
            self.tf_vec_ = pickle.load(f)
        with open(self.best_feature_model_name, 'rb') as f:
            self.best_ = pickle.load(f)
        self.init_flag = True

    def transform(self, x_test):
        if not self.init_flag:
            self.loadModel()

        x_vec = self.tf_vec_.transform(x_test)
        x_best = self.best_.transform(x_vec)

        return x_best
