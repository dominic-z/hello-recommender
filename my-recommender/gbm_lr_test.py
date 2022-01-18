from sklearn.datasets import load_iris
from sklearn.metrics import classification_report

from gdm_lr_core.gbm_lr import XgboostAndLR,RFAndLR,GDBTAndLR

if __name__ == '__main__':
    iris_data = load_iris()
    X, y = iris_data['data'], iris_data['target']
    # X=X[y!=0]
    # y=y[y!=0]

    xgb_lr = XgboostAndLR()
    xgb_lr.set_param_for_xgb_dt(n_estimators=20)
    xgb_lr.set_param_for_lr(multi_class='auto',solver='liblinear')
    xgb_lr.fit(X,y)
    pre_y = xgb_lr.predict(X)
    print(pre_y)
    print(classification_report(y,pre_y))

    rf_lr = RFAndLR()
    rf_lr.set_param_for_rf(n_estimators=20)
    rf_lr.set_param_for_lr(multi_class='auto',solver='liblinear')
    rf_lr.fit(X,y)
    pre_y = rf_lr.predict(X)
    print(pre_y)
    print(classification_report(y, pre_y))

    gdbt_lr = GDBTAndLR()
    gdbt_lr.set_param_for_gdbt(n_estimators=20)
    gdbt_lr.set_param_for_lr(multi_class='auto', solver='liblinear')
    gdbt_lr.fit(X, y)
    pre_y = gdbt_lr.predict(X)
    print(pre_y)
    print(classification_report(y, pre_y))

