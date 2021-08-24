import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, plot_confusion_matrix
import shap 
import matplotlib.pyplot as plt 

df = pd.read_csv('water_potability.csv')

## Remove Missing Value
def remove_missing(self):
    X = df.iloc[:, :-1]
    for col in X.columns:
        Avg = X[col].mean()
        X.fillna(Avg, inplace = True)
    return X

X = list(map(remove_missing, df))[0]
## Train_test_split
X.drop(['Solids'], axis = 1, inplace = True)
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .1, random_state = 5)
##XGB
XGB = XGBClassifier(num_class = 2)

params = {'objective': ['multi:softmax'],
          'n_estimators': [i for i in range(5, 50, 2)],
          'max_depth': [i for i in range(10, 100, 5)],
          'eta':[10e-5, 10e-4, 10e-3, 10e-2, .1, .3, .5, 1],
          'min_child_weight': [.1, 1, 3, 5, 7, 9],
          'gamma': [10e-2, 10e-1, .3, .5, .7, 1, 2, 5]
          }
def GSCV(self):
    gscv = GridSearchCV(XGB, param_grid = params, scoring = 'accuracy',
                            cv = 5, n_jobs = -1)
    gscv.fit(X_train, y_train)
    return None

rscv = RandomizedSearchCV(XGB, param_distributions = params, scoring = 'accuracy',
                         n_iter = 10, cv = 5, n_jobs = -1, random_state = 5)
rscv.fit(X_train, y_train)

print(f'Best Estimator is: {rscv.best_estimator_} \n Best Score is: {rscv.best_score_}')

##RSCV XGB Fit
rscv.best_estimator_.fit(X_train, y_train, early_stopping_rounds = 100, eval_set = [(X_test, y_test)],
                        eval_metric = 'merror') #eval_metrics = 'error'

##Shapley Value
Explainer = shap.TreeExplainer(rscv.best_estimator_, X_test)
shap_value = Explainer.shap_values(X_test)
shap.initjs()
shap.summary_plot(shap_values = shap_value[0], features = X_test, feature_names = X.columns)