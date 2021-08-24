import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 

data = pd.read_csv('water_potability.csv')
 
## Remove Nan value
def remove_missing(self):
    X = data.iloc[:, :-1]
    for col in X.columns:
        Avg = X[col].mean()
        X[col].fillna(Avg, inplace = True)
    return X 

X = list(map(remove_missing, data))[0]

## Correlation Analysis
'''sns.heatmap(data.corr(), cmap="YlGnBu", vmin = 0, vmax = 1, annot = True, linewidths = .5)
plt.show()
plt.savefig('Water_quality_Correlation.png')'''

## Outlier Detection
'''data.boxplot()
plt.show()
plt.savefig('Water_Quality_Outlier_Boxplot.jpg') #Solids Feature shall be removed'''

X.drop(['Solids'], axis = 1, inplace = True)
y = data.iloc[:, -1].values.reshape(-1,)

## Data Modelling
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import MinMaxScaler

##Test_train_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .1, random_state = 5)
LR = LogisticRegressionCV(penalty = 'l2', cv = 5, scoring = 'accuracy',  n_jobs = -1)
RFC = RandomForestClassifier(n_estimators = 50, criterion = 'gini', max_depth = 20, 
    min_samples_split = 50, random_state = 5)

svc = SVC(C = .5, kernel = 'rbf', max_iter = 50, random_state = 5)

## RandomizedSearchCV
params = {'min_samples_split': [i for i in range(300, 100, -10)], 
          'max_depth': [i for i in range(3, 30, 3)],
          'criterion': ['gini', 'entropy'],
          'max_leaf_nodes': [i for i in range(30, 60, 3)]
          }

rscv = RandomizedSearchCV(estimator = RFC, param_distributions = params, n_iter = 20, 
                        scoring = 'accuracy', n_jobs = -1, cv = 5, random_state = 5, refit = True)
rscv.fit(X_train, y_train)

RFC_Model = rscv.best_estimator_
print(RFC_Model)

minmax = MinMaxScaler(feature_range = (0, 1))
X_train_minmax = minmax.fit_transform(X_train)
X_test_minmax = minmax.transform(X_test)

vc = VotingClassifier(estimators = [('lr',LR), ('svc', svc), ('rf', RFC)], voting = 'hard',
        weights = [10, 1, 20])

svc.fit(X_train_minmax, y_train)
vc.fit(X_train_minmax, y_train)
LR.fit(X_train, y_train)
RFC.fit(X_train, y_train)

#print(LR.score(X_test, y_test), LR.score(X_train, y_train))
print(svc.score(X_train_minmax, y_train), svc.score(X_test_minmax, y_test))

## Shapley Value Analysis (Find cause-and-effect relation)
import shap 

'''vc_test = X_test_minmax[0:100]
vc_explainer = shap.KernelExplainer(vc.predict, vc_test)
vc_shap = vc_explainer.shap_values(X_test_minmax)
shap.initjs()
shap.summary_plot(vc_shap[0], X_test_minmax, feature_names = X.columns)'''
#plt.savefig('VC Shapley Values.png')

RFCExplainer = shap.TreeExplainer(RFC_Model, X_test)
RFCshap_values = RFCExplainer.shap_values(X_test)
shap.initjs()
shap.summary_plot(shap_values = RFCshap_values[0], features = X_test, plot_type = 'dot', feature_names = X.columns)
#shap.summary_plot(RFCshap_values, X_test, feature_names = X.columns)
print(f'RFC Shapley value {RFCshap_values[1]}')
shap.force_plot(RFCExplainer.expected_value[0], RFCshap_values[0], X_test)

'''Explainer = shap.LinearExplainer(LR, X_test)
shap_values = Explainer.shap_values(X_test)
shap.initjs()
print(f'Logistics Regression Explainer {Explainer.expected_value}')
shap.summary_plot(shap_values = shap_values, features = X_test, plot_type = 'dot',
                 feature_names = X.columns)
shap.force_plot(Explainer.expected_value, shap_values, X_test)'''
#plt.savefig('Logistics Regression Shapley Value on Features.png')

'''svc_test = X_test_minmax[0:100]
svc_explainer = shap.KernelExplainer(svc.predict, svc_test)
svc_shap = svc_explainer.shap_values(X_test_minmax)
shap.initjs()
shap.summary_plot(svc_shap, X_test_minmax)
#plt.savefig('SV Classifier Shapley Values.png')'''

