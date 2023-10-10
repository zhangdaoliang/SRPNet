import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import multiprocessing
import timeit
import operator

import xgboost as xgb
from joblib import Parallel, delayed
import multiprocessing

from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn import preprocessing  # 预处理模块
from sklearn.model_selection import RepeatedKFold,cross_validate
from sklearn.linear_model import Perceptron,LogisticRegression,LinearRegression
from sklearn.linear_model import SGDClassifier
from itertools import product
from itertools import combinations
import itertools

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV,ElasticNetCV,MultiTaskElasticNet, Lasso
from sklearn.model_selection import cross_val_score
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import random
def sampler(df, col, records):

  # Calculate number of rows
  colmax = df.count()

  # Create random sample from range
  vals = random.sample(range(1, colmax), records)

  # Use 'vals' to filter DataFrame using 'isin'
  return df.filter(df[col].isin(vals))

def my_permutation(num_list):

 res_list = []
 for i in range(1, len(num_list) + 1):
     res_list += list(combinations(num_list, i))
 return res_list

def Regressionsearch(Xdata,y,data_name,cv=5,njob=128):
    c = Xdata.columns.tolist()
    print(y.value_counts())
    c = Xdata.columns.tolist()
    print(y.value_counts())
    y = y.values
    allper = my_permutation(c)
    scoring = ['precision_macro', 'recall_macro']  # precision_macro为精度，recall_macro为召回率
    lenper  = len(allper)
    test_precision_macro = np.zeros([lenper, 8])
    test_recall_macro = np.zeros([lenper , 8])
    train_precision_macro = np.zeros([lenper , 8])
    train_recall_macro = np.zeros([lenper , 8])
    i = 0
    methodlist = []
    for k in allper:
        print(i,k)

def findbyml(i):
    cv=5
    njob=16
    k=allper[i]
    scoring = ['precision_macro', 'recall_macro']
    print(i, k)
    # methodlist.append(list(k))
    X = Xdata[list(k)].values

    clf = SVC(kernel='linear', C=1)
    scores = cross_validate(clf, X, y, scoring=scoring, cv=cv, return_train_score=True, n_jobs=njob)
    test_precision_macro[i, 0] = scores['test_precision_macro'].mean()
    test_recall_macro[i, 0] = scores['test_recall_macro'].mean()
    train_precision_macro[i, 0] = scores['train_precision_macro'].mean()
    train_recall_macro[i, 0] = scores['train_recall_macro'].mean()
    print(scores['train_recall_macro'].mean())

    knn = KNeighborsClassifier(n_neighbors=3)
    scores = cross_validate(knn, X, y, scoring=scoring, cv=cv, return_train_score=True, n_jobs=njob)
    test_precision_macro[i, 1] = scores['test_precision_macro'].mean()
    test_recall_macro[i, 1] = scores['test_recall_macro'].mean()
    train_precision_macro[i, 1] = scores['train_precision_macro'].mean()
    train_recall_macro[i, 1] = scores['train_recall_macro'].mean()

    gaussian = GaussianNB()
    scores = cross_validate(gaussian, X, y, scoring=scoring, cv=cv, return_train_score=True, n_jobs=njob)
    test_precision_macro[i, 2] = scores['test_precision_macro'].mean()
    test_recall_macro[i, 2] = scores['test_recall_macro'].mean()
    train_precision_macro[i, 2] = scores['train_precision_macro'].mean()
    train_recall_macro[i, 2] = scores['train_recall_macro'].mean()


    decision_tree = DecisionTreeClassifier()
    scores = cross_validate(decision_tree, X, y, scoring=scoring, cv=cv, return_train_score=True, n_jobs=njob)
    test_precision_macro[i, 3] = scores['test_precision_macro'].mean()
    test_recall_macro[i, 3] = scores['test_recall_macro'].mean()
    train_precision_macro[i, 3] = scores['train_precision_macro'].mean()
    train_recall_macro[i, 3] = scores['train_recall_macro'].mean()

    mlp = MLPClassifier(max_iter=2000)
    scores = cross_validate(mlp, X, y, scoring=scoring, cv=cv, return_train_score=True, n_jobs=njob)
    test_precision_macro[i, 4] = scores['test_precision_macro'].mean()
    test_recall_macro[i, 4] = scores['test_recall_macro'].mean()
    train_precision_macro[i, 4] = scores['train_precision_macro'].mean()
    train_recall_macro[i, 4] = scores['train_recall_macro'].mean()
    #
    gbdt = GradientBoostingClassifier()
    scores = cross_validate(gbdt, X, y, scoring=scoring, cv=cv, return_train_score=True, n_jobs=njob)
    test_precision_macro[i, 5] = scores['test_precision_macro'].mean()
    test_recall_macro[i, 5] = scores['test_recall_macro'].mean()
    train_precision_macro[i, 5] = scores['train_precision_macro'].mean()
    train_recall_macro[i, 5] = scores['train_recall_macro'].mean()

    adboost = AdaBoostClassifier()
    scores = cross_validate(adboost, X, y, scoring=scoring, cv=cv, return_train_score=True, n_jobs=njob)
    test_precision_macro[i, 6] = scores['test_precision_macro'].mean()
    test_recall_macro[i, 6] = scores['test_recall_macro'].mean()
    train_precision_macro[i, 6] = scores['train_precision_macro'].mean()
    train_recall_macro[i, 6] = scores['train_recall_macro'].mean()
    #
    rm = RandomForestClassifier()
    scores = cross_validate(rm, X, y, scoring=scoring, cv=cv, return_train_score=True, n_jobs=njob)
    test_precision_macro[i, 7] = scores['test_precision_macro'].mean()
    test_recall_macro[i, 7] = scores['test_recall_macro'].mean()
    train_precision_macro[i, 7] = scores['train_precision_macro'].mean()
    train_recall_macro[i, 7] = scores['train_recall_macro'].mean()

    with open("result/out.txt", "a") as file:
        file.write(
             "###;{};{};{};{};{};{};{}\n"
            .format(i,k,len(k),test_precision_macro[i],test_recall_macro[i],
                    train_precision_macro[i],train_recall_macro[i])
        )


def allsearch(Xdata,y,data_name,cv=2,njob=16):
    c = Xdata.columns.tolist()
    print(y.value_counts())
    y = y.values
    allper= my_permutation(c)
    scoring = ['precision_macro', 'recall_macro']  # precision_macro为精度，recall_macro为召回率


    print(lenper )

    items = [i for i in range(lenper )]
    p = multiprocessing.Pool(njob)

    b = p.map(findbyml,range(lenper ))
    # methodlist = "sssss"
    p.close()
    p.join()




    resultfile = 'result/test_precision_macro-' + data_name + '.csv'
    np.savetxt(resultfile, test_precision_macro, delimiter=',')
    resultfile = 'result/test_recall_macro-' + data_name + '.csv'
    np.savetxt(resultfile, test_recall_macro, delimiter=',')
    resultfile = 'result/train_precision_macro-' + data_name + '.csv'
    np.savetxt(resultfile, train_precision_macro, delimiter=',')
    resultfile = 'result/train_recall_macro-' + data_name + '.csv'
    np.savetxt(resultfile, train_recall_macro, delimiter=',')
    resultfile = 'result/methodlist-' + data_name + '.csv'
    np.savetxt(resultfile, allper, fmt='%s')




data_name = 'CSDC2011'
file_name = 'data/' + data_name + '.csv'
sname = 'data/sample-' + data_name + '.csv'
data_x = pd.read_csv(file_name)

data_x['NM'] =1
data_x.loc[data_x["aMarriage"] == 10, "NM"] = 0
data_x.loc[data_x["aMarriage"] == 23, "NM"] = 2
data_x.loc[data_x["aMarriage"] == 22, "NM"] = 2
data_x['MS'] = 1
data_x.loc[data_x["aMarriage"] == 10, "MS"] = 0
data_x.loc[data_x["aMarriage"] == 30, "MS"] = 0
data_x.loc[data_x["aMarriage"] == 40, "MS"] = 0
data_x['MO'] = 0
data_x.loc[data_x["aMarriage"] == 90 , "MO"] = 1

data_x.loc[data_x["aNation"] != 1, "aNation"] = 0

data_x.loc[data_x["aCategory"] == 50, "aCategory"] = 0
data_x.loc[data_x["aCategory"] == 60, "aCategory"] = 1
# 1北京 2山东 3河南 4山西 5陕西 6四川
data_x['pgdp'] = 0
data_x.loc[data_x["Province"] ==1 , "pgdp"] = 80394
data_x.loc[data_x["Province"] == 2, "pgdp"] = 46976
data_x.loc[data_x["Province"] == 3, "pgdp"] = 28716
data_x.loc[data_x["Province"] == 4, "pgdp"] = 30802
data_x.loc[data_x["Province"] == 5, "pgdp"] = 33197
data_x.loc[data_x["Province"] == 6, "pgdp"] = 26147
data_x['PLA'] = 0
data_x.loc[data_x["Province"] ==1 , "PLA"] = 39.9
data_x.loc[data_x["Province"] == 2, "PLA"] = 36.4
data_x.loc[data_x["Province"] == 3, "PLA"] = 33.9
data_x.loc[data_x["Province"] == 4, "PLA"] = 37.5
data_x.loc[data_x["Province"] == 5, "PLA"] = 35.6
data_x.loc[data_x["Province"] == 6, "PLA"] = 30.1
data_x['PLO'] = 0
data_x.loc[data_x["Province"] ==1 , "PLO"] = 116.4
data_x.loc[data_x["Province"] == 2, "PLO"] = 118.5
data_x.loc[data_x["Province"] == 3, "PLO"] = 113.5
data_x.loc[data_x["Province"] == 4, "PLO"] = 112.3
data_x.loc[data_x["Province"] == 5, "PLO"] = 118.3
data_x.loc[data_x["Province"] == 6, "PLO"] = 100.4

data_x['PP'] = 0
data_x.loc[data_x["Province"] ==1 , "PP"] = 500
data_x.loc[data_x["Province"] == 2, "PP"] = 710
data_x.loc[data_x["Province"] == 3, "PP"] = 700
data_x.loc[data_x["Province"] == 4, "PP"] = 520
data_x.loc[data_x["Province"] == 5, "PP"] = 600
data_x.loc[data_x["Province"] == 6, "PP"] = 1100


data_x['PHT'] = 0
data_x.loc[data_x["Province"] ==1 , "PHT"] = 27.3
data_x.loc[data_x["Province"] == 2, "PHT"] = 27.1
data_x.loc[data_x["Province"] == 3, "PHT"] = 27.1
data_x.loc[data_x["Province"] == 4, "PHT"] = 24
data_x.loc[data_x["Province"] == 5, "PHT"] = 31.6
data_x.loc[data_x["Province"] == 6, "PHT"] = 25.4

data_x['PLT'] = 0
data_x.loc[data_x["Province"] ==1 , "PLT"] = -3
data_x.loc[data_x["Province"] == 2, "PLT"] = -0.6
data_x.loc[data_x["Province"] == 3, "PLT"] = 0.5
data_x.loc[data_x["Province"] == 4, "PLT"] = -5
data_x.loc[data_x["Province"] == 5, "PLT"] = -0.3
data_x.loc[data_x["Province"] == 6, "PLT"] = 5.6

data_x['PHH'] = 0
data_x.loc[data_x["Province"] ==1 , "PHH"] = 71
data_x.loc[data_x["Province"] == 2, "PHH"] = 77
data_x.loc[data_x["Province"] == 3, "PHH"] = 79
data_x.loc[data_x["Province"] == 4, "PHH"] = 74
data_x.loc[data_x["Province"] == 5, "PHH"] = 82
data_x.loc[data_x["Province"] == 6, "PHHx"] = 86

data_x['PLH'] = 0
data_x.loc[data_x["Province"] ==1 , "PLH"] = 42
data_x.loc[data_x["Province"] == 2, "PLH"] = 45
data_x.loc[data_x["Province"] == 3, "PLH"] = 59
data_x.loc[data_x["Province"] == 4, "PLH"] = 45
data_x.loc[data_x["Province"] == 5, "PLH"] = 63
data_x.loc[data_x["Province"] == 6, "PLH"] = 77

data_x = data_x.drop('Province', axis=1)
data_x =data_x.drop('aMarriage', axis=1)
print(data_x.shape)



def lower_sample_data(df, percent=0.5):

    m1 = df[df['risk'] == 1]
    m2 = df[df['risk'] == 2]
    m3 = df[df['risk'] == 3]
    m4 = df[df['risk'] == 4]
    m5 = df[df['risk'] == 5]
    n_m5=len(m5)

    index1 = np.random.randint(len(m1), size=int(percent *n_m5) )
    index2 = np.random.randint(len(m2), size=int(percent * n_m5))
    index3 = np.random.randint(len(m3), size=int(percent * n_m5))
    index4 = np.random.randint(len(m4), size=int(percent * n_m5))
    index5 = np.random.randint(len(m5), size=int(percent * n_m5))
    #下采样后数据样本
    l1 = m1.iloc[list(index1)]
    l2 = m2.iloc[list(index2)]
    l3 = m3.iloc[list(index3)]
    l4 = m4.iloc[list(index4)]
    l5 = m5.iloc[list(index5)]
    lower_data=pd.concat([l1, l2,l3,l4,l5])

    return shuffle(lower_data)

lower_data=lower_sample_data(data_x,1)

Xdata=lower_data.drop('risk', axis=1)
Xdata=lower_data[["AG", "Gender", "Smoking", "MS", "Occupation", "ES", "HS",
                  "HYP", "AF", "LDL-C", "Diabetes", "LE", "Overweight", "FHS/HYP/CHD", "PLT",  "PLH"]]

y = lower_data['risk']
all_name = Xdata.columns.values.tolist()
Xdata=Xdata.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
print(all_name)
c = Xdata.columns.tolist()
allper = my_permutation(c)
lenper  = len(allper)
print(lenper )


print(Xdata.shape)
print("...................")
allsearch(Xdata,y,data_name,cv=5,njob=64)
# Regressionsearch(Xdata,y,data_name,cv=5,njob=128)


