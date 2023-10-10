import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV,ElasticNetCV,MultiTaskElasticNet, Lasso
from sklearn.model_selection import cross_val_score
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import random

def lasso(Xdata, y):
    model_lasso = LassoCV(alphas=[0.1,0.02,0.03,0.05,0.07]).fit(Xdata, y)
    print(model_lasso.alpha_)
    print(model_lasso.coef_)
    coef = pd.Series(model_lasso.coef_, index=Xdata.columns)

    rmse = np.sqrt(-cross_val_score(model_lasso, Xdata, y, scoring="neg_mean_squared_error", cv=3))
    print(rmse.mean())

    imp_coef = pd.concat([coef.sort_values().head(17),
                          coef.sort_values().tail(0)])
    plt.rcParams['figure.figsize'] = (8.0, 10.0)
    imp_coef.plot(kind="barh")
    plt.title("Coefficients in the Lasso Model")
    plt.show()

    l2l1 = ElasticNetCV(alphas=[0.1, 0.02, 0.03, 0.05, 0.07]).fit(Xdata, y)
    print('l2l1 ',l2l1.alpha_)
    rmse = np.sqrt(-cross_val_score(l2l1, Xdata, y, scoring="neg_mean_squared_error", cv=3))
    print(rmse.mean())
    print(l2l1.coef_)
    # 1 / (2 * n_samples) * | | y - Xw | | ^ 2_2
    # + alpha * l1_ratio * | | w | | _1
    # + 0.5 * alpha * (1 - l1_ratio) * | | w | | ^ 2_2

    #
    # a * L1 + b * L2
    #
    # alpha = a + b and l1_ratio = a / (a + b).
    radge = RidgeCV(alphas=[0.1, 0.02, 0.03, 0.05, 0.07]).fit(Xdata, y)
    print('radge',radge.alpha_)

    return Xdata.columns[model_lasso.coef_!=0]

def featurec(Xdata, y,k=15):
        lassofea=lasso(Xdata, y)
        print('leasso:',lassofea)

        selector=SelectKBest(chi2, k=k)
        selector.fit_transform(Xdata, y)
        chifea= Xdata.columns[selector.get_support()]
        print('chifea：',chifea)

        selector=SelectKBest(lambda X, Y: np.array(map(lambda x: pearsonr(x, Y), X.T)).T, k=2)
        selector.fit_transform(Xdata, y)
        peafea= Xdata.columns[selector.get_support()]
        print(peafea)

data_name ='CSDC2011'
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

    l1 = m1.iloc[list(index1)]
    l2 = m2.iloc[list(index2)]
    l3 = m3.iloc[list(index3)]
    l4 = m4.iloc[list(index4)]
    l5 = m5.iloc[list(index5)]
    lower_data=pd.concat([l1, l2,l3,l4,l5])

    return shuffle(lower_data)

lower_data=lower_sample_data(data_x,1)

y = lower_data['risk']
X=lower_data.drop('risk', axis=1)
X=X.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
all_name = X.columns.values.tolist()
print(len(all_name),all_name)

featurec(X, y,k=8)





