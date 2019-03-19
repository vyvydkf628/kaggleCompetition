# Loading packages
import matplotlib

import pandas as pd #Analysis
import matplotlib.pyplot as plt #Visulization
import seaborn as sns #Visulization
import numpy as np #Analysis
from scipy.stats import norm #Analysis
from sklearn.preprocessing import StandardScaler #Analysis
from scipy import stats #Analysis
import warnings
warnings.filterwarnings('ignore')
import os
from os.path import join
import missingno as msno
import sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
import xgboost as xgb
import lightgbm as lgb


import gc
df_train = pd.read_csv('C:/train.csv')
df_test  = pd.read_csv('C:/test.csv')
print("train.csv. Shape: ",df_train.shape)
print("test.csv. Shape: ",df_test.shape)

df_train.head()
df_train['price'].describe()

#histogram
f, ax = plt.subplots(figsize=(8, 6))
sns.distplot(df_train['price'])

fig = plt.figure(figsize = (15,10))

fig.add_subplot(1,2,1)
res = stats.probplot(df_train['price'], plot=plt)

fig.add_subplot(1,2,2)
res = stats.probplot(np.log1p(df_train['price']), plot=plt)

df_train['price'] = np.log1p(df_train['price'])
f, ax = plt.subplots(figsize=(8,6))
sns.distplot(df_train['price'])

gboost = GradientBoostingRegressor(random_state=2019)
xgboost = xgb.XGBRegressor(random_state=2019)
lightgbm = lgb.LGBMRegressor(random_state=2019)

models = [{'model':gboost, 'name':'GradientBoosting'}, {'model':xgboost, 'name':'XGBoost'},
          {'model':lightgbm, 'name':'LightGBM'}]
