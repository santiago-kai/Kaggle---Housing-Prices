# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# pandas
import pandas as pd
# from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
# jupyter notebook
# %matplotlib inline

from IPython.display import display

# remove warnings
import warnings
warnings.filterwarnings('ignore')

# check input data
import os
print(os.listdir('./input'))

# +
# load date with continous counting Id 
train = pd.read_csv('./input/train.csv', index_col='Id')
test  = pd.read_csv('./input/test.csv',  index_col='Id')

print(train.shape)
display(train.head())

print(test.shape)
display(test.head())
# -

# matplot flat style 
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)   #size
# plt.rcParams['savefig.dpi']  = 300       #pixel
# plt.rcParams['figure.dpi']   = 300       #resolution
train.head(5)
train.SalePrice.describe()
print("Skew is:", train.SalePrice.skew())
plt.hist(train.SalePrice, color='blue')
plt.show()

# select numeric features
numeric_feature = train.select_dtypes(include=[np.number])
numeric_feature.dtypes

# analyse features correlated with each other
corr = numeric_feature.corr()
# select features corr-to SalePrice 
print(corr['SalePrice'].sort_values(ascending=False)[1:11], '\n')
print(corr['SalePrice'].sort_values(ascending=False)[-10:])

# How many unique features
train.OverallQual.unique()


# Pivot and plot the intended aggregate function
def pivotandplot(data, variable, onVariable, aggfunc):
    pivot_var = data.pivot_table(index   = variable,
                                 values  = onVariable,
                                 aggfunc = aggfunc)
    pivot_var.plot(kind='bar', color='blue')
    plt.xlabel(variable)
    plt.ylabel(onVariable)
    plt.xticks(rotation=0) #xlabel rotation angle
    plt.show()


pivotandplot(train, 'OverallQual', 'SalePrice', np.median)

_ = sns.regplot(train['GrLivArea'], train['SalePrice'])

# Remove the outliers(2 points in rightdown corner)
train = train.drop(train[(train['GrLivArea']>4000) & 
                         (train['SalePrice']<300000)].index)
_ = sns.regplot(train['GrLivArea'], train['SalePrice'])

_ = sns.regplot(train['GarageArea'], train['SalePrice'])

train = train[train['GarageArea']<1200]
_ = sns.regplot(train['GarageArea'], train['SalePrice'])
# Do not delete data with 0 GarageArea

# ### Impute the Data for missing values
# * Before imputing the categorical values it is very important that we impute it on entire (Dev + Test)
# * This is of outmost importance since some of the categories might be missing in the test data  which will create problems in OneHotEncoding later when we run models on test data

# Merge train and text data
# get the ln of SalePrice, +1 because ln0 is illogical
train['log_SalePrice'] = np.log(train['SalePrice']+1)
# 2-dimension array use double []
salePrices = train[['SalePrice', 'log_SalePrice']]
salePrices.head()

# make sure the features are the same in two dataset
train = train.drop(columns=['SalePrice', 'log_SalePrice'])
print(train.shape)
print(test.shape)

all_data = pd.concat((train, test))
print(all_data.shape)
all_data.head()

# +
# processing null data
null_data = pd.DataFrame(all_data.isnull().sum().
                         sort_values(ascending=False))[:50]

null_data.columns = ['Null Count']
null_data.index.name = 'Feature'
null_data
# -

# get the null percentage
(null_data/len(all_data)) * 100

# * 99% of Pool Quality Data is missing.In the case of PoolQC, the column refers to Pool Quality. Pool quality is NaN when PoolArea is 0, or there is no pool.
# * Similar is case for Garage column
#
# But what are the 96% missing Miscelleanous features ?

print("Unique values are:", train.MiscFeature.unique())

# Impute Categorical features for missing values and relace by 'None'
for col in ('PoolQC', 'MiscFeature', 'Alley', 'Fence', 
            'FireplaceQu', 'GarageType', 'GarageFinish', 
            'GarageQual', 'GarageCond', 'BsmtQual', 
            'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
            'BsmtFinType2', 'MasVnrType', 'MSSubClass'):
    all_data[col] = all_data[col].fillna('None')

# Impute Numerical features for missing values and replace by zero
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars', 
            'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
            'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 
            'MasVnrArea'):
    all_data[col] = all_data[col].fillna(0)

# mode(): getting the most frequent number, return a df use the 1st
for col in ('MSZoning', 'Electrical', 'KitchenQual', 
            'Exterior1st', 'Exterior2nd', 'SaleType', 
            'Functional', 'Utilities'):
    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])

_ = sns.regplot(train['LotFrontage'], salePrices['SalePrice'])

# Impute the LotFrontage with Median values
all_data['LotFrontage'] = all_data.groupby('Neighborhood')[
    'LotFrontage'].apply(lambda x: x.fillna(x.median()))

# ### New Features
# * TotalBsmtSF - Total Basement Square Feet
# * 1stFlrSF - First Floor Square Feet
# * 2ndFlrSF - Second Floor Square Feet
#
# All the above three feature define area of the house and we can easily combine these to form TotalSF - Total Area in square feet

# check if this thought works
figure, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, 
                                                ncols=2)
figure.set_size_inches(14, 10)
_ = sns.regplot(train['TotalBsmtSF'], salePrices['SalePrice'], 
                ax=ax1)
_ = sns.regplot(train['1stFlrSF'], salePrices['SalePrice'], 
                ax=ax2)
_ = sns.regplot(train['2ndFlrSF'], salePrices['SalePrice'], 
                ax=ax3)
_ = sns.regplot(train['TotalBsmtSF']+train['1stFlrSF']+\
                train['2ndFlrSF'], 
                salePrices['SalePrice'], 
                ax=ax4)

# +
# Impute the entire data set
all_data['TotalSF'] = all_data['TotalBsmtSF'] + \
                      all_data['1stFlrSF'] + \
                      all_data['2ndFlrSF']
# why not drop ???    
    
# Add two new variables for No nd floor and no basement
all_data['No2ndFlr'] = (all_data['2ndFlrSF']==0)
all_data['NoBsmt'] = (all_data['TotalBsmtSF']==0)
# -
figure, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=3, ncols=2)
figure.set_size_inches(14, 15)
_ = sns.barplot(train['BsmtFullBath'], salePrices['SalePrice'], 
                ax=ax1)
_ = sns.barplot(train['FullBath'], salePrices['SalePrice'], 
                ax=ax2)
_ = sns.barplot(train['BsmtHalfBath'], salePrices['SalePrice'], 
                ax=ax3)
_ = sns.barplot(train['HalfBath'], salePrices['SalePrice'], 
                ax=ax4)
_ = sns.barplot(train['BsmtFullBath']+train['FullBath']+train['BsmtHalfBath']+train['HalfBath'], 
                salePrices['SalePrice'], 
                ax=ax5)


all_data['TotalBath'] = all_data['BsmtFullBath'] + \
                        all_data['FullBath'] + \
                        all_data['BsmtHalfBath'] + \
                        all_data['HalfBath']
# why not drop ??? 

figure, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
figure.set_size_inches(18, 8)
_ = sns.regplot(train['YearBuilt'], salePrices['SalePrice'], ax=ax1)
_ = sns.regplot(train['YearRemodAdd'], salePrices['SalePrice'], ax=ax2)
_ = sns.regplot((train['YearBuilt']+train['YearRemodAdd'])/2, 
                salePrices['SalePrice'], ax=ax3)
# why use the average value ???

all_data['YrBltAndRemod'] = all_data['YearBuilt'] + \
                            all_data['YearRemodAdd']

# ## Some features dominating is over 97%

print(all_data.Street.unique())
print(all_data['Street'].value_counts())
all_data['Street'].value_counts().head(3).plot.bar()

# Deleting dominating features over 97%
all_data=all_data.drop(columns=['Street','Utilities',
                                'Condition2','RoofMatl',
                                'Heating'])
# why doing this ???

# treat some numeric values as str which is actually a categorical data
all_data['MSSubClass']=all_data['MSSubClass'].astype(str)
all_data['MoSold']=all_data['MoSold'].astype(str)
all_data['YrSold']=all_data['YrSold'].astype(str)

# I found these features might look better without 0 data. (just like the column '2ndFlrSF' above.)
# why and how you found this ???
all_data['NoLowQual']=(all_data['LowQualFinSF']==0)
all_data['NoOpenPorch']=(all_data['OpenPorchSF']==0)
all_data['NoWoodDeck']=(all_data['WoodDeckSF']==0)
all_data['NoGarage']=(all_data['GarageArea']==0)
# most of the houses has no pools. 
all_data=all_data.drop(columns=['PoolArea','PoolQC'])
# most of the houses has no misc feature.
all_data=all_data.drop(columns=['MiscVal','MiscFeature'])

# Group the similar featurtes related to a House Feature and analyze
Basement = ['BsmtCond', 'BsmtExposure', 'BsmtFinSF1', 
            'BsmtFinSF2', 'BsmtFinType1', 'BsmtFinType2', 
            'BsmtQual', 'BsmtUnfSF','TotalBsmtSF']
Bsmt=all_data[Basement]

Bsmt['BsmtCond'].unique()

# BsmtQual: Evaluates the height of the basement
#
#        Ex	Excellent (100+ inches)	
#        Gd	Good (90-99 inches)
#        TA	Typical (80-89 inches)
#        Fa	Fair (70-79 inches)
#        Po	Poor (<70 inches
#        NA	No Basement
# 		
# BsmtCond: Evaluates the general condition of the basement
#
#        Ex	Excellent
#        Gd	Good
#        TA	Typical - slight dampness allowed
#        Fa	Fair - dampness or some cracking or settling
#        Po	Poor - Severe cracking, settling, or wetness
#        NA	No Basement
# 	
# BsmtExposure: Refers to walkout or garden level walls
#
#        Gd	Good Exposure
#        Av	Average Exposure (split levels or foyers typically score average or above)	
#        Mn	Mimimum Exposure
#        No	No Exposure
#        NA	No Basement
# 	
# BsmtFinType1: Rating of basement finished area
#
#        GLQ	Good Living Quarters
#        ALQ	Average Living Quarters
#        BLQ	Below Average Living Quarters	
#        Rec	Average Rec Room
#        LwQ	Low Quality
#        Unf	Unfinshed
#        NA	No Basement
# 		
# BsmtFinSF1: Type 1 finished square feet
#
# BsmtFinType2: Rating of basement finished area (if multiple types)
#
#        GLQ	Good Living Quarters
#        ALQ	Average Living Quarters
#        BLQ	Below Average Living Quarters	
#        Rec	Average Rec Room
#        LwQ	Low Quality
#        Unf	Unfinshed
#        NA	No Basement
#
# BsmtFinSF2: Type 2 finished square feet
#
# BsmtUnfSF: Unfinished square feet of basement area
#
# TotalBsmtSF: Total square feet of basement area

Bsmt.head()

# +
from sklearn.preprocessing import LabelEncoder
# Replacing Categorical values to numbers
cond_encoder = LabelEncoder()
Bsmt['BsmtCond'] = cond_encoder.fit_transform(Bsmt['BsmtCond'])

exposure_encoder = LabelEncoder()
Bsmt['BsmtExposure'] = exposure_encoder.fit_transform(Bsmt['BsmtExposure'])

finTyp1_encoder = LabelEncoder()
Bsmt['BsmtFinType1'] = finTyp1_encoder.fit_transform(Bsmt['BsmtFinType1'])

finTyp2_encoder = LabelEncoder()
Bsmt['BsmtFinType2'] = finTyp2_encoder.fit_transform(Bsmt['BsmtFinType2'])

qual_encoder = LabelEncoder()
Bsmt['BsmtQual'] = qual_encoder.fit_transform(Bsmt['BsmtQual'])

Bsmt.head()
# -
# design a function to calculate a score
Bsmt['BsmtScore'] = Bsmt['BsmtQual'] * \
                    Bsmt['BsmtCond'] * \
                    Bsmt['TotalBsmtSF']
all_data['BsmtScore']=Bsmt['BsmtScore']


Bsmt['BsmtFin'] = (Bsmt['BsmtFinSF1'] * Bsmt['BsmtFinType1']) \
                + (Bsmt['BsmtFinSF2'] * Bsmt['BsmtFinType2'])
all_data['BsmtFinScore']=Bsmt['BsmtFin']
all_data['BsmtDNF']=(all_data['BsmtFinScore']==0)

# LotFrontage: Linear feet of street connected to property
#
# LotArea: Lot size in square feet
#
# LotShape: General shape of property
#
# - Reg  Regular 
# - IR1  Slightly irregular
# - IR2  Moderately Irregular
# - IR3  Irregular
#
# LotConfig: Lot configuration
#
# - Inside   Inside lot
# - Corner   Corner lot
# - CulDSac  Cul-de-sac
# - FR2  Frontage on 2 sides of property
# - FR3  Frontage on 3 sides of property

lot=['LotFrontage', 'LotArea','LotConfig','LotShape']
Lot=all_data[lot]
Lot.head()

garage=['GarageArea','GarageCars','GarageCond','GarageFinish',
        'GarageQual','GarageType','GarageYrBlt']
Garage=all_data[garage]
Garage.head()

# +
garcond_encoder = LabelEncoder()
Garage['GarageCond'] = garcond_encoder.fit_transform(
                       Garage['GarageCond'])

garfin_encoder = LabelEncoder()
Garage['GarageFinish'] = garfin_encoder.fit_transform(
                         Garage['GarageFinish'])

garqual_encoder = LabelEncoder()
Garage['GarageQual'] = garqual_encoder.fit_transform(
                       Garage['GarageQual'])

gartyp_encoder = LabelEncoder()
Garage['GarageType'] = gartyp_encoder.fit_transform(
                       Garage['GarageType'])

Garage.head()
# -

Garage['GarageScore'] = Garage['GarageArea'] * \
                        Garage['GarageCars'] * \
                        Garage['GarageFinish'] * \
                        Garage['GarageQual'] * \
                        Garage['GarageType']
all_data['GarageScore']=Garage['GarageScore']
all_data.head()

# ### Data Preprocessesing

# don't want boolean either
non_numeric = all_data.select_dtypes(exclude=[np.number, bool])
non_numeric.head()


# encode function
def onehot(col_list):
    global all_data
    while len(col_list) != 0:
        # return column 0
        col = col_list.pop(0)
        # convert(encode) categorical date into dummy data 
        data_encoded = pd.get_dummies(all_data[col], prefix=col)
        all_data=pd.merge(all_data, data_encoded, on='Id')
        all_data=all_data.drop(columns=col)
    print(all_data.shape)


# list function return the features name list
onehot(list(non_numeric))


def log_transform(col_list):
    transformed_col = []
    while len(col_list) != 0:
        col = col_list.pop(0)
        if all_data[col].skew() > 0.5:
            all_data[col] = np.log(all_data[col]+1)
            transformed_col.append(col)
        else:
            pass
    print(f'{len(transformed_col)} features had been transformed')
    print(all_data.shape)


numeric=all_data.select_dtypes(include=np.number)
log_transform(list(numeric))

print(train.shape)
print(test.shape)
train = all_data[:len(train)]
test = all_data[len(train):]
print(train.shape)
print(test.shape)

# ### Modeling

# +
# loading pakages for model. 
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

from sklearn import linear_model, model_selection, ensemble, preprocessing
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb

#Evaluation Metrics
from sklearn.metrics import mean_squared_error, mean_squared_log_error, r2_score,mean_absolute_error


# +
def rmse(predict, actual):
    score = mean_squared_error(Ytrain, y_pred) ** 0.5
    return score
rmse_score = make_scorer(rmse)

feature_names = list(all_data)
Xtrain = train[feature_names]
Xtest  = test[feature_names]
Ytrain = salePrices['log_SalePrice']

def score(model):
    score = cross_val_score(model, Xtrain, Ytrain, cv=5, 
                            scoring=rmse_score).mean()
    return score

scores = {}
# -

# #### Simple Linear Regression

# +
lr_model = LinearRegression(n_jobs=-1)
lr_model.fit(Xtrain, Ytrain)

#accuracies = cross_val_score(estimator=lr_model,
                         #   X=Xtrain,
                         #   y=Ytrain,
                          #  cv=5,
                          #  verbose=1)
                
y_pred = lr_model.predict(Xtrain)

print('')
print('####### Linear Regression #######')
meanCV = score(lr_model)
print('Mean CV Score : %.4f' % meanCV)

mse = mean_squared_error(Ytrain,y_pred)
mae = mean_absolute_error(Ytrain, y_pred)
rmse = mean_squared_error(Ytrain, y_pred)**0.5
r2 = r2_score(Ytrain, y_pred)
scores.update({'OLS':[meanCV,mse,mae,rmse,r2]})

print('')
print('MSE(RSS)    : %0.4f ' % mse)
print('MAE         : %0.4f ' % mae)
print('RMSE        : %0.4f ' % rmse)
print('R2          : %0.4f ' % r2)
# -

# #### Lasso Regression

# +
model_lasso = Lasso(random_state=42, alpha=0.00035)
lr_lasso = make_pipeline(RobustScaler(), model_lasso)
lr_lasso.fit(Xtrain, Ytrain)

y_pred = lr_lasso.predict(Xtrain)

print('')
print('####### Lasso Regression #######')
meanCV = score(lr_lasso)
print('Mean CV Score : %.4f' % meanCV)

mse = mean_squared_error(Ytrain,y_pred)
mae = mean_absolute_error(Ytrain, y_pred)
rmse = mean_squared_error(Ytrain, y_pred)**0.5
r2 = r2_score(Ytrain, y_pred)
scores.update({'Lasso':[meanCV,mse,mae,rmse,r2]})

print('')
print('MSE(RSS)    : %0.4f ' % mse)
print('MAE         : %0.4f ' % mae)
print('RMSE        : %0.4f ' % rmse)
print('R2          : %0.4f ' % r2)
# -

# #### Ridge Regression

# +
lr_ridge = make_pipeline(RobustScaler(), 
                         Ridge(random_state=42,alpha=0.002))
lr_ridge.fit(Xtrain,Ytrain)

y_pred = lr_ridge.predict(Xtrain)

print('')
print('####### Ridge Regression #######')
meanCV = score(lr_ridge)
print('Mean CV Score : %.4f' % meanCV)


mse = mean_squared_error(Ytrain,y_pred)
mae = mean_absolute_error(Ytrain, y_pred)
rmse = mean_squared_error(Ytrain, y_pred)**0.5
r2 = r2_score(Ytrain, y_pred)
scores.update({'Ridge':[meanCV,mse,mae,rmse,r2]})

print('')
print('MSE(RSS)    : %0.4f ' % mse)
print('MAE         : %0.4f ' % mae)
print('RMSE        : %0.4f ' % rmse)
print('R2          : %0.4f ' % r2)
# -

# #### ElasticNet Regression

# +
lr_elasticnet = make_pipeline(RobustScaler(),
                              ElasticNet(alpha=0.02, l1_ratio=0.7,random_state=42))
lr_elasticnet.fit(Xtrain,Ytrain)

y_pred = lr_elasticnet.predict(Xtrain)

print('')
print('####### ElasticNet Regression #######')
meanCV = score(lr_elasticnet)
print('Mean CV Score : %.4f' % meanCV)

mse = mean_squared_error(Ytrain,y_pred)
mae = mean_absolute_error(Ytrain, y_pred)
rmse = mean_squared_error(Ytrain, y_pred)**0.5
r2 = r2_score(Ytrain, y_pred)
scores.update({'ElasticNet':[meanCV,mse,mae,rmse,r2]})

print('')
print('MSE(RSS)    : %0.4f ' % mse)
print('MAE         : %0.4f ' % mae)
print('RMSE        : %0.4f ' % rmse)
print('R2          : %0.4f ' % r2)
# -

# #### KNN Regression

# +
knn = make_pipeline(RobustScaler(),KNeighborsRegressor())
knn.fit(Xtrain,Ytrain)

y_pred = knn.predict(Xtrain)

print('')
print('####### KNN Regression #######')
meanCV = score(knn)
print('Mean CV Score : %.4f' % meanCV)

mse = mean_squared_error(Ytrain,y_pred)
mae = mean_absolute_error(Ytrain, y_pred)
rmse = mean_squared_error(Ytrain, y_pred)**0.5
r2 = r2_score(Ytrain, y_pred)
scores.update({'KNN':[meanCV,mse,mae,rmse,r2]})

print('')
print('MSE(RSS)    : %0.4f ' % mse)
print('MAE         : %0.4f ' % mae)
print('RMSE        : %0.4f ' % rmse)
print('R2          : %0.4f ' % r2)
# -

# #### GradientBoosting Regression

# +
model_GBoost = GradientBoostingRegressor(n_estimators=3000, 
                                         learning_rate=0.05,
                                         max_depth=4, 
                                         max_features='sqrt',
                                         min_samples_leaf=15, 
                                         min_samples_split=10,
                                         loss='huber', 
                                         random_state =42)
model_GBoost.fit(Xtrain,Ytrain)

y_pred = model_GBoost.predict(Xtrain)

print('')
print('####### GradientBoosting Regression #######')
meanCV = score(model_GBoost)
print('Mean CV Score : %.4f' % meanCV)

mse = mean_squared_error(Ytrain,y_pred)
mae = mean_absolute_error(Ytrain, y_pred)
rmse = mean_squared_error(Ytrain, y_pred)**0.5
r2 = r2_score(Ytrain, y_pred)
scores.update({'GradientBoosting':[meanCV,mse,mae,rmse,r2]})

print('')
print('MSE(RSS)    : %0.4f ' % mse)
print('MAE         : %0.4f ' % mae)
print('RMSE        : %0.4f ' % rmse)
print('R2          : %0.4f ' % r2)
# -

# #### RandomForest Regressor

# +
forest_reg = RandomForestRegressor(random_state=42)
forest_reg.fit(Xtrain, Ytrain)

y_pred = forest_reg.predict(Xtrain)

print('')
print('####### RandomForest Regression #######')
meanCV = score(forest_reg)
print('Mean CV Score : %.4f' % meanCV)


mse = mean_squared_error(Ytrain,y_pred)
mae = mean_absolute_error(Ytrain, y_pred)
rmse = mean_squared_error(Ytrain, y_pred)**0.5
r2 = r2_score(Ytrain, y_pred)
scores.update({'RandomForest':[meanCV,mse,mae,rmse,r2]})

print('')
print('MSE(RSS)    : %0.4f ' % mse)
print('MAE         : %0.4f ' % mae)
print('RMSE        : %0.4f ' % rmse)
print('R2          : %0.4f ' % r2)
# -

# #### Grid Search for finding best params for RandomForest

# +
from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [70,100], 'max_features': [150]},
    {'bootstrap': [True], 'n_estimators': [70,100], 
     'max_features': [150]},
  ]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error', 
                           return_train_score=True)
grid_search.fit(Xtrain, Ytrain)

y_pred = grid_search.predict(Xtrain) #???? missing

print('')
print('####### GridSearch RF Regression #######')
meanCV = score(grid_search)
print('Mean CV Score : %.4f' % meanCV)

mse = mean_squared_error(Ytrain,y_pred)
mae = mean_absolute_error(Ytrain, y_pred)
rmse = mean_squared_error(Ytrain, y_pred)**0.5
r2 = r2_score(Ytrain, y_pred)
scores.update({'GridSearchRF':[meanCV,mse,mae,rmse,r2]})

print('')
print('MSE(RSS)    : %0.4f ' % mse)
print('MAE         : %0.4f ' % mae)
print('RMSE        : %0.4f ' % rmse)
print('R2          : %0.4f ' % r2)
# -

grid_search.best_estimator_

# +
scores_list =[]
for k,v in scores.items():
    temp_lst =[]
    temp_lst.append(k)
    temp_lst.extend(v)
    scores_list.append(temp_lst)
    
scores_df = pd.DataFrame(scores_list, 
                         columns=['Model','CV_Mean_Score',
                                  'MSE(RSS)','MAE','RMSE',
                                  'R2Squared'])

scores_df.sort_values(['CV_Mean_Score'])
# -

_ =sns.scatterplot(x='Model',y='CV_Mean_Score',
                   data=scores_df,style='Model')

# +
Lasso_Predictions=np.exp(lr_lasso.predict(Xtest))-1

GBoost_Predictions=np.exp(model_GBoost.predict(Xtest))-1

KNN_Predictions=np.exp(knn.predict(Xtest))-1

GridSearch_Predictions = np.exp(grid_search.best_estimator_.predict(Xtest))-1
# -

submission=pd.read_csv('./input/sample_submission.csv')

# +
submission['SalePrice'] = Lasso_Predictions
submission.to_csv('./input/Lasso.csv',index=False)

submission['SalePrice'] = GBoost_Predictions
submission.to_csv('./input/GBoost.csv',index=False)

submission['SalePrice'] = KNN_Predictions
submission.to_csv('./input/KNN.csv',index=False)

submission['SalePrice'] = GridSearch_Predictions
submission.to_csv('./input/GidSearch.csv',index=False)
# -


