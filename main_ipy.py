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
figure, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
figure.set_size_inches(14, 10)
_ = sns.regplot(train['TotalBsmtSF'], salePrices['SalePrice'], 
                ax=ax1)
_ = sns.regplot(train['1stFlrSF'], salePrices['SalePrice'], 
                ax=ax2)
_ = sns.regplot(train['2ndFlrSF'], salePrices['SalePrice'], 
                ax=ax3)
_ = sns.regplot(train['TotalBsmtSF']+train['1stFlrSF']+train['2ndFlrSF'], 
                salePrices['SalePrice'], 
                ax=ax4)

# +
# Impute the entire data set
all_data['TotalSF'] = all_data['TotalBsmtSF'] + \
                      all_data['1stFlrSF'] + \
                      all_data['2ndFlrSF']
        
# Add two new variables for No nd floor and no basement
all_data['No2ndFlr'] = (all_data['2ndFlrSF']==0)
all_data['NoBsmt'] = (all_data['TotalBsmtSF']==0)
# -


