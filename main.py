# Code you have previously used to load data
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer
# from learntools.core import *

# Path of the file to read. We changed the directory structure to simplify submitting to a competition
iowa_file_path = './input/train.csv'

home_data = pd.read_csv(iowa_file_path)
# print(home_data.columns.values)

# Create target object and call it y
y = home_data.SalePrice
# Create X
# get all columns's name list    ---- list(home_data.columns.values)
features = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
            'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 
            'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 
            'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 
            'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
            'MiscVal', 'MoSold', 'YrSold']
X = home_data[features]

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# An Extension To Imputation----------------------------------------------------------------------------------
# make copy to avoid changing original data (when Imputing)
imputed_X_train_plus = train_X.copy()
imputed_X_val_plus = val_X.copy()

# make new columns indicating what will be imputed
# cols_with_missing = (col for col in train_X.columns 
#                                  if train_X[col].isnull().any())
# for col in cols_with_missing:
#     imputed_X_train_plus[col + '_was_missing'] = imputed_X_train_plus[col].isnull()
#     imputed_X_val_plus[col + '_was_missing'] = imputed_X_val_plus[col].isnull()
# print(train_X.head())
# print(imputed_X_train_plus.head())

# Imputation
my_imputer = SimpleImputer()
imputed_X_train_plus = my_imputer.fit_transform(imputed_X_train_plus)
imputed_X_val_plus = my_imputer.transform(imputed_X_val_plus)

# print("Mean Absolute Error from Imputation while Track What Was Imputed:")
# print(score_dataset(imputed_X_train_plus, imputed_X_val_plus, train_y, val_y))
# ----------------------------------------------------------------------------------An Extension To Imputation

# Specify Model
iowa_model = DecisionTreeRegressor(random_state=1)
# Fit Model
iowa_model.fit(imputed_X_train_plus, train_y)

# Make validation predictions and calculate mean absolute error
val_predictions = iowa_model.predict(imputed_X_val_plus)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))

# Using best value for max_leaf_nodes
iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
iowa_model.fit(imputed_X_train_plus, train_y)
val_predictions = iowa_model.predict(imputed_X_val_plus)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))

# Define the model. Set random_state to 1
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(imputed_X_train_plus, train_y)
rf_val_predictions = rf_model.predict(imputed_X_val_plus)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)
print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))

# Creating a Model For the Competition
# Build a Random Forest model and train it on all of X and y.

# To improve accuracy, create a new Random Forest model which you will train on all training data
rf_model_on_full_data = RandomForestRegressor(max_leaf_nodes = 340, random_state=1)

# An Extension To Imputation----------------------------------------------------------------------------------
# make copy to avoid changing original data (when Imputing)
imputed_X = X.copy()

# make new columns indicating what will be imputed
# cols_with_missing = (col for col in X.columns 
#                                  if X[col].isnull().any())
# for col in cols_with_missing:
#     imputed_X[col + '_was_missing'] = imputed_X[col].isnull()

# Imputation
my_imputer = SimpleImputer()
imputed_X = my_imputer.fit_transform(imputed_X)
# ----------------------------------------------------------------------------------An Extension To Imputation

# print(imputed_X.shape[1])  #feature number
# fit rf_model_on_full_data on all data from the training data
rf_model_on_full_data.fit(imputed_X, y)
print('fit done!')

# To get best leaf nodes
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, random_state=1)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

leaf_nodes = [100, 300, 310, 320, 330, 340, 350, 360, 370, 380, 400, 500, 500, 580, 600, 610, 620, 630, 650, 700, 1000]
mae_dict = {max_leaf_nodes: get_mae(max_leaf_nodes, imputed_X_train_plus, imputed_X_val_plus, train_y, val_y) 
            for max_leaf_nodes in leaf_nodes}
best_leaf_nodes = min(mae_dict, key = mae_dict.get)
print(best_leaf_nodes)

# Make Predictions
# Read the file of "test" data. And apply your model to make predictions

# path to file you will use for predictions
test_data_path = './input/test.csv'

# read test data file using pandas
test_data = pd.read_csv(test_data_path)
# test_data.describe()

# create test_X which comes from test_data but includes only the columns you used for prediction.
# The list of columns is stored in a variable called features
test_X = test_data[features]
# test_y = test_data.SalePrice

# An Extension To Imputation----------------------------------------------------------------------------------
# make copy to avoid changing original data (when Imputing)
imputed_test_X = test_X.copy()

# make new columns indicating what will be imputed
# cols_with_missing = (col for col in test_X.columns 
#                                  if test_X[col].isnull().any())
# for col in cols_with_missing:
#     imputed_test_X[col + '_was_missing'] = imputed_test_X[col].isnull()

# Imputation
my_imputer = SimpleImputer()
imputed_test_X = my_imputer.fit_transform(imputed_test_X)
# ----------------------------------------------------------------------------------An Extension To Imputation

# make predictions which we will submit. 
test_preds = rf_model_on_full_data.predict(imputed_test_X)
# val_rf_full_data_mae = mean_absolute_error(test_preds, test_y)

# The lines below shows how to save predictions in format used for competition scoring
# Just uncomment them.

output = pd.DataFrame({'Id': test_data.Id,
                      'SalePrice': test_preds})
output.to_csv('./input/submission.csv', index=False)