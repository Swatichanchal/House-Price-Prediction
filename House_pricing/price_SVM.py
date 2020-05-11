# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the dataset
data = pd.read_csv(r'C:\Users\Swati\Downloads\data.csv')
'''
X = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values
'''
X = data.iloc[:, 1:-1]
y = data.iloc[:, -1]
print(X.shape)

# REMOVING A ROW FROM TRAINING SET 
dataset.columns.get_loc('Condition1')
dataset.columns.get_loc('GarageQual')
dataset.columns.get_loc('Street')
dataset.columns.get_loc('YearBuilt')
'''
X = np.delete(X,13 , 1 )
X = np.delete(X,63 , 1 )
X = np.delete(X,5 , 1 )
X = np.delete(X,19 , 1 )
# DELETE IS NUMPY FUNCTION 
'''
X = X.drop(['Condition1','Street','YearBuilt'] , axis=1)
print(X.shape)
print('The X set shape is : %s' %str(X.shape))
print('The y set shape is : %s' %str(y.shape))
print('The data set shape is : %s' %str(data.shape))

'''
# Determines all the unique element in the array
np.unique(X[: , 2])

# Determine the LENGTH of the unique element present in the array 
len(np.unique(X[: , 17]))     
X = np.delete(X, 17 , 1 )
'''
'''
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()

X[:, 2] = labelencoder.fit_transform(X[:, 2])
X[:, 2].reshape(-1, 1)
onehotencoder = OneHotEncoder(categorical_features = [147])
X = onehotencoder.fit_transform(X[:, 2]).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]
'''

'''
# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
'''

# Imputing the missing values
missing =X.isna().sum(axis=0).sort_values(ascending=False)
missing_value_columns = missing[missing>0]
print('They are %s columns with missing values : \n%s ' %(missing_value_columns.count() , [(index , value) for (index , value) in missing_value_columns.iteritems()]))

def impute_value(X):
    dataset =X
    dataset['PoolQC'].fillna('NA' , inplace = True)
    dataset['MiscFeature'].fillna('NA' , inplace = True)
    dataset['Alley'].fillna('NA' , inplace = True)
    dataset['Fence'].fillna('NA' , inplace = True)
    dataset['FireplaceQu'].fillna('NA' , inplace = True)
    dataset['LotFrontage'].fillna(dataset['LotFrontage'].median() , inplace = True)
    dataset['GarageCond'].fillna('NA' , inplace = True)
    dataset['GarageType'].fillna('NA' , inplace = True)
    dataset['GarageYrBlt'].fillna(dataset['GarageYrBlt'].median() , inplace = True)
    dataset['GarageFinish'].fillna('NA' , inplace = True)
    dataset['GarageQual'].fillna('NA' , inplace = True)
    dataset['BsmtExposure'].fillna('NA' , inplace = True)
    dataset['BsmtFinType2'].fillna('NA' , inplace = True)
    dataset['BsmtFinType1'].fillna('NA' , inplace = True)
    dataset['BsmtCond'].fillna('NA' , inplace = True)
    dataset['BsmtQual'].fillna('NA' , inplace = True)
    dataset['MasVnrArea'].fillna(dataset['MasVnrArea'].median() , inplace = True)
    dataset['MasVnrType'].fillna('None' , inplace = True)
    dataset['Electrical'].fillna('SBrkr' , inplace = True)
    return dataset

X = impute_value(X)

# Cross checking the missing value
missing =X.isna().sum(axis=0).sort_values(ascending=False)
missing_value_columns = missing[missing>0]
print('They are %s columns with missing values : \n%s ' %(missing_value_columns.count() , [(index , value) for (index , value) in missing_value_columns.iteritems()]))

# to check the missing value is present or not
print(X.isnull())
print(X.isnull().any())
print(X.isnull().any().any())
# no. of missing values present 
print(X.isnull().any().sum())
print(X.isnull().sum())

# to check the numeric variables
numeric = list(X.select_dtypes(include=[np.number]))
numeric.remove('MSSubClass')
print('Here are the %s numeric variables : \n %s' %(len(numeric) , numeric))

# to check the categorical variables
object = list(X.select_dtypes(include=[np.object]))
print('Here are the %s object variables : \n %s' %(len(object) , object)) 

# Categorical variables
def transform_variable(dataset):
    dataset = X
    copy = dataset
    # Important categorical variables
    copy.replace({'MSSubClass' : {
        20	:'1-STORY 1946 & NEWER ALL STYLES' ,
            30	:'1-STORY 1945 & OLDER' ,
            40	 :    '1-STORY W/FINISHED ATTIC ALL AGES',
            45	:'1-1/2 STORY - UNFINISHED ALL AGES' ,
            50	:'1-1/2 STORY FINISHED ALL AGES' ,
            60	:'2-STORY 1946 & NEWER' ,
            70	 : '2-STORY 1945 & OLDER' ,
            75	:'2-1/2 STORY ALL AGES' ,
            80	:'SPLIT OR MULTI-LEVEL' ,
            85	:'SPLIT FOYER',
            90	:'DUPLEX - ALL STYLES AND AGES',
           120	:'1-STORY PUD (Planned Unit Development) - 1946 & NEWER',
           150	:'1-1/2 STORY PUD - ALL AGES',
           160	:'2-STORY PUD - 1946 & NEWER',
           180	:'PUD - MULTILEVEL - INCL SPLIT LEV/FOYER',
           190	:'2 FAMILY CONVERSION - ALL STYLES AND AGES',
    }} , inplace = True)
    
    # one hot encoding
    columns = ['MSZoning','MSSubClass', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']
    for col_name in columns:
             copy = pd.concat([copy , pd.get_dummies(copy[col_name] , prefix = col_name)] , axis =1)
        copy = copy.drop(col_name , axis=1)
    '''  
    # ordinal variables transformation
    quality = {'Ex' : 5 , 'Gd' : 4 , 'TA' : 3 , 'Po' : 1 , 'NA' : 0}
    basement = { 'NA' : 0 , 'Unf' : 1 , 'LwQ' : 2 , 'Rec' : 3, 'BLQ' : 4 , 'ALQ' : 5 , 'GLQ' : 6 }
    ordinal = {
        'HeatingQC': quality,
        'ExterQual': quality,
        'ExterCond': quality,
        'BsmtQual': quality,
        'BsmtCond': quality,
        'KitchenQual': quality,
        'FireplaceQu': quality,
        'GarageQual': quality,
        'GarageCond': quality,
        'PoolQC': quality,
        'BsmtFinType1': basement,
        'BsmtFinType2': basement,
        'LandSlope':{'Gtl': 0 , 'Mod' : 1, 'Sev' :2},
        'MasVnrType':{'None' : 0 , 'BrkCmn' : 0 , 'BrkFace' : 1, 'Stone' : 2},
        'BsmtExposure': {'Gd' : 4 , 'Av' : 3 , 'Mn' : 2 , 'No' : 1, 'NA' : 0},
        'CentralAir': { 'N' : 0, 'Y' : 1},
        'GarageFinish':{'NA' : 0 , 'Unf' : 1, 'Rfn' : 2 , 'Fin' : 3},
        'PavedDrive':{ 'N' : 0,'P' : 1, 'Y' : 2}
    }
    
    for col_name , matching_map in ordinal.items():
            copy[col_name] = copy[col_name].replace(matching_map , inplace=True)
    '''
       
    return copy

X = transform_variable(X)
print(X.shape)

'''
result =pd.DataFrame()
result['HeatingQC'] = X['HeatingQC'].replace({'Ex' : 5 , 'Gd' : 4 , 'TA' : 3 , 'Po' : 1 , 'NA' : 0})
result.head(8)
result['ExterQual'] = X['ExterQual'].replace({'Ex' : 5 , 'Gd' : 4 , 'TA' : 3 , 'Po' : 1 , 'NA' : 0})
result['ExterCond'] = X['ExterCond'].replace({'Ex' : 5 , 'Gd' : 4 , 'TA' : 3 , 'Po' : 1 , 'NA' : 0})
result['BsmtQual'] = X['BsmtQual'].replace({'Ex' : 5 , 'Gd' : 4 , 'TA' : 3 , 'Po' : 1 , 'NA' : 0})
result['BsmtCond'] = X['BsmtCond'].replace({'Ex' : 5 , 'Gd' : 4 , 'TA' : 3 , 'Po' : 1 , 'NA' : 0})
result[ 'KitchenQual'] = X[ 'KitchenQual'].replace({'Ex' : 5 , 'Gd' : 4 , 'TA' : 3 , 'Po' : 1 , 'NA' : 0})
result['FireplaceQu'] = X['FireplaceQu'].replace({'Ex' : 5 , 'Gd' : 4 , 'TA' : 3 , 'Po' : 1 , 'NA' : 0})
result[ 'GarageQual'] = X[ 'GarageQual'].replace({'Ex' : 5 , 'Gd' : 4 , 'TA' : 3 , 'Po' : 1 , 'NA' : 0})
result['GarageCond'] = X['GarageCond'].replace({'Ex' : 5 , 'Gd' : 4 , 'TA' : 3 , 'Po' : 1 , 'NA' : 0})
result['PoolQC'] = X['PoolQC'].replace({'Ex' : 5 , 'Gd' : 4 , 'TA' : 3 , 'Po' : 1 , 'NA' : 0})
result['BsmtFinType1'] = X['BsmtFinType1'].replace({ 'NA' : 0 , 'Unf' : 1 , 'LwQ' : 2 , 'Rec' : 3, 'BLQ' : 4 , 'ALQ' : 5 , 'GLQ' : 6 })
result['BsmtFinType2'] = X['BsmtFinType2'].replace({ 'NA' : 0 , 'Unf' : 1 , 'LwQ' : 2 , 'Rec' : 3, 'BLQ' : 4 , 'ALQ' : 5 , 'GLQ' : 6 })
result['LandSlope'] = X['LandSlope'].replace({'Gtl': 0 , 'Mod' : 1, 'Sev' :2})
result['MasVnrType'] = X['MasVnrType'].replace({'None' : 0 , 'BrkCmn' : 0 , 'BrkFace' : 1, 'Stone' : 2})
result['BsmtExposure'] = X['BsmtExposure'].replace({'Gd' : 4 , 'Av' : 3 , 'Mn' : 2 , 'No' : 1, 'NA' : 0})
result['CentralAir'] = X['CentralAir'].replace({ 'N' : 0, 'Y' : 1})
result['GarageFinish'] = X['GarageFinish'].replace({'NA' : 0 , 'Unf' : 1, 'Rfn' : 2 , 'Fin' : 3})
result['PavedDrive'] = X['PavedDrive'].replace({ 'N' : 0,'P' : 1, 'Y' : 2})
X['ExterQual'] = result['ExterQual']
X['ExterCond'] = result['ExterCond'] 
X['BsmtQual']=result['BsmtQual']
X['BsmtCond']=result['BsmtCond']
X[ 'KitchenQual']=result[ 'KitchenQual']
X['FireplaceQu']=result['FireplaceQu']
X[ 'GarageQual']=result[ 'GarageQual']
X['GarageCond']=result['GarageCond']
X['PoolQC']=result['PoolQC']
X['BsmtFinType1']=result['BsmtFinType1']
X['BsmtFinType2']=result['BsmtFinType2']
X['LandSlope']=result['LandSlope'] 
X['MasVnrType']=result['MasVnrType']
X['BsmtExposure']=result['BsmtExposure']
X['CentralAir']=result['CentralAir']
X['GarageFinish']=result['GarageFinish']
X['PavedDrive']=result['PavedDrive']
'''

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = True)

X_train =X_train.values
# print(X_train.dtypes) :: to checkkk if the X_train is converted into numpy array or not [if error occur it is , if not it's not]
X_test =X_test.values
y_train=y_train.values
y_test=y_test.values
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = y_train.reshape(y_train.shape[0] , 1)
y_test = y_test.reshape(y_test.shape[0] , 1)
y_train = sc_y.fit_transform(y_train)
y_test = sc_y.transform(y_test)

# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf') 
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

y_pred
print(y_pred.shape)
y_test
print(y_test.shape)

# to check the accuracy of the model 
# TRAINING SET
print(regressor.score(X_train, y_train))       # 91.873884
# TEST SET 
print(regressor.score(X_test, y_test))         # 68.3784539

