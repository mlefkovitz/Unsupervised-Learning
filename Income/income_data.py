import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit

income_df = pd.read_csv("C:/Users/Myles/Documents/OMSCS/CS7641 ML/Assignment 1/Supervised-Learning//gth836x/Income/train.csv")

# Drop non-predictive factors:
#   fnlwgt doesn't mean anything
#   education is represented by the education-num factor
#   native-country is 90% USA
income_df = income_df.drop(['fnlwgt', 'education', 'native-country'], axis=1)

# Convert "workclass" factor into numbers
income_df_workclass = pd.get_dummies(income_df['workclass'], prefix='workclass')
income_df = pd.concat([income_df, income_df_workclass], axis=1)
income_df = income_df.drop(['workclass'], axis=1)

# Convert "marital-status" factor into numbers
income_df_maritalstatus = pd.get_dummies(income_df['marital-status'], prefix='maritalstatus')
income_df = pd.concat([income_df, income_df_maritalstatus], axis=1)
income_df = income_df.drop(['marital-status'], axis=1)

# Convert "occupation" factor into numbers
income_df_occupation = pd.get_dummies(income_df['occupation'], prefix='occupation')
income_df = pd.concat([income_df, income_df_occupation], axis=1)
income_df = income_df.drop(['occupation'], axis=1)

# Convert "relationship" factor into numbers
income_df_relationship = pd.get_dummies(income_df['relationship'], prefix='relationship')
income_df = pd.concat([income_df, income_df_relationship], axis=1)
income_df = income_df.drop(['relationship'], axis=1)

# Convert "race" factor into numbers
income_df_race = pd.get_dummies(income_df['race'], prefix='race')
income_df = pd.concat([income_df, income_df_race], axis=1)
income_df = income_df.drop(['race'], axis=1)

# Convert "sex" factor into numbers
income_df['sex'].replace(' Female', 1, inplace=True)
income_df['sex'].replace(' Male', 0, inplace=True)

# Convert "Income" factor into numbers
income_df['Income'].replace(' >50K', 1, inplace=True)
income_df['Income'].replace(' <=50K', 0, inplace=True)

# we are classifying on all features
y = income_df['Income'].values
X = income_df[['age', 'education-num', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
              'workclass_ ?', 'workclass_ Federal-gov', 'workclass_ Local-gov',
              'workclass_ Private', 'workclass_ Self-emp-inc',
              'workclass_ Self-emp-not-inc', 'workclass_ State-gov',
              'maritalstatus_ Divorced', 'maritalstatus_ Married-AF-spouse', 'maritalstatus_ Married-civ-spouse',
              'maritalstatus_ Married-spouse-absent', 'maritalstatus_ Never-married', 'maritalstatus_ Separated',
              'maritalstatus_ Widowed', 'occupation_ ?', 'occupation_ Adm-clerical', 'occupation_ Armed-Forces',
              'occupation_ Craft-repair', 'occupation_ Exec-managerial', 'occupation_ Farming-fishing',
              'occupation_ Handlers-cleaners', 'occupation_ Machine-op-inspct', 'occupation_ Other-service',
              'occupation_ Priv-house-serv', 'occupation_ Prof-specialty', 'occupation_ Protective-serv',
              'occupation_ Sales', 'occupation_ Tech-support', 'occupation_ Transport-moving',
              'relationship_ Husband', 'relationship_ Not-in-family', 'relationship_ Other-relative',
              'relationship_ Own-child', 'relationship_ Unmarried', 'relationship_ Wife',
              'race_ Amer-Indian-Eskimo', 'race_ Asian-Pac-Islander', 'race_ Black',
              'race_ Other', 'race_ White'
              ]].values

#print(X.shape)
#print(X)

#print(y.shape)
#print(y)


le = LabelEncoder()

# encode classifications as 1 and 0 rather than 2 and 3 (original state)
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)