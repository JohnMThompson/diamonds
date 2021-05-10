import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Import csv with known diamond inventory and prices
features = pd.read_csv('diamonds.csv')

# Remove unwanted columns
features = features.drop(
    ['Stock Number', 'RecordID', 'Lab', 'Store'], axis=1)

# Create a mapping table for our features to convert from strings to int
# I would like to store this somewhere portable. Right now it's duplicated in the modelling and the predict files
mapping = {'Color': {'D': 1, 'E': 2, 'F': 3, 'G': 4, 'H': 5, 'I': 6, 'J': 7},
           'Clarity': {'IF': 0, 'VVS1': 1, 'VVS2': 2, 'VS1': 3, 'VS2': 4, 'SI1': 5, 'SI2': 6, 'I1': 7, 'I2': 8}
           }

# Replace the values with the mappings
features = features.replace(mapping)

# Cut out our target variable
labels = np.array(features['Price'])

# Creat a data frame with only the predictor features
features = features.drop(['Price'], axis=1)
feature_list = list(features.columns)
features = np.array((features))

# Create training and test sets - I need to work on this some more,
# I skipped the test phase a bit since I already had results from Alteryx
train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.1, random_state=42)

# Create regression and train
rf = RandomForestRegressor(n_estimators=1000, random_state=42)
rf.fit(train_features, train_labels)

# Save model
joblib.dump(rf, "./random_forest.joblib")
