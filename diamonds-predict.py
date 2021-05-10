import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import scipy.sparse as sparse
import joblib

# Define value mapping
mapping = {'Color': {'D': 1, 'E': 2, 'F': 3, 'G': 4, 'H': 5, 'I': 6, 'J': 7},
           'Clarity': {'IF': 0, 'VVS1': 1, 'VVS2': 2, 'VS1': 3, 'VS2': 4, 'SI1': 5, 'SI2': 6, 'I1': 7, 'I2': 8}
           }

# "Live" predictions

# Define the diamond features

# Values for testing
# center_carat = 1.2
# center_color = 'G'
# center_clarity = 'VS1'

# side_carat = 0.5
# side_color = 'H'
# side_clarity = 'VS2'

# Setting = 2500

print(f'\nCenter Stone')
center_carat = float(input('Enter carat weight:'))
center_color = input('Enter color:')
center_clarity = input('Enter clarity:')

print(f'\nSide Stones')
side_carat = float(input('Enter carat weight:'))
side_color = input('Enter color:')
side_clarity = input('Enter clarity:')

print(f'\nSetting')
Setting = int(input('Enter setting cost:'))

# Create the data frame from the above variables

d_center = {'Carat': [center_carat],
            'Color': [center_color],
            'Clarity': [center_clarity],
            'Position': 'Center'
            }
d_side = {'Carat': [side_carat],
          'Color': [side_color],
          'Clarity': [side_clarity],
          'Position': 'Side'
          }
df_center = pd.DataFrame(data=d_center)
df_side = pd.DataFrame(data=d_side)
frames = [df_center, df_side]
diamond = pd.concat(frames)
diamond = diamond.reset_index()

# Map the features
test_data = diamond.replace(mapping)
test_data = test_data.drop(['Position', 'index'], axis=1)
test_data

# Import Model from File
rf = joblib.load('./random_forest.joblib')

# Predict
prediction = DataFrame(rf.predict(test_data))
prediction

# Output
diamond['Prediction'] = prediction[0]
diamond['total_cost'] = np.where(
    diamond['Position'] == 'Side', diamond['Prediction']*2, diamond['Prediction'])

total_cost = diamond['total_cost'].sum()

print(f'\nEstimated Costs:')
print('Cost of stones: $', round(total_cost, 2))
print('Total Cost: $', round(total_cost+Setting, 2))
