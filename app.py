from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import pickle
from pandas.core.algorithms import take

app = Flask(__name__)

# Load pkl model
model = pickle.load(open('random_forest.pkl', 'rb'))


@app.route('/')
def Home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    f = request.form.to_dict()
    df = pd.DataFrame(list(f.items()), columns=['d_item', 'value'])

    df_split = df.d_item.str.split('_', expand=True)
    df = df.join(df_split)
    df = df.drop(['d_item'], axis=1)
    settingdf = df.loc[df[0] == 'setting']
    setting = float(settingdf.iloc[0]['value'])
    df = df[df[0] != 'setting']

    df = df.pivot(index=0, columns=1, values='value')
    df = df.reset_index()
    df = df.rename(columns={0: 'position'})

    mapping = {'color': {'D': 1, 'E': 2, 'F': 3, 'G': 4, 'H': 5, 'I': 6, 'J': 7},
               'clarity': {'IF': 0, 'VVS1': 1, 'VVS2': 2, 'VS1': 3, 'VS2': 4, 'SI1': 5, 'SI2': 6, 'I1': 7, 'I2': 8}
               }
    test_data = df.replace(mapping)
    test_data = test_data.drop(['position'], axis=1)

    # Modelling

    prediction = model.predict(test_data)

    center_prediction = float(prediction[0])
    side_prediction = float(prediction[1])*2
    total_cost = (center_prediction, side_prediction, setting)
    total_cost = sum(total_cost)

    return render_template(
        'index.html', prediction_text=total_cost
    )


if __name__ == '__main__':
    app.run(debug=True)
