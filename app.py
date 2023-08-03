import pandas as pd
import numpy as np
from flask import Flask, request, render_template
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
import joblib

application = Flask(__name__)

model = joblib.load('artifacts/model.pkl')

app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        gdp = float(request.form.get('GDP'))
        social_support = float(request.form.get('SocialSupport'))
        healthy_life = float(request.form.get('HealthyLife'))
        freedom = float(request.form.get('Freedom'))
        generosity = float(request.form.get('Generosity'))
        corruption = float(request.form.get('Corruption'))

        '''data = CustomData(
            GDP=gdp,
            SocailSupport=social_support,
            HealthyLife=healthy_life,
            Freedom=freedom,
            Generosity=generosity,
            Corruption=corruption
        )
        pred_df = data.get_data_as_data_frame()

        predict_pipeline = PredictPipeline()'''

        list = np.array([gdp,social_support,healthy_life,freedom,generosity,corruption])
        list= list.reshape(1,-1)
        results = model.predict(list)

        return render_template('home.html', results=results)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
