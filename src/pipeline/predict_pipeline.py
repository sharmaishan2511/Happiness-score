import sys
import os
import pandas as pd

from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model = load_object(file_path=model_path)
            preds = model.predict(features)
            return preds
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 GDP: float,
                 SocailSupport: float,
                 HealthyLife: float,
                 Freedom: float,
                 Generosity: float,
                 Corruption:float):
        
        self.GDP=GDP
        self.SocialSupport = SocailSupport
        self.HealthyLife = HealthyLife
        self.Freedom = Freedom
        self.Generosity = Generosity
        self.Corruption = Corruption
    
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "GDP": [self.GDP],
                "SocialSupport": [self.SocialSupport],
                "HealthyLife": [self.HealthyLife],
                "Freedom": [self.Freedom],
                "Generosity": [self.Generosity],
                "Corruption": [self.Corruption],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)



        
