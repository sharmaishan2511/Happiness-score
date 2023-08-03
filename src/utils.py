import os
import sys

import numpy as np 
import pandas as pd
import dill
from sklearn.metrics import r2_score

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_models(X_train, y_train,X_test,y_test,models):
    try:
        report = 2
        modelname = "none"

        for i in range(len(list(models))):
            model = list(models.values())[i]
            
            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            if((train_model_score>0.6 and test_model_score>0.6) and abs(train_model_score-test_model_score)<report):
                report= abs(train_model_score-test_model_score)
                modelname = list(models.keys())[i]

            #report[list(models.keys())[i]] = [train_model_score,test_model_score]

        return modelname,report

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)