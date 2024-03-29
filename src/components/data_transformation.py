import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import os
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            features=["GDP per capita","Social support","Healthy life expectancy","Freedom to make life choices","Generosity","Perceptions of corruption"]

            num_pipeline= Pipeline(
                steps=[
                
                ("scaler",StandardScaler())

                ]
            )
            
            logging.info("In data transformer object")

            preprocessing = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,features)
                ]
            )
            
            return preprocessing

        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,X_train_path,X_test_path,y_train_path,y_test_path):

        try:
            X_train=pd.read_csv(X_train_path)
            X_test=pd.read_csv(X_test_path)
            y_train=pd.read_csv(y_train_path)
            y_test=pd.read_csv(y_test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            '''target_column_name="Score"
            numerical_columns = ["GDP per capita","Social support","Healthy life expectancy","Freedom to make life choices","Generosity","Perceptions of corruption"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]'''

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            X_train=preprocessing_obj.fit_transform(X_train)
            X_test=preprocessing_obj.transform(X_test)

            print(X_train)

            y_train = np.array(y_train).ravel()
            y_test = np.array(y_test).ravel()
            

            '''train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]'''

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                X_train,X_test,y_train,y_test,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
        