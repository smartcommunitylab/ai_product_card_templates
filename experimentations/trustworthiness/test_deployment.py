import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from fairness.holisticAI.src.data_preparation import *

import requests
import json

def test_model_evaluation():
    model_name = "RidgeClassifier"
    model_version = "1"

    # Load the model from the Model Registry
    model_uri = f"models:/{model_name}/{model_version}"
    abs_path = "/home/albana/Desktop/Albana/DataScience/AI4DT/Projects/ai_product_card_templates_july/experimentations/trustworthiness/mlruns/0/models/m-6537703297e8417288fe7fff724f04bf/artifacts"
    #model = mlflow.sklearn.load_model(model_uri)  
    model = mlflow.sklearn.load_model(abs_path)    
    
    # prepare a validation dataset for prediction and predict
    data = pd.read_parquet("data.parquet")
    data_train, data_test = train_test_split(data, test_size=0.3, random_state=4)
    X_test, y_test, dem_test = split_data_from_df(data_test)
    y_pred_new = model.predict(X_test)
    print(y_pred_new)
    
    
def test_inference_mlflow_mlserver():
    url = f"http://localhost:1234/invocations"
    
    # prepare a validation dataset for prediction and predict
    data = pd.read_parquet("data.parquet")
    data_train, data_test = train_test_split(data, test_size=0.3, random_state=4)
    X_test, y_test, dem_test = split_data_from_df(data_test)

    payload = json.dumps(
        {
            "inputs": X_test.tolist(),
        }
    )
    response = requests.post(
        url=url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    print(response.json())
    
    
def test_inference_kserve():
    url = f"http://localhost:8080/v2/models/sklearn-hiring/infer"
    
    # prepare a validation dataset for prediction and predict
    data = pd.read_parquet("data.parquet")
    data_train, data_test = train_test_split(data, test_size=0.3, random_state=4)
    X_test, y_test, dem_test = split_data_from_df(data_test)

    
    payload = json.dumps(
        {
        "inputs": [
            {
            "name": "input",
            "shape": X_test.shape,
            "datatype": "FP32",
            "data": X_test.tolist()
            }
        ]}
    )
    response = requests.post(
        url=url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    print(response)
        

if __name__ == "__main__":
    test_inference_kserve()