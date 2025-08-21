import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from fairness.holisticAI.src.data_preparation import *


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
    
    
    
if __name__ == "__main__":
    test_model_evaluation()