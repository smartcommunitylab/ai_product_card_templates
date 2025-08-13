from kserve import Model, KFServer
import numpy as np
import joblib

from sklearn.metrics import accuracy_score, confusion_matrix

import holisticai
from holisticai.bias.metrics import disparate_impact,statistical_parity, average_odds_diff
from holisticai.bias.mitigation import EqualizedOdds


class CustomModel(Model):
    def __init__(self, name: str, model_dir: str):
        super().__init__(name)
        self.model_dir = model_dir
        self.model = None

    def load(self):
        # Load the model from the specified directory
        model_path = f"{self.model_dir}/model.joblib"
        self.model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
        return self

    def predict(self, inputs: dict) -> dict:
        # Perform inference
        data = np.array(inputs["instances"])
        predictions = self.model.predict(data)
        return {"predictions": predictions.tolist()}


def model_deployment():
    model_name = "custom-model"
    model_dir = "/mnt/models"

    # Create and start the KFServer
    model = CustomModel(model_name, model_dir)
    KFServer(workers=1).start([model])
    


# Function to calculate and return model accuracy and fairness metrics for two groups
def get_metrics(group_a, group_b, y_pred, y_true):
    """
    Returns a DataFrame of model accuracy and fairness metrics for two groups.
    """
    metrics = [['Model Accuracy', round(accuracy_score(y_true, y_pred), 2), 1]]  # Calculate accuracy
    metrics += [['Black vs. White Disparate Impact', round(disparate_impact(group_a, group_b, y_pred), 2), 1]]  # Calculate disparate impact
    metrics += [['Black vs. White Statistical Parity', round(statistical_parity(group_a, group_b, y_pred), 2), 0]]  # Calculate statistical parity
    metrics += [['Black vs. White Average Odds Difference', round(average_odds_diff(group_a, group_b, y_pred, y_true), 2), 0]]  # Calculate average odds difference
    return pd.DataFrame(metrics, columns=['Metric', 'Value', 'Reference'])  # Return metrics as DataFrame


def post_processing_fairness(prediction):
    """
    Equalized Odds is a post-processing fairness technique that adjusts the model's predictions 
    after training to ensure fairness across different groups. 
    It aims to equalize the true positive rate (TPR) and false positive rate (FPR) between groups. 
    By modifying the decision thresholds for different groups, Equalized Odds ensures that 
    the model’s performance is equally favorable for all groups, 
    regardless of their demographic characteristics. 
    This method helps to correct any disparities in prediction outcomes that may have arisen 
    during the training phase.
    """
    ############################ Training a model and then treating it like a 'black-box'

    # Split Data into Training and Testing Sets
    data_train, data_test = train_test_split(data, test_size=0.3, random_state=4)  # Split into train/test sets
    X_train, y_train, dem_train = split_data_from_df(data_train)  # Extract features, labels, and demographics for the training set
    X_test, y_test, dem_test = split_data_from_df(data_test)  # Extract features, labels, and demographics for the test set

    # Initialize and Train Model
    model = RidgeClassifier(random_state=42)  # Initialize the RidgeClassifier model with a fixed random seed
    model.fit(X_train, y_train)  # Train the model using the training features and labels
    y_pred_test = model.predict(X_test)
    
    group_a_test = (dem_test['Ethnicity'] == 'Black')
    group_b_test = (dem_test['Ethnicity'] == 'White')
    metrics_orig = get_metrics(group_a_test, group_b_test, y_pred_test, y_test)
    
    ############################ Post-Processing Fairness with Equalized Odds
    
    # Split Testing set to have a post-processor 'Training'
    data_pp_train, data_pp_test = train_test_split(data_test,test_size = 0.4,random_state=42)
    X_pp_train,y_pp_train,dem_pp_train = split_data_from_df(data_pp_train)
    X_pp_test,y_pp_test,dem_pp_test = split_data_from_df(data_pp_test)

    group_a_pp_train = (dem_pp_train['Ethnicity']=='Black')
    group_b_pp_train = (dem_pp_train['Ethnicity']=='White')
    group_a_pp_test = (dem_pp_test['Ethnicity']=='Black')
    group_b_pp_test = (dem_pp_test['Ethnicity']=='White')

    # Fit processor on the 'training' data
    eq = EqualizedOdds(solver='highs', seed=42)
    fit_params = {
        "group_a": group_a_pp_train,
        "group_b": group_b_pp_train,
    }
    y_pred_pp_train = model.predict(X_pp_train)

    eq.fit(y_pp_train, y_pred_pp_train, **fit_params)

    # Apply Processor to Predictions from 'Test' Data
    fit_params = {
        "group_a": group_a_pp_test,  # Define the first group (e.g., 'Black' candidates)
        "group_b": group_b_pp_test,  # Define the second group (e.g., 'White' candidates)
    }

    y_pred_pp_test = model.predict(X_pp_test)  # Predict the labels for the test set
    d = eq.transform(y_pred_pp_test, **fit_params)  # Apply equalized odds processor to the predictions

    # Extract the new predictions after applying the fairness processor
    y_pred_pp_new = d['y_pred']
    
    # Evaluate and Plot Metrics
    metrics_eq = get_metrics(group_a_pp_test, group_b_pp_test, y_pred_pp_new, y_pp_test)  # Get fairness metrics
    metrics_orig['mitigation'] = 'None'
    metrics_eq['mitigation'] = 'Equalized Odds'
    metrics = pd.concat([metrics_orig, metrics_eq], axis=0, ignore_index=True)
    return metrics


def visualise_metrics(metrics):    
    display(metrics_eq)  # Display the fairness metrics
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=metrics, x='Metric', y='Value', hue='mitigation')
    plt.axhline(y=0.8, linewidth=2, color='r', linestyle="--")
    plt.axhline(y=-0.05, linewidth=2, color='r', linestyle="--")
    plt.axhline(y=1, linewidth=2, color='g')
    plt.axhline(y=0, linewidth=2, color='g')
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.show()
    
################################################ Model monitoring
# experiments tracking: efficient 

def model_monitoring():
    