
def get_metrics(group_a, group_b, y_pred, y_true):
    """
    Function to calculate and return model accuracy and fairness metrics for two groups
    Returns a DataFrame of model accuracy and fairness metrics for two groups.
    """
    metrics = [['Model Accuracy', round(accuracy_score(y_true, y_pred), 2), 1]]  # Calculate accuracy
    metrics += [['Black vs. White Disparate Impact', round(disparate_impact(group_a, group_b, y_pred), 2), 1]]  # Calculate disparate impact
    metrics += [['Black vs. White Statistical Parity', round(statistical_parity(group_a, group_b, y_pred), 2), 0]]  # Calculate statistical parity
    metrics += [['Black vs. White Average Odds Difference', round(average_odds_diff(group_a, group_b, y_pred, y_true), 2), 0]]  # Calculate average odds difference
    return 
pd.DataFrame(metrics, columns=['Metric', 'Value', 'Reference'])  # Return metrics as DataFrame
