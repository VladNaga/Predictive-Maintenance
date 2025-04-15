# Predictive Maintenance Model Analysis Report
Generated on: 2025-04-15 15:46:09

## Model Information
- Model Type: RandomForestClassifier
- Number of Features: 11
- Best Parameters: {
  "bootstrap": true,
  "ccp_alpha": 0.0,
  "class_weight": null,
  "criterion": "gini",
  "max_depth": null,
  "max_features": "sqrt",
  "max_leaf_nodes": null,
  "max_samples": null,
  "min_impurity_decrease": 0.0,
  "min_samples_leaf": 1,
  "min_samples_split": 2,
  "min_weight_fraction_leaf": 0.0,
  "monotonic_cst": null,
  "n_estimators": 200,
  "n_jobs": null,
  "oob_score": false,
  "random_state": 42,
  "verbose": 0,
  "warm_start": false
}

## Performance Metrics
- Accuracy: 0.991
- Precision: 0.957
- Recall: 0.721
- F1 Score: 0.822

## Feature Importance
Top 5 most important features:
- Power: 0.1660
- Speed_Torque_Interaction: 0.1524
- Rotational_speed__rpm_: 0.1390
- Temperature_Difference: 0.1376
- Torque__Nm_: 0.1157

## Class Distribution
### Test Set
- Class 0 (No Failure): 1939
- Class 1 (Failure): 61

### Predictions
- Class 0 (No Failure): 1954
- Class 1 (Failure): 46

## Insights
### Model Reliability
The model shows excellent performance with:
- High accuracy (0.991) indicating overall correct predictions
- Good precision (0.957) for failure detection
- Moderate recall (0.721) suggesting some missed failures
- Strong F1 score (0.822) showing good balance between precision and recall

### Feature Analysis
The most influential features for predicting machine failure are:
- Power: 0.1660
- Speed_Torque_Interaction: 0.1524
- Rotational_speed__rpm_: 0.1390
- Temperature_Difference: 0.1376
- Torque__Nm_: 0.1157

### System Architecture
The model architecture is optimized with:
- 200 decision trees
- Maximum depth: Unlimited
- Minimum samples per split: 2

## Visualizations
The following visualizations are available in the analysis_output directory:
1. confusion_matrix.png - Model's prediction performance
2. roc_curve.png - Receiver Operating Characteristic curve
3. feature_importance.png - Feature importance ranking
4. learning_curve.png - Model's learning progress
5. precision_recall_curve.png - Precision-Recall tradeoff
6. feature_correlations.png - Feature correlation heatmap
7. feature_distributions.png - Distribution of important features
8. error_analysis.png - Analysis of misclassified samples
