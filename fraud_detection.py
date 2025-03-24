# Import necessary libraries for data manipulation, modeling, visualization, etc.
import pandas as pd               # For handling dataframes and data manipulation
import numpy as np                # For numerical operations
import xgboost as xgb             # XGBoost library for gradient boosting models
import lightgbm as lgb            # LightGBM library for gradient boosting models
import joblib                    # For saving/loading Python objects
import matplotlib.pyplot as plt   # For plotting graphs and figures
import seaborn as sns             # For enhanced data visualizations
import json                      # For handling JSON data (saving metadata)
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
                                  # Functions for splitting data and model tuning/validation
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
                                  # For scaling and transforming feature values
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_auc_score, precision_recall_curve, auc,
    classification_report, average_precision_score
)
                                  # Metrics for evaluating model performance
from sklearn.ensemble import VotingClassifier
                                  # For combining multiple models into an ensemble
from sklearn.utils.class_weight import compute_class_weight
                                  # To compute class weights to handle imbalanced datasets
from sklearn.metrics import roc_curve
                                  # For computing ROC curve values
from imblearn.over_sampling import SMOTE, ADASYN
                                  # Over-sampling techniques for imbalanced datasets
from imblearn.under_sampling import RandomUnderSampler
                                  # Under-sampling technique for imbalanced datasets
from imblearn.pipeline import Pipeline as ImbPipeline
                                  # Pipeline utility from imbalanced-learn (allows integration of resampling)
from imblearn.combine import SMOTETomek, SMOTEENN
                                  # Combined over- and under-sampling techniques
import shap                        # For model explainability using SHAP values
import pickle                      # For serializing and saving the model
from datetime import datetime      # For handling date/time operations
import os                          # For interacting with the operating system (e.g., file paths)
import warnings                    # To manage warning messages

# Ignore warning messages for cleaner output
warnings.filterwarnings('ignore')

# Set a constant random seed for reproducibility across experiments
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def load_and_explore_data(filepath):
    """
    Load a CSV file into a DataFrame, explore the data and generate basic plots.
    Parameters:
      filepath (str): Path to the CSV file.
    Returns:
      DataFrame: The loaded dataset.
    """
    print("Loading and exploring data...")
    
    # Try to read the CSV file; if the file is not found, print an error message.
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File {filepath} not found. Please check the path.")
        return None
    
    # Display the shape of the dataset (number of rows and columns)
    print(f"Dataset shape: {df.shape}")
    # Print data types of each column
    print(f"Data types:\n{df.dtypes}")
    # Count and display the total number of missing values in the dataset
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    # Compute and display class distribution (for example, fraud vs. non-fraud)
    class_counts = df["Class"].value_counts()
    print("Class distribution:")
    print(class_counts)
    # Calculate and print the percentage of fraud cases in the dataset
    fraud_percentage = class_counts[1] / len(df) * 100
    print(f"Fraud percentage: {fraud_percentage:.4f}%")
    
    # Create a directory named "plots" if it does not exist, to save visualizations
    os.makedirs("plots", exist_ok=True)
    
    # Visualize class distribution using a count plot
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Class', data=df)
    plt.title('Class Distribution (Fraud vs Non-Fraud)')
    plt.savefig('plots/class_distribution.png')  # Save plot as an image
    plt.close()  # Close the plot to free memory
    
    # If the dataset contains an "Amount" column, visualize its distribution by class
    if "Amount" in df.columns:
        plt.figure(figsize=(12, 6))
        
        # Plot histogram with KDE for the "Amount" feature separated by class (using logarithmic x-scale)
        plt.subplot(1, 2, 1)
        sns.histplot(df[df["Class"]==0]["Amount"], kde=True, stat="density", color="blue", alpha=0.5)
        sns.histplot(df[df["Class"]==1]["Amount"], kde=True, stat="density", color="red", alpha=0.5)
        plt.title('Amount Distribution by Class')
        plt.legend(['Normal', 'Fraud'])
        plt.xlabel('Amount')
        plt.xscale('log')
        
        # Plot a boxplot for the "Amount" feature grouped by class (using logarithmic y-scale)
        plt.subplot(1, 2, 2)
        sns.boxplot(x="Class", y="Amount", data=df)
        plt.title('Amount Boxplot by Class')
        plt.yscale('log')
        
        plt.tight_layout()  # Adjust subplot spacing for a neat layout
        plt.savefig('plots/amount_distribution.png')
        plt.close()
    
    # If the dataset has more than two columns, compute a correlation matrix for a sample of features
    if df.shape[1] > 2:  # Ensuring there are features aside from the target "Class"
        # Select "Class" and the first 10 additional features for the correlation matrix
        corr_cols = ['Class'] + list(df.drop(columns=['Class']).columns[:10])
        corr_matrix = df[corr_cols].corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix of Features with Class')
        plt.tight_layout()
        plt.savefig('plots/correlation_matrix.png')
        plt.close()
    
    return df

def detect_anomalies_in_features(df):
    """
    Detect anomalies (outliers) in features using the IQR (Interquartile Range) method.
    Parameters:
      df (DataFrame): The dataset.
    Returns:
      dict: A report containing features that have significant outlier percentages.
    """
    print("Detecting anomalies in features...")
    
    anomaly_report = {}  # Dictionary to store anomaly statistics per feature
    features = df.drop(columns=["Class"])  # Exclude the target variable from anomaly detection
    
    # Iterate over each feature (column)
    for col in features.columns:
        # Calculate the first (Q1) and third quartiles (Q3)
        Q1 = features[col].quantile(0.25)
        Q3 = features[col].quantile(0.75)
        IQR = Q3 - Q1  # Compute the interquartile range
        
        # Define the lower and upper bounds to identify outliers (1.5 * IQR rule)
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Identify outliers that fall outside the computed bounds
        outliers = features[(features[col] < lower_bound) | (features[col] > upper_bound)]
        outlier_count = len(outliers)
        outlier_percentage = (outlier_count / len(features)) * 100
        
        # If more than 5% of values in the feature are outliers, record this in the report
        if outlier_percentage > 5:
            anomaly_report[col] = {
                'outlier_count': outlier_count,
                'outlier_percentage': outlier_percentage,
                'bounds': (lower_bound, upper_bound)
            }
    
    # Display the results of the anomaly detection
    if anomaly_report:
        print("Features with significant outliers:")
        for col, stats in anomaly_report.items():
            print(f"  - {col}: {stats['outlier_percentage']:.2f}% outliers")
    else:
        print("No significant outliers detected in features.")
    
    return anomaly_report

def preprocess_data(df, anomaly_report=None):
    """
    Preprocess the dataset by applying transformations, handling time and amount features,
    capping outliers, scaling and transforming data.
    Parameters:
      df (DataFrame): The original dataset.
      anomaly_report (dict): Optional report of anomalies to cap outliers.
    Returns:
      tuple: Preprocessed features DataFrame, target series, and a dictionary of preprocessing artifacts.
    """
    print("Preprocessing data...")
    
    # Make a copy of the dataset to avoid altering the original data
    processed_df = df.copy()
    
    # Separate features (X) and target variable (y)
    X = processed_df.drop(columns=["Class"])
    y = processed_df["Class"]
    
    # Copy features to start applying transformations
    preprocessed_features = X.copy()
    
    # Check if a 'Time' column exists; often present in credit card fraud datasets
    if "Time" in X.columns:
        # Convert raw time in seconds to cyclical features (sine and cosine) representing the hour of the day
        preprocessed_features["Hour_sin"] = np.sin(2 * np.pi * X["Time"].apply(lambda x: (x / 3600) % 24) / 24)
        preprocessed_features["Hour_cos"] = np.cos(2 * np.pi * X["Time"].apply(lambda x: (x / 3600) % 24) / 24)
        
        # Also extract the day of the transaction (assuming Time is seconds elapsed since the first transaction)
        preprocessed_features["Day"] = X["Time"] // (24 * 3600)
        
        # Drop the original 'Time' column as its information is now represented in the new features
        preprocessed_features.drop(columns=["Time"], inplace=True)
    
    # If an "Amount" column exists, perform transformations and scaling
    if "Amount" in X.columns:
        # Create a new feature by applying a logarithmic transformation to "Amount" to reduce skewness
        preprocessed_features["LogAmount"] = np.log1p(X["Amount"])
        # Create a binned version of "Amount" using quantile-based discretization
        preprocessed_features["AmountBin"] = pd.qcut(X["Amount"], 10, labels=False, duplicates='drop')
        
        # Apply robust scaling to the original "Amount" to reduce the effect of outliers
        amount_scaler = RobustScaler()
        preprocessed_features["Amount"] = amount_scaler.fit_transform(X[["Amount"]])
    
    # If anomaly report is provided, cap the outliers in each affected feature to the calculated bounds
    if anomaly_report:
        for col, stats in anomaly_report.items():
            if col in preprocessed_features.columns:
                lower_bound, upper_bound = stats['bounds']
                preprocessed_features[col] = preprocessed_features[col].clip(lower_bound, upper_bound)
    
    # Apply the Yeo-Johnson transformation to make feature distributions more normal
    pt = PowerTransformer(method='yeo-johnson')
    # Select numerical columns for transformation
    numerical_cols = preprocessed_features.select_dtypes(include=['float64', 'int64']).columns
    preprocessed_features[numerical_cols] = pt.fit_transform(preprocessed_features[numerical_cols])
    
    # Standardize all features so they have mean 0 and standard deviation 1
    scaler = StandardScaler()
    preprocessed_features = pd.DataFrame(
        scaler.fit_transform(preprocessed_features),
        columns=preprocessed_features.columns
    )
    
    # (Optional) Feature engineering: PCA could be applied for dimensionality reduction if needed.
    
    # Calculate the correlation of each feature with the target variable and store in a dictionary
    correlation_with_target = {}
    for col in preprocessed_features.columns:
        correlation_with_target[col] = np.corrcoef(preprocessed_features[col], y)[0, 1]
    
    # Print the top 10 features most correlated (by absolute value) with the target
    top_corr = sorted(correlation_with_target.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    print("Top correlated features with Class:")
    for feature, corr in top_corr:
        print(f"  - {feature}: {corr:.4f}")
    
    # Save the transformers and scaler used for potential reuse in production
    preprocessing_artifacts = {
        'power_transformer': pt,
        'scaler': scaler,
        'correlation_with_target': correlation_with_target
    }
    
    return preprocessed_features, y, preprocessing_artifacts

def create_train_test_split(X, y):
    """
    Create a stratified split of the dataset into training, validation, and testing sets.
    Parameters:
      X (DataFrame): Features.
      y (Series): Target variable.
    Returns:
      tuple: Split datasets (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    print("Creating train/validation/test split...")
    
    # First split: reserve 20% of the data as the test set using stratification to preserve class ratios
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # Second split: from the remaining data, take 25% as the validation set (which is 20% of original data)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=RANDOM_STATE, stratify=y_temp
    )
    
    # Print the shapes and number of fraud cases in each split
    print(f"Training set shape: {X_train.shape}, Fraud samples: {sum(y_train)}")
    print(f"Validation set shape: {X_val.shape}, Fraud samples: {sum(y_val)}")
    print(f"Testing set shape: {X_test.shape}, Fraud samples: {sum(y_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def get_optimal_class_weights(y_train):
    """
    Compute class weights to handle imbalanced data.
    Parameters:
      y_train (Series): Training target variable.
    Returns:
      dict: Dictionary mapping class labels to their corresponding computed weights.
    """
    classes = np.unique(y_train)  # Get unique classes present in y_train
    # Compute weights using the 'balanced' mode, which inversely scales with class frequencies
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))
    print(f"Class weights: {class_weights}")
    return class_weights

def train_model(X_train, y_train, X_val, y_val):
    """
    Train and tune multiple models using advanced resampling techniques to handle class imbalance.
    This includes tuning hyperparameters using GridSearchCV and combining models in an ensemble.
    Parameters:
      X_train (DataFrame): Training features.
      y_train (Series): Training target.
      X_val (DataFrame): Validation features.
      y_val (Series): Validation target.
    Returns:
      tuple: Best model, its name, and a dictionary containing results from different models.
    """
    print("Training models with advanced techniques...")
    
    # Compute class weights based on training data to address imbalance issues
    class_weights = get_optimal_class_weights(y_train)
    
    # Dictionary to store performance results for different models
    model_results = {}
    
    # Define various resampling strategies to try on the imbalanced dataset
    resampling_strategies = {
        'smote': SMOTE(sampling_strategy=0.1, random_state=RANDOM_STATE),
        'adasyn': ADASYN(sampling_strategy=0.1, random_state=RANDOM_STATE),
        'smote_tomek': SMOTETomek(sampling_strategy=0.1, random_state=RANDOM_STATE),
        'smote_enn': SMOTEENN(sampling_strategy=0.1, random_state=RANDOM_STATE)
    }
    
    # Identify the best resampling strategy using cross-validation performance (average precision)
    print("Selecting optimal resampling strategy...")
    best_resampler = None
    best_score = 0
    
    # Define a base XGBoost model with parameters tuned for imbalanced data using scale_pos_weight
    base_model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric="auc",
        objective="binary:logistic",
        random_state=RANDOM_STATE,
        tree_method='hist',
        scale_pos_weight=class_weights[1] / class_weights[0]
    )
    
    # Iterate over each resampling strategy to evaluate its performance
    for name, resampler in resampling_strategies.items():
        # Create an imbalanced-learn pipeline that first applies resampling and then fits the model
        pipeline = ImbPipeline([
            ('resampler', resampler),
            ('model', base_model)
        ])
        
        # Evaluate using 3-fold stratified cross-validation with average precision as the metric
        cv_scores = cross_val_score(
            pipeline, X_train, y_train, 
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
            scoring='average_precision'
        )
        
        avg_score = np.mean(cv_scores)
        print(f"  - {name}: Average Precision = {avg_score:.4f}")
        
        # Keep track of the best performing resampler based on the cross-validation score
        if avg_score > best_score:
            best_score = avg_score
            best_resampler = resampler
    
    print(f"Selected resampling strategy: {best_resampler.__class__.__name__}")
    
    # -------------------------------
    # Train XGBoost model with optimal resampling strategy
    print("Training XGBoost model...")
    xgb_pipeline = ImbPipeline([
        ('resampler', best_resampler),
        ('model', xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric="auc",
            objective="binary:logistic",
            random_state=RANDOM_STATE,
            tree_method='hist',
            scale_pos_weight=class_weights[1] / class_weights[0],
            n_jobs=-1
        ))
    ])
    
    # Define grid search parameters for the XGBoost model
    xgb_param_grid = {
        'model__max_depth': [3, 5, 7],
        'model__learning_rate': [0.01, 0.05, 0.1],
        'model__subsample': [0.8, 0.9, 1.0],
        'model__colsample_bytree': [0.7, 0.8, 0.9],
        'model__min_child_weight': [1, 3, 5],
        'model__gamma': [0, 0.1, 0.2],
        'model__n_estimators': [100, 200, 300]
    }
    
    # Set up grid search with stratified 3-fold cross-validation for hyperparameter tuning
    xgb_grid_search = GridSearchCV(
        xgb_pipeline, xgb_param_grid, 
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
        scoring='average_precision', verbose=1, n_jobs=-1
    )
    
    # Fit grid search using training data and validate on validation set with early stopping
    xgb_grid_search.fit(
        X_train, y_train, 
        model__eval_set=[(X_val, y_val)],
        model__early_stopping_rounds=50,
        model__verbose=0
    )
    
    # Retrieve the best XGBoost model from grid search
    best_xgb_model = xgb_grid_search.best_estimator_
    print(f"Best XGBoost parameters: {xgb_grid_search.best_params_}")
    
    # -------------------------------
    # Train LightGBM model with optimal resampling strategy
    print("Training LightGBM model...")
    lgb_pipeline = ImbPipeline([
        ('resampler', best_resampler),
        ('model', lgb.LGBMClassifier(
            objective='binary',
            random_state=RANDOM_STATE,
            class_weight='balanced',
            n_jobs=-1
        ))
    ])
    
    # Define grid search parameters for the LightGBM model
    lgb_param_grid = {
        'model__num_leaves': [31, 63, 127],
        'model__learning_rate': [0.01, 0.05, 0.1],
        'model__n_estimators': [100, 200, 300],
        'model__min_child_samples': [20, 30, 50],
        'model__subsample': [0.8, 0.9, 1.0],
        'model__colsample_bytree': [0.7, 0.8, 0.9]
    }
    
    # Set up grid search for LightGBM
    lgb_grid_search = GridSearchCV(
        lgb_pipeline, lgb_param_grid, 
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
        scoring='average_precision', verbose=1, n_jobs=-1
    )
    
    # Fit grid search using training data with validation for early stopping
    lgb_grid_search.fit(
        X_train, y_train,
        model__eval_set=[(X_val, y_val)],
        model__early_stopping_rounds=50,
        model__verbose=0
    )
    
    # Retrieve the best LightGBM model
    best_lgb_model = lgb_grid_search.best_estimator_
    print(f"Best LightGBM parameters: {lgb_grid_search.best_params_}")
    
    # -------------------------------
    # Create an ensemble model combining both XGBoost and LightGBM using a soft voting classifier
    print("Creating ensemble model...")
    ensemble_model = VotingClassifier(
        estimators=[
            ('xgb', best_xgb_model),
            ('lgb', best_lgb_model)
        ],
        voting='soft'  # Use predicted probabilities to combine the models
    )
    
    # Fit the ensemble model on the training data
    ensemble_model.fit(X_train, y_train)
    
    # Evaluate each model (XGBoost, LightGBM, Ensemble) on the validation set
    models = {
        'xgboost': best_xgb_model,
        'lightgbm': best_lgb_model,
        'ensemble': ensemble_model
    }
    
    for name, model in models.items():
        # Predict probability scores for the validation set
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # Calculate the Precision-Recall AUC (useful for imbalanced datasets)
        pr_auc = average_precision_score(y_val, y_pred_proba)
        # Calculate the ROC AUC score
        roc_auc = roc_auc_score(y_val, y_pred_proba)
        
        print(f"{name.capitalize()} - PR-AUC: {pr_auc:.4f}, ROC-AUC: {roc_auc:.4f}")
        # Store the results along with the trained model
        model_results[name] = {'pr_auc': pr_auc, 'roc_auc': roc_auc, 'model': model}
    
    # Select the best model based on the highest PR-AUC score
    best_model_name = max(model_results.items(), key=lambda x: x[1]['pr_auc'])[0]
    best_model = model_results[best_model_name]['model']
    print(f"Selected best model: {best_model_name}")
    
    # -------------------------------
    # If the best model is XGBoost, perform feature importance analysis and SHAP interpretation
    if best_model_name == 'xgboost':
        # Extract the XGBoost component from the pipeline
        model_component = best_model.named_steps['model']
        # Retrieve feature importances computed by XGBoost
        feature_importance = model_component.feature_importances_
        feature_names = X_train.columns
        
        # Create a DataFrame with feature names and their corresponding importance scores
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values(by='Importance', ascending=False)
        
        # Plot the top 20 most important features
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
        plt.title(f'Top 20 Important Features - {best_model_name.capitalize()}')
        plt.tight_layout()
        plt.savefig(f'plots/{best_model_name}_feature_importance.png')
        plt.close()
        
        print("Top 10 important features:")
        print(importance_df.head(10))
        
        # Generate SHAP values for model interpretability using a subset of the validation data
        try:
            X_shap = X_val.iloc[:1000]  # Sample a subset for faster computation
            explainer = shap.Explainer(model_component)
            shap_values = explainer(X_shap)
            
            # Plot a summary bar plot of SHAP values to understand feature impacts
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False)
            plt.tight_layout()
            plt.savefig(f'plots/{best_model_name}_shap_summary.png')
            plt.close()
            
            # Plot the full beeswarm plot for SHAP values to visualize detailed impacts
            plt.figure(figsize=(12, 10))
            shap.summary_plot(shap_values, X_shap, show=False)
            plt.tight_layout()
            plt.savefig(f'plots/{best_model_name}_shap_beeswarm.png')
            plt.close()
        except Exception as e:
            print(f"SHAP analysis error: {e}")
    
    return best_model, best_model_name, model_results

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the final model on the test dataset using various performance metrics and generate plots.
    Parameters:
      model: Trained model.
      X_test (DataFrame): Test set features.
      y_test (Series): Test set target.
    Returns:
      dict: Dictionary of evaluation metrics and related results.
    """
    print("Evaluating model on test set...")
    
    # Generate predictions for the test set
    y_pred = model.predict(X_test)
    # Generate probability predictions for the positive class
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate various performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    
    # Print the computed metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")
    
    # Compute and display the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # Plot and save the confusion matrix using a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('plots/confusion_matrix.png')
    plt.close()
    
    # Plot and save the ROC curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for random guessing
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('plots/roc_curve.png')
    plt.close()
    
    # Plot and save the Precision-Recall curve
    plt.figure(figsize=(8, 6))
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.plot(recall_curve, precision_curve, label=f'PR curve (area = {pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="best")
    plt.savefig('plots/pr_curve.png')
    plt.close()
    
    # Print a detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': cm,
        'y_pred_proba': y_pred_proba
    }

def optimize_threshold(y_test, y_pred_proba):
    """
    Optimize the classification threshold based on a cost-sensitive analysis and F-scores.
    Parameters:
      y_test (Series): True labels.
      y_pred_proba (array): Predicted probability scores.
    Returns:
      tuple: Final chosen threshold and a dictionary summarizing various thresholds.
    """
    print("Optimizing classification threshold...")
    
    # Compute precision, recall, and thresholds from the precision-recall curve
    precision_curve, recall_curve, thresholds = precision_recall_curve(y_test, y_pred_proba)
    
    # Calculate F1 score for each threshold using the harmonic mean of precision and recall
    f1_scores = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-10)
    
    # Calculate F2 score (which weighs recall higher than precision) for each threshold
    f2_scores = (1 + 2**2) * (precision_curve * recall_curve) / ((2**2 * precision_curve) + recall_curve + 1e-10)
    
    # Define example cost values for false negatives and false positives
    cost_fn = 50  # Cost for missing a fraud (false negative)
    cost_fp = 1   # Cost for falsely flagging a non-fraud (false positive)
    
    # Calculate expected cost for each threshold based on the cost matrix
    n_samples = len(y_test)
    n_pos = np.sum(y_test)
    n_neg = n_samples - n_pos
    
    costs = []
    for precision, recall, threshold in zip(precision_curve[:-1], recall_curve[:-1], thresholds):
        # Estimate true positives using recall and total positive samples
        tp = recall * n_pos
        # False negatives are the remaining positives not captured
        fn = n_pos - tp
        # Calculate false positives from precision (if precision > 0)
        fp = (tp / precision) - tp if precision > 0 else float('inf')
        # Calculate total cost normalized by the number of samples
        total_cost = (fn * cost_fn + fp * cost_fp) / n_samples
        costs.append((threshold, total_cost))
    
    # Identify the threshold that results in the minimum cost
    min_cost_threshold, min_cost = min(costs, key=lambda x: x[1])
    print(f"Minimum cost threshold: {min_cost_threshold:.4f} with average cost: {min_cost:.4f}")
    
    # Determine the threshold that gives the best F1 score
    best_f1_idx = np.argmax(f1_scores[:-1])  # Exclude last item as it does not correspond to a threshold
    best_f1_threshold = thresholds[best_f1_idx]
    best_f1 = f1_scores[best_f1_idx]
    print(f"Best F1 threshold: {best_f1_threshold:.4f} with F1 score: {best_f1:.4f}")
    
    # Determine the threshold that gives the best F2 score
    best_f2_idx = np.argmax(f2_scores[:-1])
    best_f2_threshold = thresholds[best_f2_idx]
    best_f2 = f2_scores[best_f2_idx]
    print(f"Best F2 threshold: {best_f2_threshold:.4f} with F2 score: {best_f2:.4f}")
    
    # Plot the expected cost and F-scores vs threshold for visualization
    plt.figure(figsize=(12, 8))
    
    # Extract thresholds and cost values for which cost is finite
    cost_thresholds, cost_values = zip(*[(t, c) for t, c in costs if c < float('inf')])
    
    plt.subplot(2, 1, 1)
    plt.plot(cost_thresholds, cost_values, 'r-', label='Expected Cost')
    plt.axvline(x=min_cost_threshold, color='r', linestyle='--', label=f'Min Cost Threshold ({min_cost_threshold:.2f})')
    plt.xlabel('Threshold')
    plt.ylabel('Expected Cost')
    plt.title('Expected Cost vs Threshold')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(thresholds, f1_scores[:-1], 'b-', label='F1 Score')
    plt.plot(thresholds, f2_scores[:-1], 'g-', label='F2 Score')
    plt.axvline(x=best_f1_threshold, color='b', linestyle='--', label=f'Best F1 Threshold ({best_f1_threshold:.2f})')
    plt.axvline(x=best_f2_threshold, color='g', linestyle='--', label=f'Best F2 Threshold ({best_f2_threshold:.2f})')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('F-Scores vs Threshold')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('plots/threshold_optimization.png')
    plt.close()
    
    # Choose the final threshold based on business requirements (here, minimum cost threshold is selected)
    final_threshold = min_cost_threshold
    
    # Recalculate predictions using the final threshold
    optimized_preds = (y_pred_proba >= final_threshold).astype(int)
    
    # Evaluate and print metrics using the optimized threshold
    print("Metrics with optimized threshold:")
    print(f"Accuracy: {accuracy_score(y_test, optimized_preds):.4f}")
    print(f"Precision: {precision_score(y_test, optimized_preds):.4f}")
    print(f"Recall: {recall_score(y_test, optimized_preds):.4f}")
    print(f"F1 Score: {f1_score(y_test, optimized_preds):.4f}")
    
    return final_threshold, {
        'min_cost_threshold': min_cost_threshold,
        'best_f1_threshold': best_f1_threshold,
        'best_f2_threshold': best_f2_threshold,
        'final_threshold': final_threshold
    }

def save_model(model, metadata, output_path="fraud_detection_model"):
    """
    Save the trained model and its metadata (such as preprocessing artifacts and threshold)
    Parameters:
      model: The trained model to save.
      metadata (dict): Additional information about the model (excluding callable objects).
      output_path (str): Directory path where the model and metadata will be saved.
    """
    print(f"Saving model to {output_path}...")
    
    # Ensure the output directory exists; if not, create it
    os.makedirs(os.path.dirname(f"{output_path}/model.pkl") if os.path.dirname(f"{output_path}/model.pkl") else ".", exist_ok=True)
    
    # Save the model using pickle
    with open(f"{output_path}/model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    # Save metadata as JSON, filtering out callables and the model object itself
    with open(f"{output_path}/metadata.json", "w") as f:
        json.dump({k: v for k, v in metadata.items() if not callable(v) and k != 'model'}, f, default=str)
    
    print("Model and metadata saved successfully.")

def create():
    # Placeholder for a function that might encapsulate the entire workflow
    # Currently, it does nothing (pass). You can later fill this function to sequentially call
    # the data loading, preprocessing, training, evaluation, and saving functions.
    pass
