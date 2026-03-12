# atmotwin_ml/train_model.py

import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import (
    train_test_split, 
    GridSearchCV, 
    StratifiedKFold,
    cross_val_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    accuracy_score
)
import joblib

# Import our data loader
from data_loader import prepare_dataset, CLASS_NAMES

# =============================================================================
# CONFIGURATION
# =============================================================================

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Hyperparameter grids
RF_PARAM_GRID = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# Smaller grid for faster initial testing
RF_PARAM_GRID_QUICK = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_leaf': [1, 4],
    'max_features': ['sqrt']
}

LR_PARAM_GRID = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],
    'max_iter': [1000]
}

# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def split_data(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Split data with stratification to maintain class balance."""
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        stratify=y, 
        random_state=random_state
    )
    
    print(f"Training set: {len(y_train)} samples")
    print(f"Test set: {len(y_test)} samples")
    print(f"\nClass distribution (train):")
    for i, name in enumerate(CLASS_NAMES):
        count = np.sum(y_train == i)
        print(f"  {name}: {count}")
    
    return X_train, X_test, y_train, y_test

def train_random_forest(X_train, y_train, quick=False, verbose=True):
    """
    Train Random Forest with hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training labels
        quick: Use smaller grid for faster testing
        verbose: Print progress
    
    Returns:
        best_model: Trained model with best hyperparameters
        cv_results: Cross-validation results
    """
    
    param_grid = RF_PARAM_GRID_QUICK if quick else RF_PARAM_GRID
    
    if verbose:
        print("\n" + "=" * 60)
        print("Training Random Forest with GridSearchCV")
        print("=" * 60)
        n_combinations = 1
        for v in param_grid.values():
            n_combinations *= len(v)
        print(f"Parameter combinations: {n_combinations}")
        print(f"CV folds: {CV_FOLDS}")
        print(f"Total fits: {n_combinations * CV_FOLDS}")
    
    # Stratified cross-validation
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    # Grid search
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1 if verbose else 0,
        return_train_score=True
    )
    
    grid_search.fit(X_train, y_train)
    
    if verbose:
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best CV accuracy: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.cv_results_

def train_logistic_regression(X_train, y_train, verbose=True):
    """
    Train Logistic Regression as baseline comparison.
    
    Uses StandardScaler since LR is sensitive to feature scales.
    """
    
    if verbose:
        print("\n" + "=" * 60)
        print("Training Logistic Regression (baseline)")
        print("=" * 60)
    
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    # Pipeline with scaling (important for LR)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(random_state=RANDOM_STATE, max_iter=1000))
    ])
    
    # Adjust param grid for pipeline
    param_grid = {f'lr__{k}': v for k, v in LR_PARAM_GRID.items()}
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1 if verbose else 0
    )
    
    grid_search.fit(X_train, y_train)
    
    if verbose:
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best CV accuracy: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.cv_results_

# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Comprehensive model evaluation."""
    
    print("\n" + "=" * 60)
    print(f"Evaluation: {model_name}")
    print("=" * 60)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print_confusion_matrix(cm)
    
    # Confidence analysis
    analyze_confidence(y_test, y_pred, y_proba)
    
    return {
        'accuracy': accuracy,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'confusion_matrix': cm
    }

def print_confusion_matrix(cm):
    """Pretty print confusion matrix."""
    
    # Header
    print(f"\n{'':>20} {'Predicted':^45}")
    print(f"{'':>15}", end='')
    for name in CLASS_NAMES:
        short = name[:8]
        print(f"{short:>10}", end='')
    print()
    print(" " * 15 + "-" * 42)
    
    # Rows
    for i, name in enumerate(CLASS_NAMES):
        short = name[:12]
        if i == 0:
            print(f"{'Actual':>7} {short:>7}", end='')
        else:
            print(f"{'':>7} {short:>7}", end='')
        for j in range(len(CLASS_NAMES)):
            print(f"{cm[i, j]:>10}", end='')
        print()

def analyze_confidence(y_test, y_pred, y_proba):
    """Analyze prediction confidence."""
    
    print("\nConfidence Analysis:")
    
    # Overall confidence
    max_proba = np.max(y_proba, axis=1)
    print(f"  Mean confidence: {max_proba.mean():.3f}")
    print(f"  Min confidence: {max_proba.min():.3f}")
    print(f"  Max confidence: {max_proba.max():.3f}")
    
    # Confidence on correct vs incorrect predictions
    correct_mask = y_test == y_pred
    
    if correct_mask.sum() > 0:
        conf_correct = max_proba[correct_mask].mean()
        print(f"  Mean confidence (correct): {conf_correct:.3f}")
    
    if (~correct_mask).sum() > 0:
        conf_incorrect = max_proba[~correct_mask].mean()
        print(f"  Mean confidence (incorrect): {conf_incorrect:.3f}")

# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================

def get_feature_importance(model, feature_names, top_n=20):
    """Extract and rank feature importances from Random Forest."""
    
    # Handle pipeline (for Logistic Regression)
    if hasattr(model, 'named_steps'):
        if 'lr' in model.named_steps:
            # For LR, use absolute coefficient values
            coefs = np.abs(model.named_steps['lr'].coef_).mean(axis=0)
            importances = coefs / coefs.sum()
        else:
            print("Cannot extract feature importance from this model type")
            return None
    else:
        # Random Forest
        importances = model.feature_importances_
    
    # Sort by importance
    indices = np.argsort(importances)[::-1]
    
    print("\n" + "=" * 60)
    print(f"Top {top_n} Most Important Features")
    print("=" * 60)
    
    results = []
    for i in range(min(top_n, len(indices))):
        idx = indices[i]
        name = feature_names[idx]
        imp = importances[idx]
        results.append({'feature': name, 'importance': imp})
        print(f"  {i+1:>2}. {name:<25} {imp:.4f}")
    
    return results, indices, importances

# =============================================================================
# SAVE/LOAD MODELS
# =============================================================================

def save_model(model, feature_names, metadata, output_dir='models'):
    """Save trained model and metadata."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save model
    model_path = output_dir / 'atmotwin_classifier.joblib'
    joblib.dump(model, model_path)
    print(f"\nModel saved: {model_path}")
    
    # Save feature names
    features_path = output_dir / 'feature_names.json'
    with open(features_path, 'w') as f:
        json.dump(feature_names, f, indent=2)
    print(f"Features saved: {features_path}")
    
    # Save metadata
    meta_path = output_dir / 'model_metadata.json'
    
    # Convert numpy types for JSON serialization
    meta_serializable = {}
    for k, v in metadata.items():
        if isinstance(v, np.ndarray):
            meta_serializable[k] = v.tolist()
        elif isinstance(v, np.integer):
            meta_serializable[k] = int(v)
        elif isinstance(v, np.floating):
            meta_serializable[k] = float(v)
        else:
            meta_serializable[k] = v
    
    with open(meta_path, 'w') as f:
        json.dump(meta_serializable, f, indent=2)
    print(f"Metadata saved: {meta_path}")

def load_model(model_dir='models'):
    """Load trained model and metadata."""
    
    model_dir = Path(model_dir)
    
    model = joblib.load(model_dir / 'atmotwin_classifier.joblib')
    
    with open(model_dir / 'feature_names.json', 'r') as f:
        feature_names = json.load(f)
    
    with open(model_dir / 'model_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    return model, feature_names, metadata

# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================

def train_pipeline(data_path='psg_dataset/atmotwin_training_data.npz',
                   quick=False,
                   save=True):
    """
    Full training pipeline.
    
    Args:
        data_path: Path to training data
        quick: Use smaller hyperparameter grid for faster testing
        save: Save trained model to disk
    
    Returns:
        results: Dict with models, evaluations, feature importances
    """
    
    print("=" * 60)
    print("AtmoTwin ML Training Pipeline")
    print("=" * 60)
    
    # Load and prepare data
    print("\n[1/5] Loading data...")
    X, y, feature_names, metadata = prepare_dataset(data_path)
    print(f"Loaded {metadata['n_samples']} samples, {metadata['n_features']} features")
    
    # Split data
    print("\n[2/5] Splitting data...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Train Random Forest
    print("\n[3/5] Training Random Forest...")
    rf_model, rf_cv_results = train_random_forest(X_train, y_train, quick=quick)
    
    # Train Logistic Regression (baseline)
    print("\n[4/5] Training Logistic Regression (baseline)...")
    lr_model, lr_cv_results = train_logistic_regression(X_train, y_train)
    
    # Evaluate both models
    print("\n[5/5] Evaluating models...")
    rf_eval = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    lr_eval = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    
    # Compare models
    print("\n" + "=" * 60)
    print("Model Comparison")
    print("=" * 60)
    print(f"Random Forest accuracy:       {rf_eval['accuracy']:.4f}")
    print(f"Logistic Regression accuracy: {lr_eval['accuracy']:.4f}")
    
    if rf_eval['accuracy'] > lr_eval['accuracy']:
        print("\n→ Random Forest performs better")
        print("  (Suggests non-linear patterns in the data)")
        best_model = rf_model
        best_name = 'Random Forest'
    elif rf_eval['accuracy'] < lr_eval['accuracy']:
        print("\n→ Logistic Regression performs better")
        print("  (Classes may be linearly separable - simpler model preferred)")
        best_model = lr_model
        best_name = 'Logistic Regression'
    else:
        print("\n→ Models perform equally")
        print("  (Prefer simpler Logistic Regression)")
        best_model = lr_model
        best_name = 'Logistic Regression'
    
    # Feature importance (Random Forest)
    print("\n" + "=" * 60)
    print("Feature Importance Analysis (Random Forest)")
    print("=" * 60)
    importance_results, importance_indices, importances = get_feature_importance(
        rf_model, feature_names, top_n=20
    )
    
    # Check if engineered features are important
    print("\nEngineered Feature Rankings:")
    engineered = ['o3_depth', 'ch4_depth', 'n2o_depth', 'co_depth', 
                  'h2o_depth', 'co2_depth', 'o3_ch4_ratio', 
                  'co_o3_ratio', 'ch4_co2_ratio']
    
    for eng_feat in engineered:
        if eng_feat in feature_names:
            idx = feature_names.index(eng_feat)
            rank = list(importance_indices).index(idx) + 1
            imp = importances[idx]
            print(f"  {eng_feat:<20} rank: {rank:>3}, importance: {imp:.4f}")
    
    # Save best model
    if save:
        print("\n" + "=" * 60)
        print("Saving Model")
        print("=" * 60)
        
        save_metadata = {
            **metadata,
            'best_model': best_name,
            'rf_accuracy': rf_eval['accuracy'],
            'lr_accuracy': lr_eval['accuracy'],
            'rf_params': rf_model.get_params() if hasattr(rf_model, 'get_params') else {},
        }
        
        save_model(rf_model, feature_names, save_metadata)
    
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)
    
    return {
        'rf_model': rf_model,
        'lr_model': lr_model,
        'rf_eval': rf_eval,
        'lr_eval': lr_eval,
        'feature_names': feature_names,
        'feature_importance': importance_results,
        'X_test': X_test,
        'y_test': y_test,
        'metadata': metadata
    }

# =============================================================================
# RUN
# =============================================================================

if __name__ == '__main__':
    # Use quick=True for fast testing, quick=False for full grid search
    results = train_pipeline(quick=True, save=True)