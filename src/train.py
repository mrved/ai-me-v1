import pandas as pd
import joblib
import numpy as np
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, 
    r2_score, 
    mean_absolute_error,
    mean_absolute_percentage_error
)

DB_PATH = "sqlite:///data/metadata.db"
MODEL_PATH = "model.pkl"

def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive regression metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    # Calculate MAPE (handle division by zero)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    
    r2 = r2_score(y_true, y_pred)
    
    # Mean percentage error
    mean_error = np.mean(y_pred - y_true)
    mean_abs_error = np.mean(np.abs(y_pred - y_true))
    
    # Calculate accuracy as percentage within tolerance
    # For stress prediction, 10% tolerance is reasonable
    tolerance = 0.10  # 10%
    within_tolerance = np.abs((y_pred - y_true) / (y_true + 1e-10)) <= tolerance
    accuracy_pct = np.mean(within_tolerance) * 100
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R² Score': r2,
        'Mean Error': mean_error,
        'Mean Absolute Error': mean_abs_error,
        'Accuracy (10% tolerance)': accuracy_pct
    }

def train_model():
    # 1. Load Data
    engine = create_engine(DB_PATH)
    df = pd.read_sql('simulations', engine)
    
    print(f"Loaded {len(df)} samples from database")
    
    # 2. Features (X) and Target (y)
    # Use all available parameters for better predictions
    # Start with basic parameters
    feature_cols = ['length', 'width', 'height', 'load']
    
    # Add additional parameters if available
    if 'drag_coefficient' in df.columns:
        feature_cols.append('drag_coefficient')
    if 'wheelbase' in df.columns:
        feature_cols.append('wheelbase')
    if 'roof_angle' in df.columns:
        feature_cols.append('roof_angle')
    
    # Only use columns that exist and have valid data
    available_cols = [col for col in feature_cols if col in df.columns and df[col].notna().any()]
    
    print(f"Using {len(available_cols)} features: {available_cols}")
    X = df[available_cols]
    y = df['max_stress']
    
    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # 4. Train Model
    # Using Random Forest as a robust baseline
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 5. Evaluate on test set
    y_pred_test = model.predict(X_test)
    test_metrics = calculate_metrics(y_test, y_pred_test)
    
    # Also evaluate on training set to check for overfitting
    y_pred_train = model.predict(X_train)
    train_metrics = calculate_metrics(y_train, y_pred_train)
    
    # 6. Print Results
    print("\n" + "="*60)
    print("MODEL EVALUATION METRICS")
    print("="*60)
    print("\n📊 Test Set Performance:")
    print(f"  R² Score:           {test_metrics['R² Score']:.4f} (1.0 = perfect)")
    print(f"  RMSE:               {test_metrics['RMSE']:,.2f} Pa")
    print(f"  MAE:                {test_metrics['MAE']:,.2f} Pa")
    print(f"  MAPE:               {test_metrics['MAPE']:.2f}%")
    print(f"  Accuracy (10% tol): {test_metrics['Accuracy (10% tolerance)']:.2f}%")
    print(f"  Mean Error:         {test_metrics['Mean Error']:,.2f} Pa")
    
    print("\n📈 Training Set Performance (for overfitting check):")
    print(f"  R² Score:           {train_metrics['R² Score']:.4f}")
    print(f"  RMSE:               {train_metrics['RMSE']:,.2f} Pa")
    
    # Check for overfitting
    r2_diff = train_metrics['R² Score'] - test_metrics['R² Score']
    if r2_diff > 0.15:
        print(f"\n⚠️  Warning: Possible overfitting (R² diff: {r2_diff:.3f})")
    else:
        print(f"\n✅ Model generalizes well (R² diff: {r2_diff:.3f})")
    
    print("\n" + "="*60)
    
    # 7. Save Model
    joblib.dump(model, MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")
    
    # 8. Save metrics for later use
    metrics_df = pd.DataFrame({
        'Metric': list(test_metrics.keys()),
        'Value': list(test_metrics.values())
    })
    metrics_path = "model_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"✅ Metrics saved to {metrics_path}")
    
    return model, test_metrics, train_metrics

if __name__ == "__main__":
    train_model()
