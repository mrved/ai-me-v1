"""
Model Evaluation Script
Run this anytime to evaluate the trained model's performance
"""
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sklearn.metrics import (
    mean_squared_error, 
    r2_score, 
    mean_absolute_error
)
from pathlib import Path

DB_PATH = "sqlite:///data/metadata.db"
MODEL_PATH = "model.pkl"

def load_model_and_data():
    """Load the trained model and data"""
    # Load model
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train.py first.")
    
    model = joblib.load(MODEL_PATH)
    
    # Load data
    engine = create_engine(DB_PATH)
    df = pd.read_sql('simulations', engine)
    
    X = df[['length', 'width', 'height', 'load']]
    y = df['max_stress']
    
    return model, X, y, df

def calculate_comprehensive_metrics(y_true, y_pred):
    """Calculate all regression metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    
    # Accuracy within different tolerances
    tolerances = [0.05, 0.10, 0.15, 0.20]  # 5%, 10%, 15%, 20%
    accuracy_dict = {}
    for tol in tolerances:
        within_tol = np.abs((y_pred - y_true) / (y_true + 1e-10)) <= tol
        accuracy_dict[f'Accuracy ({int(tol*100)}% tolerance)'] = np.mean(within_tol) * 100
    
    # Mean error and bias
    mean_error = np.mean(y_pred - y_true)
    median_error = np.median(y_pred - y_true)
    
    return {
        'R² Score': r2,
        'RMSE (Pa)': rmse,
        'MAE (Pa)': mae,
        'MAPE (%)': mape,
        'Mean Error (Pa)': mean_error,
        'Median Error (Pa)': median_error,
        **accuracy_dict
    }

def plot_predictions(y_true, y_pred, save_path="prediction_plot.png"):
    """Create visualization of predictions vs actual"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot: Predicted vs Actual
    axes[0].scatter(y_true, y_pred, alpha=0.6, s=50)
    axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Stress (Pa)')
    axes[0].set_ylabel('Predicted Stress (Pa)')
    axes[0].set_title('Predicted vs Actual Stress')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Residual plot
    residuals = y_pred - y_true
    axes[1].scatter(y_true, residuals, alpha=0.6, s=50)
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Actual Stress (Pa)')
    axes[1].set_ylabel('Residuals (Predicted - Actual)')
    axes[1].set_title('Residual Plot')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved to {save_path}")
    return fig

def evaluate_model():
    """Main evaluation function"""
    print("="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Load model and data
    model, X, y, df = load_model_and_data()
    
    # Make predictions on all data
    y_pred = model.predict(X)
    
    # Calculate metrics
    metrics = calculate_comprehensive_metrics(y, y_pred)
    
    # Print results
    print("\n📊 Model Performance Metrics:")
    print("-" * 60)
    for metric, value in metrics.items():
        if 'Score' in metric or 'Accuracy' in metric or 'MAPE' in metric:
            print(f"  {metric:30s}: {value:8.2f}")
        else:
            print(f"  {metric:30s}: {value:12,.2f}")
    
    # Feature importance
    print("\n🔍 Feature Importance:")
    print("-" * 60)
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    for _, row in feature_importance.iterrows():
        print(f"  {row['Feature']:20s}: {row['Importance']:.4f}")
    
    # Create visualization
    try:
        plot_predictions(y, y_pred)
    except Exception as e:
        print(f"⚠️  Could not create visualization: {e}")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'Metric': list(metrics.keys()),
        'Value': list(metrics.values())
    })
    metrics_df.to_csv('model_evaluation.csv', index=False)
    print(f"\n✅ Metrics saved to model_evaluation.csv")
    
    return metrics, feature_importance

if __name__ == "__main__":
    evaluate_model()

