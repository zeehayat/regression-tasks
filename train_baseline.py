import os
import sys

# Ensure data is generated
if not os.path.exists(os.path.join("data", "kp_subdistrict_development_index.csv")):
    print("CSV files not found. Launching data generator first...")
    try:
        from generate_data import generate_development_dataset, generate_agricultural_dataset, generate_infrastructure_dataset
        os.makedirs("data", exist_ok=True)
        generate_development_dataset(os.path.join("data", "kp_subdistrict_development_index.csv"), 100000)
        generate_agricultural_dataset(os.path.join("data", "kp_agricultural_yields.csv"), 100000)
        generate_infrastructure_dataset(os.path.join("data", "kp_infrastructure_projects.csv"), 100000)
    except Exception as e:
        print(f"Error during automatic generation: {e}")
        print("Please check generate_data.py and run it manually.")
        sys.exit(1)

# Check for ML libraries
try:
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
except ImportError as e:
    print("\n[Missing Dependencies] Pandas, NumPy, or Scikit-Learn is not installed.")
    print("Please install them using the following command:")
    print("pip install pandas numpy scikit-learn")
    print("\nIf you just want to generate the CSV files, run: python generate_data.py")
    sys.exit(1)

def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"--- {model_name} Results ---")
    print(f"  Mean Absolute Error (MAE) : {mae:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"  R-squared (R²) Score      : {r2:.4f}")
    print("-" * 30)
    return r2

def train_regression_on_dataset(csv_path, target_col, drop_cols, dataset_name):
    print(f"\n==================================================")
    print(f"Training Regression Models on: {dataset_name}")
    print(f"Source file: {csv_path}")
    print(f"==================================================")
    
    # Load dataset
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset with shape: {df.shape}")
    
    # Separate features and target
    X = df.drop(columns=[target_col] + drop_cols)
    y = df[target_col]
    
    # Handle categorical variables (One-Hot Encoding)
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    print(f"Categorical features detected & encoded: {categorical_cols}")
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")
    
    # Model 1: Baseline Linear Regression
    print("\nTraining Linear Regression model...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)
    evaluate_model(y_test, lr_preds, "Linear Regression")
    
    # Model 2: Decision Tree Regressor (Non-linear baseline)
    print("\nTraining Decision Tree Regressor (depth=10)...")
    dt_model = DecisionTreeRegressor(max_depth=10, random_state=42)
    dt_model.fit(X_train, y_train)
    dt_preds = dt_model.predict(X_test)
    evaluate_model(y_test, dt_preds, "Decision Tree Regressor")

if __name__ == "__main__":
    # Task 1: Socio-economic Development Index Prediction
    train_regression_on_dataset(
        csv_path=os.path.join("data", "kp_subdistrict_development_index.csv"),
        target_col="development_score",
        drop_cols=["community_id"],
        dataset_name="KP Subdistrict Development Index"
    )
    
    # Task 2: Crop Yield Prediction
    train_regression_on_dataset(
        csv_path=os.path.join("data", "kp_agricultural_yields.csv"),
        target_col="crop_yield_tons_per_acre",
        drop_cols=["farm_id"],
        dataset_name="KP Farm Agricultural Yields"
    )
    
    # Task 3: Infrastructure Projects Cost Prediction
    train_regression_on_dataset(
        csv_path=os.path.join("data", "kp_infrastructure_projects.csv"),
        target_col="actual_cost_million_pkr",
        drop_cols=["project_id", "actual_duration_months"], # drop actual duration since it's a co-target
        dataset_name="KP Infrastructure Projects Cost"
    )
