# Khyber Pakhtunkhwa (KP) Developmental Landscape - Regression Datasets

This repository contains synthetic datasets modeling the developmental, agricultural, and infrastructural landscape of Khyber Pakhtunkhwa (KP), Pakistan. These datasets are designed specifically for training and testing regression models in Machine Learning.

## Datasets Overview

We have modeled three distinct aspects of KP's developmental landscape, each containing **100,000 records** to provide a rich dataset for training complex ML models (like Ridge/Lasso, Random Forests, Gradient Boosted Trees, or Deep Learning Neural Networks).

### 1. KP Subdistrict Development Index (`data/kp_subdistrict_development_index.csv`)
* **Objective**: Predict the **Development Score** (Human Development Index equivalent) of local communities.
* **Regression Target**: `development_score` (Continuous scale `0.0` - `100.0`)
* **Key Features**:
  * `community_id`: Unique identifier (`COM-000001` onwards)
  * `district` & `division`: Geographic identifiers (Peshawar, Swat, Hazara, etc.)
  * `terrain_type`: plain, hilly, or mountainous (influences infrastructural accessibility)
  * `population`: Local neighborhood population
  * `literacy_rate` & `school_enrollment_rate`: Education indicators
  * `health_facility_distance_km`: Proximity to healthcare
  * `doctors_per_1000`: Medical personnel density
  * `road_connectivity_index` & `electricity_access_pct` & `clean_water_access_pct` & `internet_penetration_pct`: Infrastructure coverage indicators
  * `average_household_income_pkr`: Household socioeconomic status
  * `public_funding_allocated_million_pkr`: Government funding injections

### 2. KP Agricultural Yields (`data/kp_agricultural_yields.csv`)
* **Objective**: Predict farm-level **Crop Yield** based on location, climate, irrigation, and management practices.
* **Regression Target**: `crop_yield_tons_per_acre` (Continuous yield per acre)
* **Key Features**:
  * `farm_id`: Unique farm identifier (`FARM-000001` onwards)
  * `district` & `crop_type`: Regional suitability (e.g., Peaches/Apples in Swat; Tobacco in Swabi/Mardan; Dates in D.I. Khan; Wheat/Maize in plains)
  * `farm_size_acres`: Land holding size
  * `elevation_meters`: Farm elevation (highly correlated with crop type and climate)
  * `soil_ph`: Soil acidity/alkalinity
  * `fertilizer_used_bags_per_acre`: Synthetic nutrient inputs
  * `organic_farming`: Boolean (Yes/No) representing chemical-free farming
  * `irrigation_type`: Canal, Tube-well, Drip, or Rain-fed (barani)
  * `annual_rainfall_mm` & `average_temperature_celsius`: Climate factors
  * `pesticide_sprays_count`: Chemical crop protection frequency
  * `mechanization_level`: Modern farming equipment penetration (0-100)
  * `distance_to_mandi_km`: Transport distance to agricultural markets

### 3. KP Infrastructure Projects (`data/kp_infrastructure_projects.csv`)
* **Objective**: Multi-target regression to predict public infrastructure project completion **costs** and **delays**.
* **Regression Targets**: 
  1. `actual_cost_million_pkr` (Final cost of project)
  2. `actual_duration_months` (Final completion time)
* **Key Features**:
  * `project_id` & `project_sector`: Sector type (Roads, Health, Education, Energy, Water, etc.)
  * `district` & `terrain_complexity`: Work conditions (Low/Medium/High complexity based on terrain)
  * `project_scale`: Project size class (Small, Medium, Large, Mega)
  * `estimated_duration_months` & `approved_cost_million_pkr`: Planned schedule and budget
  * `contractor_experience_years`: Experience profile of the executing contractor
  * `distance_from_hq_km`: Remoteness of the project site from District HQ
  * `funding_release_rate_pct`: Administrative/financial disbursement speed
  * `monitoring_frequency_per_year`: Oversight frequency by development authorities
  * `year_of_initiation`: Calendar year project launched
  * `inflation_rate_at_start_pct`: Historical inflation conditions (captures high inflation periods in Pakistan, e.g. 2022-2024)
  * `labor_availability_index`: Availability of local labor force

---

## How to Generate the Datasets

To generate the full **300,000 records** across all three files:

1. Open your terminal in the workspace directory.
2. Run the generator script:
   ```bash
   python generate_data.py
   ```
This will automatically create a `data/` folder and populate it with the three CSV files:
* `data/kp_subdistrict_development_index.csv` (100k rows)
* `data/kp_agricultural_yields.csv` (100k rows)
* `data/kp_infrastructure_projects.csv` (100k rows)

---

## Getting Started with Regression in Python

Here is a quick snippet to load one of the datasets and train a baseline regression model:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
df = pd.read_csv("data/kp_subdistrict_development_index.csv")

# Separate features and target
X = df.drop(columns=["community_id", "development_score"])
y = df["development_score"]

# Define categorical and numerical features
categorical_features = ["district", "division", "terrain_type"]
numerical_features = [col for col in X.columns if col not in categorical_features]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# Combine preprocessing and model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
print("Training Random Forest Regressor...")
model.fit(X_train, y_train)

# Predict and Evaluate
y_pred = model.predict(X_test)
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
```
