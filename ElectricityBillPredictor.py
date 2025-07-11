# Step 1: Import Libraries.
import pandas as pd
import numpy as np

# For encoding, scaling, splitting
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# For modeling
from sklearn.ensemble import RandomForestRegressor

# For evaluation
from sklearn.metrics import mean_squared_error, r2_score


if __name__ == "__main__":
    # Step 2: Load Dataset
    df = pd.read_csv('electricity_bill_dataset copy.csv')

    # Step 3: Quick Inspection
    print("First 5 rows:")
    print(df.head())
    print("\nData Types:")
    print(df.dtypes)
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Step 3.5: Visualize Correlation Matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()

    # Step 3.6: Scatter plot of MonthlyHours vs ElectricityBill
    plt.figure(figsize=(6, 4))
    plt.scatter(df['MonthlyHours'], df['ElectricityBill'], alpha=0.6, edgecolors='k')
    plt.xlabel("Monthly Hours")
    plt.ylabel("Electricity Bill")
    plt.title("MonthlyHours vs ElectricityBill")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Step 3.7: Feature Engineering - Compute Cost Per Hour
    df['CostPerHour'] = df['ElectricityBill'] / df['MonthlyHours'].replace(0, np.nan)
    df['CostPerHour'] = df['CostPerHour'].fillna(0)
    print("\nNew feature 'CostPerHour' added.")

    #gpt 1 over

    # Step 4: Encode Categorical Columns
    le_city = LabelEncoder()
    le_company = LabelEncoder()

    df['City'] = le_city.fit_transform(df['City'])
    df['Company'] = le_company.fit_transform(df['Company'])

    # Step 5: Scale Numerical Features
    scaler = StandardScaler()
    numerical_cols = ['Fan', 'Refrigerator', 'AirConditioner', 'Television', 'Monitor', 'MotorPump',
                      'MonthlyHours', 'TariffRate', 'CostPerHour']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Step 6: Define Features and Target
    X = df.drop(columns=['ElectricityBill'])
    y = df['ElectricityBill']

    # Step 7: Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\nTrain/Test split done!")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Step 8: Train Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Step 9: Make Predictions
    y_pred = model.predict(X_test)

    # Step 10: Evaluate Model
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\nModel Performance:")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.4f}")

    # Step 11: Feature Importance Plot
    import matplotlib.pyplot as plt

    importances = model.feature_importances_
    features = X.columns

    plt.figure(figsize=(10, 6))
    plt.barh(features, importances, color='skyblue')
    plt.xlabel("Feature Importance")
    plt.title("Random Forest Feature Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    # Step 12: Actual vs Predicted Scatter Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='k')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual Bill")
    plt.ylabel("Predicted Bill")
    plt.title("Actual vs Predicted Electricity Bill")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Step 13: Save Model Summary to File
    import os

    summary_path = 'report/model_summary.txt'
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    with open(summary_path, 'w') as f:
        f.write("Electricity Bill Prediction - Model Summary\n")
        f.write("==========================================\n\n")
        f.write(f"Model Used: Random Forest Regressor\n")
        f.write(f"Training Samples: {len(X_train)}\n")
        f.write(f"Test Samples: {len(X_test)}\n\n")
        f.write(f"RMSE: {rmse:.2f}\n")
        f.write(f"R² Score: {r2:.4f}\n\n")
        f.write("Top 5 Most Important Features:\n")
        top_features = sorted(zip(features, importances), key=lambda x: -x[1])[:5]
        for feat, score in top_features:
            f.write(f"- {feat}: {score:.4f}\n")

# Run everything
# Step 2 to Step 13 already defined above will execute sequentially
    pass
