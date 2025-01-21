import os
import pathlib

import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

FILEPATH=pathlib.Path(__file__).resolve().parent
BASEPATH=os.path.join(FILEPATH.parent)

if __name__ == "__main__":
    # Load data into a DataFrame
    df = pd.read_csv(os.path.join(FILEPATH,"threshold_results.csv"))
    df = df[df['sable_time'] != 0]

    df['density'] = 100 - df['perc_zeros']
    df['size'] = df['dim'] * df['dim']

    # Calculate the target variable: (nonzeros_time / sable_time) >= 1
    df['target'] = (df['nonzeros_time'] / df['sable_time']) >= 1
    df['target'] = df['target'].astype(int)  # Convert to binary (1 for True, 0 for False)

    # Define features (X) and target (y)
    X = df[['size', 'density']]
    y = df['target']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize features
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    # Train a logistic regression model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")

    model_filename = os.path.join(BASEPATH, "models", "density_threshold_spmv.pkl")
    joblib.dump(model, model_filename)

    print(model.predict([[10*10, 45]])) # Variable
    print(model.predict([[50*50, 1]])) # Expect 0
    print(model.predict([[50*50, 100]])) # Expect 1
    # print(X_train)
    # print(y_train)
    # print(df['target'].value_counts())
