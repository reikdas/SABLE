import os
import pathlib
import warnings

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

FILEPATH=pathlib.Path(__file__).resolve().parent
BASEPATH=os.path.join(FILEPATH.parent)

warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

if __name__ == "__main__":
    # Load data into a DataFrame
    df = pd.read_csv(os.path.join(FILEPATH,"threshold_results.csv"))
    df = df[df['nnz'] != 0]

    df['density'] = 100 - df['perc_zeros']
    # df['size'] = df['dim1'] * df['dim2']

    # Calculate the target variable:
    df['target'] = (df['CSR_time'] / df['sable_time'])
    df['target'] = df['target'].astype(int)  # Convert to binary (1 for True, 0 for False)

    # Define features (X) and target (y)
    X = df[['dim1', 'dim2', 'nnz', 'density']]
    y = df['target']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a logistic regression model
    model = RandomForestClassifier()
    model.fit(X_train.values, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")

    model_filename = os.path.join(BASEPATH, "models", "density_threshold_spmv.pkl")
    joblib.dump(model, model_filename)

    # print(model.predict([[10*10, 45]])) # Variable
    print(model.predict([[50, 50, 1, 1]])) # Expect 0
    print(model.predict([[50, 50, 50*50, 100]])) # Expect 1
    # print(X_train)
    # print(y_train)
    # print(df['target'].value_counts())
