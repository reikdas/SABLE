import os
import pathlib

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# from src.consts import SPEEDUP_THRESH
SPEEDUP_THRESH=1.3

FILEPATH=pathlib.Path(__file__).resolve().parent
BASEPATH=os.path.join(FILEPATH.parent)

if __name__ == "__main__":
    # Load data into a DataFrame
    df = pd.read_csv(os.path.join(FILEPATH,"threshold_results.csv"))

    # Filter for rows where dim1=1, dim2=1, and nnz=0
    filtered_df = df[(df['dim1'] == 1) & (df['dim2'] == 1) & (df['nnz'] == 0)]
    startup_time = filtered_df.iloc[0]['CSR_time']
    
    # Subtract startup_time from all CSR_time values
    df['CSR_time'] = df['CSR_time'] - startup_time

    df['density'] = 100 - df['perc_zeros']
    # df['size'] = df['dim1'] * df['dim2']

    # Calculate the target variable:
    df['target'] = (df['CSR_time'] / df['sable_time']) >= SPEEDUP_THRESH
    df['target'] = df['target'].astype(int)  # Convert to binary (1 for True, 0 for False)

    df.loc[df['density'] <= 25, 'target'] = 0

    # df.loc[(df['dim1'] < 8) & (df['dim2'] < 8), 'target'] = 0
    df.loc[(df['nnz'] < 800), 'target'] = 0

    df.loc[(df['dim2'] == 1) & (df['dim1'] < 1300), 'target'] = 0

    # Set target for size < 500 to 0
    # df.loc[(df['dim1'] * df['dim2']) < 1000, 'target'] = 0

    # Define features (X) and target (y)
    X = df[['dim1', 'dim2', 'density']]
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
    print(model.predict([[50, 50, 1]])) # Expect 0
    print(model.predict([[153, 9, 0.65]])) # Expect 0
    print(model.predict([[125, 8, 1]])) # Expect 0
    print(model.predict([[465, 2, 1]])) # Expect 0
    print(model.predict([[1000, 1, 100.0]])) # Expect 0
    print(model.predict([[1139, 1, 100.0]])) # Expect 0
    print(model.predict([[150, 150, 100]])) # Expect 1
    print(model.predict([[1,9996,100.0]])) # Expect 1
    print(model.predict([[9996,1,100.0]])) # Expect 1

    # print(X_train)
    # print(y_train)
    # print(df['target'].value_counts())
