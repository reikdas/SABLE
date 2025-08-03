import os
import pathlib
import numpy as np

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def predict_speedup(block_sx: int, block_sy: int, calc_density: float) -> float:
    """
    Predict the speedup for a block based on its dimensions and density.
    
    Args:
        block_sx: Block size in x dimension
        block_sy: Block size in y dimension  
        calc_density: Calculated density percentage
    
    Returns:
        float: Predicted speedup value (CSR_time / sable_time)
    """
    # Load the model
    model = joblib.load(os.path.join(BASEPATH, "models", "speedup_predictor_spmv.pkl"))
    
    # Calculate nnz and logarithmic features for prediction
    nnz = (calc_density / 100) * block_sx * block_sy
    log_dim1 = np.log(block_sx)
    log_dim2 = np.log(block_sy)
    log_nnz = np.log(nnz + 1)  # +1 to avoid log(0)
    log_density = np.log(calc_density + 1)  # +1 to avoid log(0)
    
    # Calculate additional features for better extrapolation
    dim_product = block_sx * block_sy
    log_dim_product = np.log(dim_product)
    density_nnz_ratio = calc_density * nnz / 100
    dim_ratio = block_sx / block_sy
    
    # Model predicts the actual speedup value
    return model.predict([[block_sx, block_sy, calc_density, nnz, log_dim1, log_dim2, log_nnz, log_density, 
                          dim_product, log_dim_product, density_nnz_ratio, dim_ratio]])[0]

def is_dense_block(block_sx: int, block_sy: int, calc_density: float) -> bool:
    """
    Determine if a block should be kept dense based on predicted speedup.
    
    Args:
        block_sx: Block size in x dimension
        block_sy: Block size in y dimension  
        calc_density: Calculated density percentage
    
    Returns:
        bool: True if block should be kept dense, False if it should be unrolled
    """
    predicted_speedup = predict_speedup(block_sx, block_sy, calc_density)
    return predicted_speedup >= SPEEDUP_THRESH

# from src.consts import SPEEDUP_THRESH
SPEEDUP_THRESH=1.3

FILEPATH=pathlib.Path(__file__).resolve().parent
BASEPATH=os.path.join(FILEPATH.parent)

if __name__ == "__main__":
    # Load data into a DataFrame
    df = pd.read_csv(os.path.join(FILEPATH,"threshold_results.csv"))

    # Filter for rows where dim1=1, dim2=1, and nnz=0
    filtered_df = df[(df['dim1'] == 1) & (df['dim2'] == 1) & (df['nnz'] == 0)]

    df['density'] = 100 - df['perc_zeros']
    
    # Add logarithmic features to help with extrapolation
    df['log_dim1'] = np.log(df['dim1'])
    df['log_dim2'] = np.log(df['dim2'])
    df['log_nnz'] = np.log(df['nnz'] + 1)  # +1 to avoid log(0)
    df['log_density'] = np.log(df['density'] + 1)  # +1 to avoid log(0)
    
    # Add additional features for better extrapolation
    df['dim_product'] = df['dim1'] * df['dim2']  # Total matrix size
    df['log_dim_product'] = np.log(df['dim_product'])
    df['density_nnz_ratio'] = df['density'] * df['nnz'] / 100  # Weighted feature
    df['dim_ratio'] = df['dim1'] / df['dim2']  # Aspect ratio
    
    # df['size'] = df['dim1'] * df['dim2']

    # Calculate the target variable as actual speedup:
    df['target'] = df['CSR_time'] / df['sable_time']

    # Filter out invalid or extreme values
    df = df[df['target'] > 0]  # Remove negative or zero speedups

    df.loc[df['density'] <= 25, 'target'] = 0

    # df.loc[(df['dim1'] < 8) & (df['dim2'] < 8), 'target'] = 0
    df.loc[(df['nnz'] < 800), 'target'] = 0

    df.loc[(df['dim2'] == 1) & (df['dim1'] < 1300), 'target'] = 0

    # Set target for size < 500 to 0
    # df.loc[(df['dim1'] * df['dim2']) < 1000, 'target'] = 0

    # Define features (X) and target (y)
    X = df[['dim1', 'dim2', 'density', 'nnz', 'log_dim1', 'log_dim2', 'log_nnz', 'log_density', 
            'dim_product', 'log_dim_product', 'density_nnz_ratio', 'dim_ratio']]
    y = df['target']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest regressor for speedup prediction
    model = RandomForestRegressor(
        n_estimators=200,  # More trees for better generalization
        max_depth=10,       # Limit depth to prevent overfitting
        min_samples_split=5,  # Require more samples to split
        min_samples_leaf=2,   # Require more samples in leaves
        random_state=42
    )
    model.fit(X_train.values, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R² Score: {r2}")
    print(f"Mean Absolute Error: {np.mean(np.abs(y_test - y_pred))}")

    model_filename = os.path.join(BASEPATH, "models", "speedup_predictor_spmv.pkl")
    joblib.dump(model, model_filename)

    # Test the predict_speedup function
    print("Speedup predictions:")
    print(f"Block (50, 50, 1%): {predict_speedup(50, 50, 1):.3f}")
    print(f"Block (153, 9, 0.65%): {predict_speedup(153, 9, 0.65):.3f}")
    print(f"Block (125, 8, 1%): {predict_speedup(125, 8, 1):.3f}")
    print(f"Block (465, 2, 1%): {predict_speedup(465, 2, 1):.3f}")
    print(f"Block (1000, 1, 100%): {predict_speedup(1000, 1, 100.0):.3f}")
    print(f"Block (1139, 1, 100%): {predict_speedup(1139, 1, 100.0):.3f}")
    print(f"Block (150, 150, 100%): {predict_speedup(150, 150, 100):.3f}")
    print(f"Block (1, 9996, 100%): {predict_speedup(1, 9996, 100.0):.3f}")
    print(f"Block (9996, 1, 100%): {predict_speedup(9996, 1, 100.0):.3f}")
    print(f"Block (10000, 10000, 100%): {predict_speedup(10000, 10000, 100.0):.3f}")
    print(f"Block (5000, 5000, 95%): {predict_speedup(5000, 5000, 95.0):.3f}")
    print(f"Block (10000, 10000, 92%): {predict_speedup(10000, 10000, 92.0):.3f}")
    print(f"Block (1000, 1000, 85%): {predict_speedup(1000, 1000, 85.0):.3f}")

    # Test the is_dense_block function
    print("\nDense block decisions:")
    print(is_dense_block(50, 50, 1))  # Expect 0
    print(is_dense_block(153, 9, 0.65))  # Expect 0
    print(is_dense_block(125, 8, 1))  # Expect 0
    print(is_dense_block(465, 2, 1))  # Expect 0
    print(is_dense_block(1000, 1, 100.0))  # Expect 0
    print(is_dense_block(1139, 1, 100.0))  # Expect 0
    print(is_dense_block(150, 150, 100))  # Expect 1
    print(is_dense_block(1, 9996, 100.0))  # Expect 1
    print(is_dense_block(9996, 1, 100.0))  # Expect 1
    print(is_dense_block(10000, 10000, 100.0))  # Expect 1
    print(is_dense_block(5000, 5000, 95.0))  # Expect 1
    print(is_dense_block(10000, 10000, 92.0))  # Expect 1
    print(is_dense_block(1000, 1000, 85.0))  # Expect 1

    # print(X_train)
    # print(y_train)
    # print(df['target'].value_counts())
