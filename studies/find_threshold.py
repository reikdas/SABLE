import os
import pathlib
import numpy as np

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def predict_speedup(block_sx: int, block_sy: int, calc_density: float, op: str) -> float:
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
    if op.lower() == "spmv":
        model = joblib.load(os.path.join(BASEPATH, "models", "speedup_predictor_spmv.pkl"))
    elif op.lower() == "spmm":
        model = joblib.load(os.path.join(BASEPATH, "models", "speedup_predictor_spmm.pkl"))
    else:
        raise Exception("Unknown operation")

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

def is_dense_block(block_sx: int, block_sy: int, calc_density: float, op: str) -> bool:
    """
    Determine if a block should be kept dense based on simple heuristics.
    
    Args:
        block_sx: Block size in x dimension
        block_sy: Block size in y dimension  
        calc_density: Calculated density percentage
        op: Operation type (unused, kept for API compatibility)
    
    Returns:
        bool: True if block should be kept dense, False if it should be unrolled
    """
    # Filter out blocks with density < 50%
    if calc_density < 50:
        return False
    
    # Calculate block characteristics
    nnz = (calc_density / 100) * block_sx * block_sy
    is_1d = (block_sy == 1 or block_sx == 1)
    max_dim = max(block_sx, block_sy)
    dim_product = block_sx * block_sy
    
    # Filter 1D vectors: length < 2500 (typically have time_ns < 1000)
    if is_1d and max_dim < 2500:
        return False
    
    # Filter 2D blocks: small size AND small nnz (typically have time_ns < 1000)
    if not is_1d and dim_product < 3000 and nnz < 3500:
        return False
    
    # Keep block dense if it passes all filters
    return True

# from src.consts import SPEEDUP_THRESH
SPEEDUP_THRESH=1.15

FILEPATH=pathlib.Path(__file__).resolve().parent
BASEPATH=os.path.join(FILEPATH.parent)

def analyze_thresholds(op: str):
    # Load data into a DataFrame
    df = pd.read_csv(os.path.join(FILEPATH, f"threshold_results_{op.lower()}.csv"))

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
    df['target'] = df['sparse'] / df['dense']

    # Filter out invalid or extreme values
    df = df[df['target'] > 0]  # Remove negative or zero speedups

    df.loc[df['density'] <= 25, 'target'] = 0

    # The following heuristics previously forced many targets to zero which
    # prevented the model from learning meaningful speedups for large dense
    # blocks. Relax them so the model can generalize. If you still want to
    # mask very small matrices, change the thresholds below instead of hard
    # zeroing everything.
    # df.loc[(df['dim1'] < 8) & (df['dim2'] < 8), 'target'] = 0
    # df.loc[(df['nnz'] < 800), 'target'] = 0
    # df.loc[(df['dim2'] == 1) & (df['dim1'] < 1300), 'target'] = 0

    # Optionally, drop rows with non-positive targets or extreme outliers, but
    # avoid mass-zeroing which leads to the degenerate model seen earlier.
    # Heuristic: treat very small blocks as not worth keeping dense, except
    # for very tall/long vectors where dense may still be beneficial
    # (e.g., dim2==1 and dim1 large, or dim1==1 and dim2 large).
    df['dim_product'] = df['dim1'] * df['dim2']
    small_block_mask = (
        (df['dim_product'] < 2000)
        & ~((df['dim2'] == 1) & (df['dim1'] >= 1000))
        & ~((df['dim1'] == 1) & (df['dim2'] >= 1000))
    )
    df.loc[small_block_mask, 'target'] = 0

    df = df[df['target'] > 0]

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

    model_filename = os.path.join(BASEPATH, "models", f"speedup_predictor_{op.lower()}.pkl")
    joblib.dump(model, model_filename)

    # Test the predict_speedup function
    print("Speedup predictions:")
    print(f"Block (2, 2, 100%): {predict_speedup(2, 2, 100, op):.3f}")
    print(f"Block (50, 50, 1%): {predict_speedup(50, 50, 1, op):.3f}")
    print(f"Block (153, 9, 0.65%): {predict_speedup(153, 9, 0.65, op):.3f}")
    print(f"Block (125, 8, 1%): {predict_speedup(125, 8, 1, op):.3f}")
    print(f"Block (465, 2, 1%): {predict_speedup(465, 2, 1, op):.3f}")
    print(f"Block (1000, 1, 100%): {predict_speedup(1000, 1, 100.0, op):.3f}")
    print(f"Block (1139, 1, 100%): {predict_speedup(1139, 1, 100.0, op):.3f}")
    print(f"Block (150, 150, 100%): {predict_speedup(150, 150, 100, op):.3f}")
    print(f"Block (1, 9996, 100%): {predict_speedup(1, 9996, 100.0, op):.3f}")
    print(f"Block (9996, 1, 100%): {predict_speedup(9996, 1, 100.0, op):.3f}")
    print(f"Block (10000, 10000, 100%): {predict_speedup(10000, 10000, 100.0, op):.3f}")
    print(f"Block (5000, 5000, 95%): {predict_speedup(5000, 5000, 95.0, op):.3f}")
    print(f"Block (10000, 10000, 92%): {predict_speedup(10000, 10000, 92.0, op):.3f}")
    print(f"Block (1000, 1000, 85%): {predict_speedup(1000, 1000, 85.0, op):.3f}")
    print(f"Block (10000, 1000, 80%): {predict_speedup(10000, 1000, 80.0, op):.3f}")
    print(f"Block (10000, 10000, 80%): {predict_speedup(10000, 10000, 80.0, op):.3f}")
    print(f"Block (3000, 1, 32%): {predict_speedup(3000, 1, 32.0, op):.3f}")
    print(f"Block (2000, 1, 57%): {predict_speedup(1828, 1, 57.0, op):.3f}")
    print(f"Block (256, 14, 43%): {predict_speedup(256, 14, 43.0, op):.3f}")

    # Test the is_dense_block function
    print("\nDense block decisions:")
    assert is_dense_block(2, 2, 100, op) == False
    assert is_dense_block(50, 50, 1, op) == False
    assert is_dense_block(153, 9, 0.65, op) == False
    assert is_dense_block(125, 8, 1, op) == False
    assert is_dense_block(465, 2, 1, op) == False
    assert is_dense_block(1, 9996, 100.0, op) == True
    assert is_dense_block(9996, 1, 100.0, op) == True
    assert is_dense_block(10000, 10000, 100.0, op) == True
    assert is_dense_block(5000, 5000, 95.0, op) == True
    assert is_dense_block(10000, 10000, 92.0, op) == True
    assert is_dense_block(1000, 1000, 85.0, op) == True
    assert is_dense_block(10000, 1000, 80.0, op) == True
    assert is_dense_block(10000, 10000, 80.0, op) == True
    assert is_dense_block(3000, 1, 32.0, op) == False
    assert is_dense_block(2000, 1, 57.0, op) == False
    assert is_dense_block(256, 14, 43.0, op) == False

    # print(X_train)
    # print(y_train)
    # print(df['target'].value_counts())

if __name__ == "__main__":
    analyze_thresholds("spmv")
