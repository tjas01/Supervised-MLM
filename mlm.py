# ================================
# 0) Imports & config
# ================================
from __future__ import annotations
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

# Keras (only needed for the NN block)
try:
    import tensorflow as tf
    from tensorflow import keras
except Exception:
    tf, keras = None, None  # handled inside run_neural_network

DATA_DIR  = Path(r"C:\GitHub")  # change if needed
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV  = DATA_DIR / "test.csv"
TARGET    = "SalePrice"
RANDOM_STATE = 42


# ================================
# 1) Load originals (df_train / df_test)
# ================================
def load_data(train_path: Path, test_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_train = pd.read_csv(train_path)
    df_test  = pd.read_csv(test_path)
    return df_train, df_test


# ================================
# 2) Make df0_* (drop identifiers)
# ================================
def make_df0(df_train: pd.DataFrame, df_test: pd.DataFrame, drop_cols=None):
    drop_cols = drop_cols or ["Id"]
    df0_train = df_train.drop(columns=drop_cols, errors="ignore").copy()
    df0_test  = df_test.drop(columns=drop_cols,  errors="ignore").copy()
    return df0_train, df0_test


# ================================
# 3) Split X0 / y0 from TRAIN only
# ================================
def split_Xy(df0_train: pd.DataFrame, target: str = TARGET):
    y0 = df0_train[target].copy()
    X0 = df0_train.drop(columns=[target])
    return X0, y0


# ================================
# 4) Preprocessors (no leakage)
# - Linear/NN: median impute + StandardScaler (numerics) + OHE (categoricals)
# - Trees (RF/GBM): median impute + OHE (no scaler)
# ================================
def build_preprocessor_linear(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[("num", numeric, num_cols), ("cat", categorical, cat_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )

def build_preprocessor_tree(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),  # no scaler for trees
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[("num", numeric, num_cols), ("cat", categorical, cat_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


# =========================================================
# ===============   L I N E A R   R E G R E S S I O N   ===
# =========================================================
def run_linear_regression(X0: pd.DataFrame, y0: pd.Series, seed: int = RANDOM_STATE) -> Dict[str, float | Pipeline]:
    Xtr0, Xva0, ytr0, yva0 = train_test_split(X0, y0, test_size=0.2, random_state=seed)
    pre = build_preprocessor_linear(Xtr0)
    lr  = LinearRegression()
    pipe = Pipeline([("prep", pre), ("model", lr)])
    pipe.fit(Xtr0, ytr0)

    preds = pipe.predict(Xva0)
    rmse = float(np.sqrt(mean_squared_error(yva0, preds)))
    mape = float(mean_absolute_percentage_error(yva0, preds) * 100.0)
    r2   = float(r2_score(yva0, preds))
    return {"rmse": rmse, "mape_pct": mape, "r2": r2, "estimator": pipe}


# =========================================================
# ==================   R A N D O M   F O R E S T   ========
# =========================================================
def run_random_forest(X0: pd.DataFrame, y0: pd.Series, seed: int = RANDOM_STATE) -> Dict[str, float | Pipeline]:
    Xtr0, Xva0, ytr0, yva0 = train_test_split(X0, y0, test_size=0.2, random_state=seed)
    pre = build_preprocessor_tree(Xtr0)  # trees: no scaler
    rf  = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=seed)
    pipe = Pipeline([("prep", pre), ("model", rf)])
    pipe.fit(Xtr0, ytr0)

    preds = pipe.predict(Xva0)
    rmse = float(np.sqrt(mean_squared_error(yva0, preds)))
    mape = float(mean_absolute_percentage_error(yva0, preds) * 100.0)
    r2   = float(r2_score(yva0, preds))
    return {"rmse": rmse, "mape_pct": mape, "r2": r2, "estimator": pipe}


# =========================================================
# ================   G R A D I E N T   B O O S T   ========
# =========================================================
def run_gradient_boosting(X0: pd.DataFrame, y0: pd.Series, seed: int = RANDOM_STATE) -> Dict[str, float | Pipeline]:
    Xtr0, Xva0, ytr0, yva0 = train_test_split(X0, y0, test_size=0.2, random_state=seed)
    pre = build_preprocessor_tree(Xtr0)  # trees: no scaler
    gb  = HistGradientBoostingRegressor(random_state=seed)
    pipe = Pipeline([("prep", pre), ("model", gb)])
    pipe.fit(Xtr0, ytr0)

    preds = pipe.predict(Xva0)
    rmse = float(np.sqrt(mean_squared_error(yva0, preds)))
    mape = float(mean_absolute_percentage_error(yva0, preds) * 100.0)
    r2   = float(r2_score(yva0, preds))
    return {"rmse": rmse, "mape_pct": mape, "r2": r2, "estimator": pipe}


# =========================================================
# =================   N E U R A L   N E T W O R K   =======
# =========================================================
def build_mlp(input_dim: int):
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def run_neural_network(X0: pd.DataFrame, y0: pd.Series, seed: int = RANDOM_STATE) -> Dict[str, float | tuple]:
    if keras is None:
        raise ImportError("TensorFlow/Keras not installed. Install with: pip install tensorflow")

    # split
    Xtr0, Xva0, ytr0, yva0 = train_test_split(X0, y0, test_size=0.2, random_state=seed)

    # preprocess (same as linear: impute + scale + OHE)
    pre = build_preprocessor_linear(Xtr0)
    XtrT = pre.fit_transform(Xtr0)
    XvaT = pre.transform(Xva0)

    # NN
    model = build_mlp(input_dim=XtrT.shape[1])
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
    model.fit(XtrT, ytr0.values, validation_data=(XvaT, yva0.values), epochs=500, batch_size=64, callbacks=[es], verbose=0)

    preds = model.predict(XvaT, verbose=0).ravel()
    rmse = float(np.sqrt(mean_squared_error(yva0.values, preds)))
    mape = float(mean_absolute_percentage_error(yva0.values, preds) * 100.0)
    r2   = float(r2_score(yva0.values, preds))

    # return the (preprocessor, model) pair for later full-fit/predict
    return {"rmse": rmse, "mape_pct": mape, "r2": r2, "estimator": (pre, model)}


# ================================
# 5) Script entry
# ================================
def main():
    df_train, df_test = load_data(TRAIN_CSV, TEST_CSV)
    df0_train, df0_test = make_df0(df_train, df_test, drop_cols=["Id"])
    X0, y0 = split_Xy(df0_train, target=TARGET)

    print("\n======== LINEAR REGRESSION (scaled numerics) ========")
    res_lr = run_linear_regression(X0, y0)
    print(f"R² = {res_lr['r2']:.4f} | RMSE = {res_lr['rmse']:,.2f} | MAPE = {res_lr['mape_pct']:.2f}%")

    print("\n=========== RANDOM FOREST (no scaling) ============")
    res_rf = run_random_forest(X0, y0)
    print(f"R² = {res_rf['r2']:.4f} | RMSE = {res_rf['rmse']:,.2f} | MAPE = {res_rf['mape_pct']:.2f}%")

    print("\n========= GRADIENT BOOSTING (no scaling) =========")
    res_gb = run_gradient_boosting(X0, y0)
    print(f"R² = {res_gb['r2']:.4f} | RMSE = {res_gb['rmse']:,.2f} | MAPE = {res_gb['mape_pct']:.2f}%")

    print("\n============= NEURAL NETWORK (scaled) =============")
    res_nn = run_neural_network(X0, y0)
    print(f"R² = {res_nn['r2']:.4f} | RMSE = {res_nn['rmse']:,.2f} | MAPE = {res_nn['mape_pct']:.2f}%")

    # (optional) example of predicting with the best model on df0_test:
    # best = min([res_lr, res_rf, res_gb, res_nn], key=lambda d: d["rmse"])
    # if isinstance(best["estimator"], Pipeline):
    #     final = best["estimator"].fit(X0, y0)
    #     preds_test = final.predict(df0_test)
    # else:
    #     pre, nn = best["estimator"]
    #     Xt = pre.fit_transform(X0); nn.fit(Xt, y0.values, epochs=10, verbose=0)
    #     preds_test = nn.predict(pre.transform(df0_test), verbose=0).ravel()
    # print("test preds preview:", preds_test[:5])


if __name__ == "__main__":
    main()
# ================================