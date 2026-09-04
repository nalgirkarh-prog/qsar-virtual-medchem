import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_predict, LeaveOneOut, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from medchem.descriptors import compute_all_descriptors, compute_descriptors_batch

DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
LEGACY_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trained_qsar_model.pkl")

DEFAULT_FEATURES = [
    "MW", "LogP", "TPSA", "HBD", "HBA", "RotatableBonds", "RingCount",
    "AromaticRings", "FractionCSP3", "MR", "HeavyAtomCount",
    "NumHeteroatoms", "LogS_ESOL", "QED", "LabuteASA", "BertzCT"
]


def prepare_training_data(
    csv_path: str,
    smiles_col: str = "SMILES",
    activity_col: str = "Activity",
    feature_names: list[str] | None = None
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Reads CSV, computes descriptors for all compounds, returns (X_df, y_array, feature_names_used)."""
    df = pd.read_csv(csv_path)
    if smiles_col not in df.columns or activity_col not in df.columns:
        raise ValueError(f"Columns {smiles_col} and {activity_col} must be present in CSV {csv_path}")

    # Drop any null SMILES or activity
    df = df.dropna(subset=[smiles_col, activity_col]).copy()
    if df.empty:
        raise ValueError("CSV contains no valid rows with SMILES and Activity.")

    records = []
    y_vals = []
    for _, row in df.iterrows():
        smi = str(row[smiles_col]).strip()
        desc = compute_all_descriptors(smi)
        if desc is not None:
            records.append(desc)
            y_vals.append(float(row[activity_col]))

    if not records:
        raise ValueError("Could not parse any SMILES from the training data.")

    desc_df = pd.DataFrame(records)
    y = np.array(y_vals)

    if feature_names is None:
        target_features = DEFAULT_FEATURES
    else:
        target_features = feature_names

    available_features = [f for f in target_features if f in desc_df.columns]
    if not available_features:
        # Fallback to all numeric columns in desc_df
        available_features = desc_df.select_dtypes(include=[np.number]).columns.tolist()

    # Filter features with variance > 0
    X = desc_df[available_features]
    variances = X.var()
    final_features = variances[variances > 0].index.tolist()

    # If only 2 or 3 samples and all features have variance, pick the most relevant
    # to avoid extreme n_features >> n_samples
    if len(y) < 5 and len(final_features) > len(y):
        # Keep top features like MW, LogP, TPSA, QED
        priority = ["MW", "LogP", "TPSA", "MR", "QED", "HeavyAtomCount"]
        reduced = [f for f in priority if f in final_features]
        if reduced:
            final_features = reduced[:min(len(y), len(reduced))]

    if not final_features:
        # If no variance, take the first available numeric feature
        final_features = available_features[:1]

    X_final = desc_df[final_features]
    return X_final, y, final_features


def train_model(
    csv_path: str,
    model_type: str = "random_forest",
    smiles_col: str = "SMILES",
    activity_col: str = "Activity",
    feature_names: list[str] | None = None,
    save_path: str | None = None
) -> dict:
    """Trains a QSAR model with cross-validation and feature importance."""
    X_df, y, features = prepare_training_data(csv_path, smiles_col, activity_col, feature_names)

    n_samples = len(y)
    n_features = len(features)

    if model_type == "random_forest":
        if n_samples < 10:
            regressor = RandomForestRegressor(n_estimators=50, max_depth=2, random_state=42)
        else:
            regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == "linear":
        regressor = LinearRegression()
    elif model_type == "ridge":
        regressor = Ridge(alpha=1.0)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", regressor)
    ])

    # Cross-validation
    cv_r2, cv_rmse, cv_mae = 0.0, 0.0, 0.0
    if n_samples >= 3:
        cv = LeaveOneOut() if n_samples < 10 else KFold(n_splits=5, shuffle=True, random_state=42)
        try:
            y_pred = cross_val_predict(pipeline, X_df, y, cv=cv)
            cv_r2 = float(r2_score(y, y_pred))
            cv_rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
            cv_mae = float(mean_absolute_error(y, y_pred))
        except Exception:
            pass

    # Fit final model on full dataset
    pipeline.fit(X_df, y)

    # Feature importance
    feature_importance = None
    if model_type == "random_forest":
        try:
            importances = pipeline.named_steps["model"].feature_importances_
            feature_importance = dict(zip(features, [float(v) for v in importances]))
        except Exception:
            pass
    elif hasattr(pipeline.named_steps["model"], "coef_"):
        try:
            coefs = np.abs(pipeline.named_steps["model"].coef_)
            feature_importance = dict(zip(features, [float(v) for v in coefs]))
        except Exception:
            pass

    # Validation diagnostics (OECD / Golbraikh & Tropsha criteria)
    warnings_list = []
    if n_samples < 10:
        warnings_list.append(
            f"Critically small dataset (N={n_samples}). Machine learning QSAR models require at least 20-30+ diverse compounds for generalizable structure-activity learning."
        )
    elif n_samples < 20:
        warnings_list.append(
            f"Small dataset (N={n_samples}). Recommended minimum is 20-30+ compounds for reliable cross-validation."
        )

    # Topliss / rule of thumb: ratio of samples to features should be >= 5:1
    ratio = n_samples / max(1, n_features)
    if ratio < 5.0:
        warnings_list.append(
            f"High risk of chance correlation / overfitting: ratio of compounds to descriptors is {ratio:.1f}:1 (N={n_samples}, features={n_features}). The recommended guideline is >= 5:1 (Topliss operational criteria)."
        )

    if cv_r2 < 0:
        warnings_list.append(
            f"Statistical invalidity: CV R² is negative ({cv_r2:.4f}), meaning the model performs worse than predicting the mean activity baseline. Predictions from this model should NOT be used for decision making."
        )
    elif cv_r2 < 0.5:
        warnings_list.append(
            f"Low predictive power: CV R² ({cv_r2:.4f}) is below standard QSAR acceptability criteria (q² >= 0.50, Tropsha/Golbraikh threshold)."
        )

    model_dict = {
        "model": pipeline,
        "features": features,
        "cv_r2": cv_r2,
        "cv_rmse": cv_rmse,
        "cv_mae": cv_mae,
        "feature_importance": feature_importance,
        "n_compounds": n_samples,
        "model_type": model_type,
        "warnings": warnings_list
    }

    # Save to primary destination
    if save_path is None:
        os.makedirs(DEFAULT_MODEL_DIR, exist_ok=True)
        save_path = os.path.join(DEFAULT_MODEL_DIR, "qsar_model.pkl")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    joblib.dump(model_dict, save_path)

    # Also save to legacy path for backward compatibility
    try:
        joblib.dump(model_dict, LEGACY_MODEL_PATH)
    except Exception:
        pass

    return model_dict


def load_model(model_path: str | None = None) -> dict:
    """Loads a saved model dict, gracefully adapting legacy models."""
    if model_path is None:
        model_path = os.path.join(DEFAULT_MODEL_DIR, "qsar_model.pkl")

    target_path = None
    if os.path.exists(model_path):
        target_path = model_path
    elif os.path.exists(LEGACY_MODEL_PATH):
        target_path = LEGACY_MODEL_PATH
    else:
        raise FileNotFoundError(f"Model not found at {model_path} or {LEGACY_MODEL_PATH}")

    loaded = joblib.load(target_path)

    # Normalize if loaded is a bare sklearn model (legacy format)
    if not isinstance(loaded, dict):
        pipeline = loaded
        # Check if legacy model had 2 features
        features = ["MW", "LogP"]
        return {
            "model": pipeline,
            "features": features,
            "cv_r2": 0.0,
            "cv_rmse": 0.0,
            "cv_mae": 0.0,
            "feature_importance": None,
            "n_compounds": 3,
            "model_type": "legacy_linear"
        }

    return loaded


def predict_activity(
    smiles_list: list[str],
    model_dict: dict | None = None,
    model_path: str | None = None
) -> pd.DataFrame:
    """Predicts activity for a list of SMILES."""
    if model_dict is None:
        model_dict = load_model(model_path)

    pipeline = model_dict["model"]
    features = model_dict["features"]

    results = []
    for smiles in smiles_list:
        desc = compute_all_descriptors(smiles)
        row = {"SMILES": smiles}

        if desc is None:
            row["Predicted_Activity"] = None
            row["Error"] = "Invalid SMILES"
            results.append(row)
            continue

        row.update(desc)

        # Build feature vector
        vector = []
        valid = True
        for f in features:
            val = desc.get(f)
            # Map legacy column names if needed
            if val is None and f.startswith("Computed_"):
                alt_f = f.replace("Computed_", "")
                val = desc.get(alt_f)
            if val is None:
                valid = False
                break
            vector.append(val)

        if not valid:
            row["Predicted_Activity"] = None
            row["Error"] = "Missing required features"
        else:
            try:
                feat_df = pd.DataFrame([vector], columns=features)
                pred = pipeline.predict(feat_df)[0]
                row["Predicted_Activity"] = float(pred)
                row["Error"] = None
            except Exception as e:
                row["Predicted_Activity"] = None
                row["Error"] = str(e)

        results.append(row)

    return pd.DataFrame(results)


def retrain_with_new_data(new_csv_path: str, existing_model_path: str | None = None, **kwargs) -> dict:
    """Convenience function to retrain model."""
    return train_model(csv_path=new_csv_path, **kwargs)


if __name__ == "__main__":
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data.csv")
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]

    if not os.path.exists(csv_path):
        print(f"Error: Data file {csv_path} not found.")
        sys.exit(1)

    print(f"Training QSAR model using data from {csv_path}...")
    model_info = train_model(csv_path)
    print("Model Training Successful!")
    print("-" * 30)
    print(f"Model Type: {model_info['model_type']}")
    print(f"Compounds used: {model_info['n_compounds']}")
    print(f"Features used: {model_info['features']}")
    if model_info.get("feature_importance"):
        print("Feature importances:")
        for k, v in model_info["feature_importance"].items():
            print(f"  {k}: {v:.4f}")
