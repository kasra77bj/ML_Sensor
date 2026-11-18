"""
Train the three StrenML models and export them for the web app.

Usage:
    python training/train_and_export.py                 # uses training/data.csv if present
    python training/train_and_export.py path/to/data.csv

The CSV must contain these columns (canonical units):
    age_days               1, 3, 5 or 7
    curing                 1 = curing applied, 0 = no curing
    resistivity_ohm_m      electrical resistivity in ohm.m
    permittivity           relative permittivity (dimensionless)
    temperature_c          average internal temperature in degrees C
    flexural_strength_mpa  target: flexural strength in MPa

If no CSV is found, the script trains on SYNTHETIC data and marks the exported
models as demo models, so the web app shows a warning banner. Replace them with
the paper's dataset before using the predictions for anything real.

Output: models/models.js (loaded by index.html; works from file:// and GitHub Pages).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

ROOT = Path(__file__).resolve().parent.parent
FEATURES = ["age_days", "curing", "resistivity_ohm_m", "permittivity", "temperature_c"]
TARGET = "flexural_strength_mpa"


def synthetic_data(n=400, seed=0):
    """Plausible-looking but invented data, only so the app works end to end."""
    rng = np.random.default_rng(seed)
    age = rng.choice([1, 3, 5, 7], n)
    curing = rng.integers(0, 2, n)
    resistivity = 20 + 12 * np.log(age) + 8 * curing + rng.normal(0, 4, n)
    permittivity = 60 - 5 * np.log(age) + rng.normal(0, 4, n)
    temperature = rng.normal(23, 3, n)
    strength = (2.0 + 1.6 * np.log(age) + 0.6 * curing + 0.04 * resistivity
                - 0.01 * permittivity + 0.02 * (temperature - 23) + rng.normal(0, 0.3, n))
    return pd.DataFrame({
        "age_days": age, "curing": curing,
        "resistivity_ohm_m": np.clip(resistivity, 1, None),
        "permittivity": np.clip(permittivity, 1, None),
        "temperature_c": temperature, TARGET: strength,
    })


def scaler_json(s):
    return {"mean": s.mean_.tolist(), "scale": s.scale_.tolist()}


def main():
    csv = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "training" / "data.csv"
    demo = not csv.exists()
    df = synthetic_data() if demo else pd.read_csv(csv)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        sys.exit(f"CSV is missing columns: {missing}")
    print("Using", "SYNTHETIC demo data" if demo else csv)

    X = df[FEATURES].to_numpy(float)
    y = df[TARGET].to_numpy(float)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    xs = StandardScaler().fit(X_tr)
    ys = StandardScaler().fit(y_tr.reshape(-1, 1))
    Xtr, Xte = xs.transform(X_tr), xs.transform(X_te)
    ytr = ys.transform(y_tr.reshape(-1, 1)).ravel()

    def unscale(p):
        return ys.inverse_transform(p.reshape(-1, 1)).ravel()

    out = {"demo": demo, "features": FEATURES, "x_scaler": scaler_json(xs),
           "y_scaler": {"mean": float(ys.mean_[0]), "scale": float(ys.scale_[0])},
           "models": {}}

    # --- Linear Regression
    lr = LinearRegression().fit(Xtr, ytr)
    out["models"]["linear"] = {"type": "linear", "coef": lr.coef_.tolist(),
                               "intercept": float(lr.intercept_)}

    # --- SVR (RBF)
    svr = SVR(kernel="rbf", C=10.0, epsilon=0.05, gamma="scale").fit(Xtr, ytr)
    out["models"]["svr"] = {"type": "svr_rbf", "gamma": float(svr._gamma),
                            "support_vectors": svr.support_vectors_.tolist(),
                            "dual_coef": svr.dual_coef_.ravel().tolist(),
                            "intercept": float(svr.intercept_[0])}

    # --- ANN (MLP)
    mlp = MLPRegressor(hidden_layer_sizes=(32, 16), activation="relu",
                       max_iter=3000, random_state=42).fit(Xtr, ytr)
    out["models"]["mlp"] = {"type": "mlp", "activation": mlp.activation,
                            "weights": [w.tolist() for w in mlp.coefs_],
                            "biases": [b.tolist() for b in mlp.intercepts_]}

    for name, m in [("linear", lr), ("svr", svr), ("mlp", mlp)]:
        p = unscale(m.predict(Xte))
        r2, mae = r2_score(y_te, p), mean_absolute_error(y_te, p)
        out["models"][name]["metrics"] = {"r2": round(r2, 4), "mae": round(mae, 4)}
        print(f"{name:7s} R2={r2:.3f}  MAE={mae:.3f} MPa")

    # Reference predictions so the JS implementation can be checked against sklearn.
    out["check"] = {"x": X_te[:3].tolist(),
                    "y": {n: unscale(m.predict(Xte[:3])).tolist()
                          for n, m in [("linear", lr), ("svr", svr), ("mlp", mlp)]}}

    dest = ROOT / "models" / "models.js"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text("// Generated by training/train_and_export.py. Do not edit by hand.\n"
                    "window.STRENML_MODELS = " + json.dumps(out) + ";\n")
    print("Wrote", dest.relative_to(ROOT))


if __name__ == "__main__":
    main()
