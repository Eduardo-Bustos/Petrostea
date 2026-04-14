from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score


# =========================
# CONFIG
# =========================

@dataclass
class DataSourceConfig:
    dataset_master_path: str = None
    merged_master_path: str = None
    state_metrics_path: str = None
    transmission_window_path: str = None
    event_paths: list = field(default_factory=list)


@dataclass
class ValidationV4Config:
    date_col: str = "date"
    event_date_col: str = "event_date"
    event_flag_col: str = "event_flag"
    target_col: str = "binding_future"

    feature_cols_full: tuple = (
        "Stress", "SG_tau", "Theta_eff",
        "FAI", "R_star", "ALT_EXEC_emp", "CP_eff"
    )

    feature_cols_baseline: tuple = ("Stress", "SG_tau")

    rolling_window: int = 80
    rolling_step: int = 5

    threshold_grid_size: int = 200

    output_dir: str = "outputs/validation_v4"


# =========================
# LOADERS
# =========================

def load_table(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix == ".csv":
        return pd.read_csv(path)

    if path.suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path)

    raise ValueError("Unsupported file")


def normalize(df):
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    return df


# =========================
# PANEL BUILDER
# =========================

def build_master_panel(ds_cfg, cfg):
    dfs = []

    for path in [
        ds_cfg.dataset_master_path,
        ds_cfg.merged_master_path,
        ds_cfg.state_metrics_path,
        ds_cfg.transmission_window_path,
    ]:
        if path:
            df = normalize(load_table(path))
            dfs.append(df)

    master = dfs[0]

    for df in dfs[1:]:
        master = master.merge(df, on=cfg.date_col, how="outer")

    master[cfg.date_col] = pd.to_datetime(master[cfg.date_col])
    master = master.sort_values(cfg.date_col)

    return master


# =========================
# FEATURE ENGINEERING
# =========================

def safe_z(x):
    x = pd.to_numeric(x, errors="coerce")
    if x.std() < 1e-9:
        return pd.Series(np.zeros(len(x)))
    return (x - x.mean()) / x.std()


def enrich_features(df, cfg):
    df["ALT_EXEC_emp"] = safe_z(df["SG_tau"]).clip(0)

    df["Theta_eff"] = df["SG_tau"] * (1 - 0.5 * df["ALT_EXEC_emp"])
    df["CP_eff"] = df["SG_tau"] * (1 + 0.5 * df["ALT_EXEC_emp"])

    df[cfg.target_col] = (
        df["Stress"].shift(-1) > df["Stress"].quantile(0.75)
    ).astype(int)

    return df


# =========================
# MODEL
# =========================

def fit_model(df, features, target):
    pipe = Pipeline([
        ("imputer", SimpleImputer()),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=3000))
    ])
    pipe.fit(df[features], df[target])
    return pipe


def rolling_validation(df, cfg):
    metrics = []

    for i in range(0, len(df) - cfg.rolling_window, cfg.rolling_step):
        train = df.iloc[i:i + cfg.rolling_window]
        test = df.iloc[i + cfg.rolling_window:i + cfg.rolling_window + cfg.rolling_step]

        model = fit_model(train, cfg.feature_cols_full, cfg.target_col)

        y = test[cfg.target_col]
        score = model.predict_proba(test[cfg.feature_cols_full])[:, 1]

        auc = roc_auc_score(y, score) if len(set(y)) > 1 else np.nan
        pred = (score > 0.5).astype(int)

        metrics.append({
            "auc": auc,
            "f1": f1_score(y, pred)
        })

    return pd.DataFrame(metrics)


# =========================
# THRESHOLD
# =========================

def optimize_threshold(y, score):
    best = {"f1": -1}

    for t in np.linspace(0, 1, 100):
        pred = (score >= t).astype(int)
        f1 = f1_score(y, pred)

        if f1 > best["f1"]:
            best = {"threshold": t, "f1": f1}

    return best


# =========================
# MAIN
# =========================

def run_validation_v4_bundle(ds_cfg, cfg):

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = build_master_panel(ds_cfg, cfg)
    df = enrich_features(df, cfg)
    df = df.dropna()

    rolling = rolling_validation(df, cfg)

    model = fit_model(df, cfg.feature_cols_full, cfg.target_col)
    score = model.predict_proba(df[cfg.feature_cols_full])[:, 1]

    threshold = optimize_threshold(df[cfg.target_col], score)

    df.to_csv(out / "panel.csv", index=False)
    rolling.to_csv(out / "rolling.csv", index=False)

    report = {
        "mean_auc": float(rolling["auc"].mean()),
        "mean_f1": float(rolling["f1"].mean()),
        "threshold": threshold
    }

    with open(out / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    return report
