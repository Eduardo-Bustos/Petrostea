from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)


@dataclass
class DataSourceConfig:
    dataset_master_path: str | None = None
    merged_master_path: str | None = None
    state_metrics_path: str | None = None
    transmission_window_path: str | None = None
    event_paths: list[str] = field(default_factory=list)
    scenarios_path: str | None = None
    simulated_paths_path: str | None = None


@dataclass
class ValidationV4Config:
    date_col: str = "date"
    event_date_col: str = "event_date"
    event_flag_col: str = "event_flag"
    target_col: str = "binding
