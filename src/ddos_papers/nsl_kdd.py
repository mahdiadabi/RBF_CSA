from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

FEATURES = [
"duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot",
"num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations",
"num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count",
"serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate",
"dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
"dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate"
]
CATEGORICAL = ["protocol_type", "service", "flag"]
DOS_LABELS = {"back", "land", "neptune", "pod", "smurf", "teardrop", "mailbomb", "apache2", "processtable", "udpstorm", "worm"}


def read_binary_nsl_kdd(path: str) -> tuple[pd.DataFrame, np.ndarray]:
    raw = pd.read_csv(path, header=None)
    if raw.shape[1] not in (42, 43):
        raise ValueError(f"Expected 42 or 43 columns, got {raw.shape[1]}")
    raw.columns = FEATURES + ["label"] + (["difficulty"] if raw.shape[1] == 43 else [])
    labels = raw["label"].astype(str).str.rstrip(".")
    keep = labels.eq("normal") | labels.isin(DOS_LABELS)
    return raw.loc[keep, FEATURES].reset_index(drop=True), labels.loc[keep].ne("normal").astype(int).to_numpy()


def build_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    numeric = [column for column in FEATURES if column not in CATEGORICAL]
    numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="mean")), ("scale", MinMaxScaler(feature_range=(-1, 1)))])
    categorical_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", numeric_pipe, numeric), ("cat", categorical_pipe, CATEGORICAL)], verbose_feature_names_out=False)


def original_feature_groups(preprocessor: ColumnTransformer) -> list[np.ndarray]:
    names = preprocessor.get_feature_names_out()
    groups = []
    for feature in FEATURES:
        groups.append(np.flatnonzero((names == feature) | np.char.startswith(names.astype(str), feature + "_")))
    return groups
