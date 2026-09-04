from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def fit_target_encoder(
    df: pd.DataFrame, categorical_columns: List[str], target_column: str
) -> Dict[str, Dict]:
    """حساب متوسط الهدف لكل فئة من بيانات التدريب فقط."""
    encoding_maps = {}
    for col in categorical_columns:
        encoding_maps[col] = df.groupby(col)[target_column].mean().to_dict()
    return encoding_maps


def transform_target_encoder(
    df: pd.DataFrame, categorical_columns: List[str], encoding_maps: Dict[str, Dict], global_mean: float
) -> pd.DataFrame:
    """تطبيق الـ Target Encoding باستخدام الخرائط المحسوبة مسبقاً."""
    encoded_data = df.copy()
    for col in categorical_columns:
        encoded_data[col] = encoded_data[col].map(encoding_maps[col]).fillna(global_mean)
    return encoded_data


def impute_and_scale_data(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """عزل الـ fit على بيانات التدريب وتطبيق transform فقط على الاختبار."""
    columns = X_train.columns

    # 1. Imputation
    imputer = SimpleImputer(strategy="mean")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    # 2. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    return (
        pd.DataFrame(X_train_scaled, columns=columns, index=X_train.index),
        pd.DataFrame(X_test_scaled, columns=columns, index=X_test.index),
    )
