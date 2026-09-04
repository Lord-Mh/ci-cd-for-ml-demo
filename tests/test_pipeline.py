import numpy as np
import pandas as pd
from src.preprocessing import fit_target_encoder, transform_target_encoder, impute_and_scale_data


def test_target_encoder():
    df = pd.DataFrame({"cat": ["A", "A", "B", "B"], "target": [1, 1, 0, 0]})
    maps = fit_target_encoder(df, ["cat"], "target")
    assert maps["cat"]["A"] == 1.0
    assert maps["cat"]["B"] == 0.0

    transformed = transform_target_encoder(df[["cat"]], ["cat"], maps, global_mean=0.5)
    assert transformed["cat"].tolist() == [1.0, 1.0, 0.0, 0.0]


def test_impute_and_scale():
    train = pd.DataFrame({"num": [1.0, np.nan, 3.0]})
    test = pd.DataFrame({"num": [2.0, np.nan]})
    train_proc, test_proc = impute_and_scale_data(train, test)

    assert not train_proc.isnull().values.any()
    assert not test_proc.isnull().values.any()
