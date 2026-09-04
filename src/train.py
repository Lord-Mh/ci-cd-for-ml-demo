import json
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from preprocessing import fit_target_encoder, transform_target_encoder, impute_and_scale_data
from evaluate import calculate_metrics, save_confusion_matrix

TARGET_COLUMN = "target"
CATEGORICAL_COLUMNS = ["category_1"]


def run_pipeline():
    # 1. تحميل أو توليد بيانات تجريبية
    if not os.path.exists("data/sample_data.csv"):
        os.makedirs("data", exist_ok=True)
        # توليد بيانات مصغرة للتجربة
        data = pd.DataFrame({
            "category_1": ["A", "B", "A", "C", "B", "C", "A", "B"] * 25,
            "feature_num": [1.2, None, 3.4, 4.1, 2.8, None, 1.5, 3.0] * 25,
            "target": [0, 1, 0, 1, 1, 0, 0, 1] * 25,
        })
        data.to_csv("data/sample_data.csv", index=False)
    else:
        data = pd.read_csv("data/sample_data.csv")

    # 2. تقسيم البيانات Train / Test
    X = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1993, stratify=y
    )

    # 3. المعالجة (Target Encoding)
    global_mean = y_train.mean()
    train_df = X_train.copy()
    train_df[TARGET_COLUMN] = y_train
    encoding_maps = fit_target_encoder(train_df, CATEGORICAL_COLUMNS, TARGET_COLUMN)

    X_train_enc = transform_target_encoder(X_train, CATEGORICAL_COLUMNS, encoding_maps, global_mean)
    X_test_enc = transform_target_encoder(X_test, CATEGORICAL_COLUMNS, encoding_maps, global_mean)

    # 4. الـ Imputation والـ Scaling
    X_train_proc, X_test_proc = impute_and_scale_data(X_train_enc, X_test_enc)

    # 5. تدريب النموذج
    clf = RandomForestClassifier(max_depth=2, n_estimators=50, random_state=1993)
    clf.fit(X_train_proc, y_train)

    # 6. التقييم وحساب المقاييس
    y_pred = clf.predict(X_test_proc)
    metrics = calculate_metrics(y_test, y_pred)
    print("Test Metrics:", metrics)

    # 7. حفظ المخرجات (Metrics, Plot, Model)
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    save_confusion_matrix(clf, X_test_proc, y_test, output_path="artifacts/confusion_matrix.png")
    joblib.dump(clf, "artifacts/model.joblib")


if __name__ == "__main__":
    run_pipeline()
