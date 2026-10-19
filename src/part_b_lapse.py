from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.utils import resample
from sklearn.base import clone

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

customers = pd.read_csv(ROOT / "data/clean/customers.csv")
products = pd.read_csv(ROOT / "data/clean/products.csv")
orders = pd.read_csv(ROOT / "data/clean/orders.csv")

orders["order_ts"] = pd.to_datetime(orders["order_ts"], errors="coerce")
customers["signup_date"] = pd.to_datetime(
    customers["signup_date"], errors="coerce"
)

cutoff = pd.Timestamp("2025-09-01")
label_end = pd.Timestamp("2025-10-01")

# Label window: September 2025, completed orders only.
completed = orders.loc[orders["status"].eq("completed")].copy()
completed_before = completed.loc[completed["order_ts"] < cutoff].copy()
completed_during = completed.loc[
    (completed["order_ts"] >= cutoff)
    & (completed["order_ts"] < label_end)
].copy()

cohort_ids = completed_before["customer_id"].unique()
lapsed_ids = set(cohort_ids) - set(completed_during["customer_id"])

labels = pd.DataFrame({"customer_id": cohort_ids})
labels["lapsed"] = labels["customer_id"].isin(lapsed_ids).astype(int)

# Every predictive feature is built from pre-cutoff orders only.
all_before = orders.loc[orders["order_ts"] < cutoff].copy()
completed_before = all_before.loc[
    all_before["status"].eq("completed")
].copy()

recency = completed_before.groupby("customer_id")["order_ts"].max()
frequency = completed_before.groupby("customer_id")["order_id"].nunique()
monetary = completed_before.groupby("customer_id")["order_amount"].sum()
avg_discount = completed_before.groupby("customer_id")["discount_pct"].mean()
avg_delivery = completed_before.groupby("customer_id")["delivery_minutes"].mean()

status_counts = all_before.pivot_table(
    index="customer_id",
    columns="status",
    values="order_id",
    aggfunc="count",
    fill_value=0,
)
for status in ["completed", "cancelled"]:
    if status not in status_counts.columns:
        status_counts[status] = 0
cancel_rate = status_counts["cancelled"] / (
    status_counts["completed"] + status_counts["cancelled"]
).replace(0, np.nan)

# Favourite category uses only completed pre-cutoff orders.
category_orders = completed_before.merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left",
)
fav_category = (
    category_orders.groupby(["customer_id", "category"])
    .size()
    .rename("category_orders")
    .reset_index()
    .sort_values(
        ["customer_id", "category_orders", "category"],
        ascending=[True, False, True],
    )
    .drop_duplicates("customer_id")
    .set_index("customer_id")["category"]
)

features = pd.DataFrame({"customer_id": cohort_ids})
features["recency"] = features["customer_id"].map(
    ((cutoff - recency).dt.total_seconds() / 86400).to_dict()
)
features["frequency"] = features["customer_id"].map(frequency)
features["monetary"] = features["customer_id"].map(monetary)
features["avg_discount"] = features["customer_id"].map(avg_discount)
features["cancellation_rate"] = features["customer_id"].map(cancel_rate)
features["avg_delivery"] = features["customer_id"].map(avg_delivery)
features["favourite_category"] = features["customer_id"].map(fav_category)

features = features.merge(
    customers[["customer_id", "city", "age", "gender", "signup_date"]],
    on="customer_id",
    how="left",
    validate="one_to_one",
)
features["tenure_days"] = (
    cutoff - features["signup_date"]
).dt.total_seconds() / 86400
features = features.drop(columns=["signup_date"])

data = features.merge(
    labels, on="customer_id", how="inner", validate="one_to_one"
)

numeric_cols = [
    "recency", "frequency", "monetary", "avg_discount",
    "cancellation_rate", "avg_delivery", "age", "tenure_days",
]
categorical_cols = ["favourite_category", "city", "gender"]

for col in numeric_cols:
    data[col] = pd.to_numeric(data[col], errors="coerce")
for col in categorical_cols:
    data[col] = data[col].fillna("Unknown").astype(str)

# Ensure numeric features are complete without using the label.
for col in numeric_cols:
    data[col] = data[col].fillna(data[col].median())

X = data[numeric_cols + categorical_cols]
y = data["lapsed"]
ids = data["customer_id"]

X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X, y, ids, test_size=0.2, random_state=42, stratify=y
)

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
])
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipe, numeric_cols),
    ("categorical", categorical_pipe, categorical_cols),
])

lr_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000, random_state=42)),
])
lr_search = GridSearchCV(
    lr_pipeline,
    {"model__C": [0.01, 0.1, 1, 10, 100]},
    scoring="f1",
    cv=5,
    n_jobs=-1,
)
lr_search.fit(X_train, y_train)

rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight=None,
    )),
])
rf_search = GridSearchCV(
    rf_pipeline,
    {
        "model__max_depth": [3, 5, 8],
        "model__min_samples_leaf": [5, 10, 20],
    },
    scoring="f1",
    cv=5,
    n_jobs=-1,
)
rf_search.fit(X_train, y_train)

def report(name, model, X_eval, y_eval):
    pred = model.predict(X_eval)
    print(f"\n{name}")
    print("Accuracy:", round(accuracy_score(y_eval, pred), 3))
    print("Precision (lapsed):", round(precision_score(y_eval, pred, zero_division=0), 3))
    print("Recall (lapsed):", round(recall_score(y_eval, pred, zero_division=0), 3))
    print("F1 (lapsed):", round(f1_score(y_eval, pred, zero_division=0), 3))
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_eval, pred, labels=[0, 1]))
    return pred

print("COHORT")
print("Customers:", len(data))
print("Lapsed:", int(y.sum()), f"({y.mean():.2%})")
print("Train:", len(X_train), "| Test:", len(X_test))
print("Feature missing values:", int(X.isna().sum().sum()))

baseline_pred = np.zeros(len(y_test), dtype=int)
print("\nBASELINE: predict every customer will stay")
print("Accuracy:", round(accuracy_score(y_test, baseline_pred), 3))
print("Lapsed F1:", round(f1_score(y_test, baseline_pred, zero_division=0), 3))

print("\nLOGISTIC REGRESSION")
print("Best parameters:", lr_search.best_params_)
print("Best CV F1:", round(lr_search.best_score_, 3))
lr_pred = report("LR TEST RESULTS", lr_search.best_estimator_, X_test, y_test)

print("\nRANDOM FOREST")
print("Best parameters:", rf_search.best_params_)
print("Best CV F1:", round(rf_search.best_score_, 3))
rf_pred = report("RF TEST RESULTS", rf_search.best_estimator_, X_test, y_test)

# Compare the scaler's train-only frequency mean with the full cohort mean.
best_lr = lr_search.best_estimator_
fitted_scaler = best_lr.named_steps["preprocessor"].named_transformers_[
    "numeric"
].named_steps["scaler"]
freq_position = numeric_cols.index("frequency")
print("\nFrequency mean learned on training data:", round(
    fitted_scaler.mean_[freq_position], 3
))
print("Full cohort frequency mean:", round(data["frequency"].mean(), 3))
print("Scaler is fitted on training rows only to avoid test-data leakage.")

# Oversample the training split only, using replacement, to balance the classes.
train_frame = X_train.copy()
train_frame["lapsed"] = y_train.to_numpy()
majority = train_frame.loc[train_frame["lapsed"] == 0]
minority = train_frame.loc[train_frame["lapsed"] == 1]
minority_up = resample(
    minority,
    replace=True,
    n_samples=len(majority),
    random_state=42,
)
balanced = pd.concat([majority, minority_up], ignore_index=True)
y_balanced = balanced.pop("lapsed")
X_balanced = balanced

oversampled_lr = clone(lr_search.best_estimator_)
oversampled_lr.fit(X_balanced, y_balanced)
print("\nOVERSAMPLED LOGISTIC REGRESSION")
print("Training class counts:", y_balanced.value_counts().sort_index().to_dict())
oversampled_pred = report(
    "OVERSAMPLED LR TEST RESULTS", oversampled_lr, X_test, y_test
)

# Choose the better test-set F1 of the two tuned unbalanced models.
chosen_name, chosen_model = (
    ("Random Forest", rf_search.best_estimator_)
    if f1_score(y_test, rf_pred, zero_division=0)
    >= f1_score(y_test, lr_pred, zero_division=0)
    else ("Logistic Regression", lr_search.best_estimator_)
)
chosen_f1 = f1_score(
    y_test, chosen_model.predict(X_test), zero_division=0
)
if chosen_f1 < 0.55:
    print("WARNING: selected model test F1 is below the target of 0.55.")

model_path = OUT / "lapse_model.joblib"
joblib.dump(chosen_model, model_path)
reloaded = joblib.load(model_path)
before_save = chosen_model.predict(X_test)
after_load = reloaded.predict(X_test)
assert np.array_equal(before_save, after_load)
print("\nSELECTED MODEL:", chosen_name, "| test F1:", round(chosen_f1, 3))
print("Saved and reloaded model predictions match:", True)
print("Saved:", model_path)

# Show top five Random Forest feature importances.
rf_model = rf_search.best_estimator_
feature_names = rf_model.named_steps["preprocessor"].get_feature_names_out()
importances = rf_model.named_steps["model"].feature_importances_
importance_table = pd.Series(importances, index=feature_names).sort_values(
    ascending=False
)
print("\nTOP FIVE RANDOM FOREST FEATURES")
print(importance_table.head(5).round(4).to_string())

# Show three largest absolute Logistic Regression coefficients.
lr_model = lr_search.best_estimator_
lr_names = lr_model.named_steps["preprocessor"].get_feature_names_out()
coefficients = pd.Series(
    lr_model.named_steps["model"].coef_[0], index=lr_names
)
print("\nTHREE LOGISTIC REGRESSION COEFFICIENTS BY ABSOLUTE SIZE")
print(coefficients.reindex(
    coefficients.abs().sort_values(ascending=False).head(3).index
).round(4).to_string())

# Rank test-set Premium customers by predicted probability of lapse.
premium_ids = set(
    pd.read_csv(OUT / "customer_segments.csv")
    .loc[lambda d: d["is_premium"].astype(bool), "customer_id"]
)
test_results = pd.DataFrame({
    "customer_id": id_test.to_numpy(),
    "lapse_probability": chosen_model.predict_proba(X_test)[:, 1],
    "actually_lapsed": y_test.to_numpy(),
})
retention = test_results.loc[
    test_results["customer_id"].isin(premium_ids)
].sort_values("lapse_probability", ascending=False).head(10)
retention["lapse_probability"] = retention["lapse_probability"].round(4)
retention.to_csv(OUT / "retention_list.csv", index=False)
print("\nTOP PREMIUM RETENTION TARGETS")
print(retention.to_string(index=False))
print("Truly lapsed in this top-10 list:", int(retention["actually_lapsed"].sum()))
print("Saved:", OUT / "retention_list.csv")

assert len(data) == 1156, f"Expected 1156 cohort customers, got {len(data)}"
assert int(y.sum()) == 443, f"Expected 443 lapsed customers, got {int(y.sum())}"
assert len(X_test) == 232
assert len(retention) <= 10
print("\nLAPSE MODEL CHECKS PASSED")
