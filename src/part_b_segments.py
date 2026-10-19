from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

customers = pd.read_csv(ROOT / "data/clean/customers.csv")
orders = pd.read_csv(ROOT / "data/clean/orders.csv")
orders["order_ts"] = pd.to_datetime(orders["order_ts"], errors="coerce")

completed = orders.loc[orders["status"].eq("completed")].copy()
as_of = pd.Timestamp("2025-10-01")

# Build RFM using completed orders only
rfm = completed.groupby("customer_id").agg(
    last_order=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum"),
).reset_index()

rfm["recency"] = (as_of - rfm["last_order"]).dt.days
rfm = rfm[["customer_id", "recency", "frequency", "monetary"]]
rfm["monetary"] = rfm["monetary"].round(2)

features = rfm[["recency", "frequency", "monetary"]]
scaler = StandardScaler()
X = scaler.fit_transform(features)

print("RFM customers:", len(rfm))
print("RFM summary:")
print(rfm[["recency", "frequency", "monetary"]].describe().round(2).to_string())

print("\nK-MEANS COMPARISON")
for k in range(2, 9):
    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = model.fit_predict(X)
    score = silhouette_score(X, labels)
    print(f"K={k}: inertia={model.inertia_:.1f}, silhouette={score:.3f}")

# Use four interpretable clusters
model = KMeans(n_clusters=4, n_init=10, random_state=42)
rfm["segment_id"] = model.fit_predict(X)

profile = rfm.groupby("segment_id").agg(
    customers=("customer_id", "size"),
    mean_recency=("recency", "mean"),
    mean_frequency=("frequency", "mean"),
    mean_monetary=("monetary", "mean"),
).round(2)

# Name segments by observable behavior; Premium has the highest mean spend
premium_id = profile["mean_monetary"].idxmax()
remaining = [idx for idx in profile.index if idx != premium_id]
loyal_id = profile.loc[remaining, "mean_frequency"].idxmax()
remaining = [idx for idx in remaining if idx != loyal_id]
at_risk_id = profile.loc[remaining, "mean_recency"].idxmax()
regular_id = next(idx for idx in remaining if idx != at_risk_id)

names = {
    premium_id: "Premium",
    loyal_id: "Loyal",
    at_risk_id: "At-Risk",
    regular_id: "Regular",
}
rfm["segment_name"] = rfm["segment_id"].map(names)
rfm["is_premium"] = rfm["segment_name"].eq("Premium")

result = rfm.merge(
    customers[["customer_id", "city", "age", "gender"]],
    on="customer_id",
    how="left",
    validate="one_to_one",
)
result.to_csv(OUT / "customer_segments.csv", index=False)

print("\nSEGMENT PROFILE")
print(profile.assign(segment_name=pd.Series(names)).sort_index().to_string())
print("\nSAVED:", OUT / "customer_segments.csv")
print("Premium customers:", int(result["is_premium"].sum()))

# PCA variance explained for two-dimensional visualization/reporting
pca = PCA(n_components=2, random_state=42)
pca.fit(X)
print("\nPCA explained variance:", np.round(pca.explained_variance_ratio_, 3))
print("PCA cumulative variance:", round(pca.explained_variance_ratio_.sum(), 3))

assert len(rfm) == 1175, f"Expected 1175 RFM customers, got {len(rfm)}"
assert result["customer_id"].is_unique
assert result["segment_name"].notna().all()
assert result["is_premium"].sum() > 0
print("\nSEGMENTATION CHECKS PASSED")
