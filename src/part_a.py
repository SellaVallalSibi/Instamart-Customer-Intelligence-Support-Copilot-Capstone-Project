from pathlib import Path
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
CLEAN.mkdir(parents=True, exist_ok=True)

# Load raw data
customers = pd.read_csv(RAW / "customers.csv", keep_default_na=False)
products = pd.read_csv(RAW / "products.csv", keep_default_na=False)
orders = pd.read_csv(RAW / "orders.csv", keep_default_na=False)

# Clean customers
customers["city"] = customers["city"].astype(str).str.strip().str.title()
city_map = {
    "Bangalore": "Bengaluru",
    "Bombay": "Mumbai",
    "New Delhi": "Delhi",
}
customers["city"] = customers["city"].replace(city_map)

allowed_cities = ["Bengaluru", "Chennai", "Delhi", "Hyderabad", "Mumbai", "Pune"]
customers.loc[~customers["city"].isin(allowed_cities), "city"] = "Delhi"

customers["gender"] = customers["gender"].astype(str).str.strip().str.lower()
customers["gender"] = customers["gender"].replace({
    "m": "male", "man": "male", "male": "male",
    "f": "female", "woman": "female", "female": "female",
})
customers.loc[~customers["gender"].isin(["male", "female"]), "gender"] = "unknown"

customers["age"] = pd.to_numeric(customers["age"], errors="coerce")
customers.loc[~customers["age"].between(10, 100), "age"] = pd.NA
customers["age"] = customers["age"].astype("Float64")
customers["age"] = customers["age"].fillna(
    customers.groupby("city")["age"].transform("median")
)
customers["age"] = customers["age"].fillna(customers["age"].median())
customers["signup_date"] = pd.to_datetime(
    customers["signup_date"], errors="coerce"
)

# Clean products
products["price_inr"] = pd.to_numeric(products["price_inr"], errors="coerce")

# Clean orders: parse dates, remove exact duplicates, keep latest per order ID
orders["order_ts"] = pd.to_datetime(orders["order_ts"], errors="coerce")
orders = orders.drop_duplicates()
orders = orders.sort_values("order_ts").drop_duplicates(
    subset=["order_id"], keep="last"
).copy()

for col in ["quantity", "discount_pct", "order_amount", "delivery_minutes"]:
    orders[col] = pd.to_numeric(orders[col], errors="coerce")

orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = pd.NA

# Fill missing/unparseable order amount using product price and quantity
orders = orders.merge(
    products[["product_id", "price_inr", "category"]],
    on="product_id",
    how="left",
    validate="many_to_one",
)
calculated_amount = (
    orders["price_inr"] * orders["quantity"] *
    (1 - orders["discount_pct"] / 100)
)
orders["order_amount"] = orders["order_amount"].fillna(calculated_amount)

# Fill missing delivery time only for completed orders, by category median
completed = orders["status"].astype(str).str.strip().str.lower().eq("completed")
category_median = orders.loc[completed].groupby("category")[
    "delivery_minutes"
].transform("median")
completed_indices = orders.index[completed]
orders.loc[completed_indices, "delivery_minutes"] = (
    orders.loc[completed_indices, "delivery_minutes"].fillna(category_median)
)

# Remove helper product columns and normalize output
orders = orders.drop(columns=["price_inr", "category"])
orders["order_ts"] = orders["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
orders["status"] = orders["status"].astype(str).str.strip().str.lower()
orders["order_amount"] = orders["order_amount"].round(2)
orders["delivery_minutes"] = orders["delivery_minutes"].round(2)

# Save clean CSVs
customers.to_csv(CLEAN / "customers.csv", index=False)
products.to_csv(CLEAN / "products.csv", index=False)
orders.to_csv(CLEAN / "orders.csv", index=False)

# Create SQLite database and three tables
db_path = ROOT / "data" / "instamart.db"
with sqlite3.connect(db_path) as conn:
    customers.to_sql("customers", conn, if_exists="replace", index=False)
    products.to_sql("products", conn, if_exists="replace", index=False)
    orders.to_sql("orders", conn, if_exists="replace", index=False)

# Validate key assessment checkpoints
assert len(customers) == 1200, f"Unexpected customer rows: {len(customers)}"
assert len(products) == 40, f"Unexpected product rows: {len(products)}"
assert len(orders) == 12309, f"Unexpected clean order rows: {len(orders)}"
assert orders["order_id"].is_unique, "Duplicate order IDs remain"
assert orders["status"].eq("pending").sum() == 0, "Pending orders remain"
assert orders["status"].eq("completed").sum() == 11110, "Completed count mismatch"
assert orders["status"].eq("cancelled").sum() == 1199, "Cancelled count mismatch"
assert orders.loc[orders["status"].eq("completed"), "delivery_minutes"].isna().sum() == 0
assert orders.loc[orders["status"].eq("cancelled"), "delivery_minutes"].isna().sum() == 1199
assert orders["order_amount"].isna().sum() == 0, "Missing order amounts remain"
assert customers["age"].isna().sum() == 0, "Missing customer ages remain"
assert set(customers["gender"].unique()).issubset({"male", "female", "unknown"})

completed_revenue = orders.loc[
    orders["status"].eq("completed"), "order_amount"
].sum()
assert round(completed_revenue, 2) == 2998668.80, (
    f"Completed revenue mismatch: {completed_revenue:.2f}"
)

print("Clean customers:", len(customers))
print("Clean products:", len(products))
print("Clean orders:", len(orders))
print("Completed revenue INR:", f"{completed_revenue:.2f}")
print("Database:", db_path)
print("ALL CHECKPOINTS PASSED")
