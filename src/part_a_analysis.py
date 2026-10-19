from pathlib import Path
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"

customers = pd.read_csv(CLEAN / "customers.csv")
products = pd.read_csv(CLEAN / "products.csv")
orders = pd.read_csv(CLEAN / "orders.csv")

completed = orders.loc[orders["status"] == "completed"].copy()

# Amount distribution and IQR outliers
amount = completed["order_amount"]
q1, q3 = amount.quantile([0.25, 0.75])
iqr = q3 - q1
upper_fence = q3 + 1.5 * iqr
lower_fence = q1 - 1.5 * iqr

print("ORDER AMOUNT DISTRIBUTION")
print(amount.describe().round(2).to_string())
print("Skew:", round(amount.skew(), 2))
print("IQR lower fence:", round(lower_fence, 2))
print("IQR upper fence:", round(upper_fence, 2))
print("Below lower fence:", int((amount < lower_fence).sum()))
print("Above upper fence:", int((amount > upper_fence).sum()))

# Correlation between numeric order fields
numeric_cols = ["order_amount", "quantity", "discount_pct", "delivery_minutes"]
print("\nNUMERIC CORRELATIONS")
print(completed[numeric_cols].corr().round(3).to_string())

# City summary
print("\nCOMPLETED REVENUE BY CITY")
completed_city = completed.merge(
    customers[["customer_id", "city"]],
    on="customer_id",
    how="left",
    validate="many_to_one",
)
city_summary = completed_city.groupby("city")["order_amount"].agg(
    ["sum", "mean", "count"]
)
city_summary.columns = ["revenue_inr", "mean_order_inr", "order_count"]
print(city_summary.round(2).to_string())

# SQL vs pandas pivot-table revenue validation
with sqlite3.connect(ROOT / "data" / "instamart.db") as conn:
    sql_revenue = pd.read_sql_query(
        """
        SELECT c.city, p.category, SUM(o.order_amount) AS revenue_inr
        FROM orders o
        JOIN customers c ON c.customer_id = o.customer_id
        JOIN products p ON p.product_id = o.product_id
        WHERE o.status = 'completed'
        GROUP BY c.city, p.category
        """,
        conn,
    )

merged = completed.merge(
    customers[["customer_id", "city"]], on="customer_id", how="left"
).merge(
    products[["product_id", "category"]], on="product_id", how="left"
)

pivot = pd.pivot_table(
    merged,
    index=["city", "category"],
    values="order_amount",
    aggfunc="sum",
).reset_index()

check = sql_revenue.merge(
    pivot, on=["city", "category"], suffixes=("_sql", "_pandas")
)
assert len(check) == 30
assert (
    check["revenue_inr"].round(2)
    == check["order_amount"].round(2)
).all()
assert round(check["revenue_inr"].sum(), 2) == 2998668.80

# Core validation checks
assert len(customers) == 1200
assert len(products) == 40
assert len(orders) == 12309
assert orders["order_id"].is_unique
assert not customers["age"].isna().any()
assert set(customers["gender"].unique()) <= {"male", "female", "unknown"}
assert orders.loc[orders["status"] == "completed", "delivery_minutes"].notna().all()
assert orders.loc[orders["status"] == "cancelled", "delivery_minutes"].isna().all()

with sqlite3.connect(ROOT / "data" / "instamart.db") as conn:
    tables = {
        row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
assert {"customers", "products", "orders"}.issubset(tables)

print("\nPANDAS PIVOT VS SQL: MATCH")
print("Database tables:", ", ".join(sorted(tables)))
print("ALL CHECKPOINTS PASSED")
