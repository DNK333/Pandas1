from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
ORDERS_FILE = BASE_DIR / "orders.csv"
PRODUCTS_FILE = BASE_DIR / "products.csv"
CLEANED_FILE = BASE_DIR / "cleaned_sales.csv"
PIVOT_FILE = BASE_DIR / "sales_pivot.csv"


def main() -> None:
    orders = pd.read_csv(ORDERS_FILE)
    products = pd.read_csv(PRODUCTS_FILE)

    orders = orders.drop_duplicates().copy()
    orders["quantity"] = pd.to_numeric(orders["quantity"], errors="raise")
    orders["quantity"] = np.where(
        orders["quantity"].isna(), 1, orders["quantity"]
    ).astype(int)
    orders["date"] = pd.to_datetime(orders["date"], errors="raise")

    sales = orders.merge(products, on="product_id", how="left", validate="many_to_one")
    if sales["product"].isna().any():
        missing_ids = sales.loc[sales["product"].isna(), "product_id"].unique()
        raise ValueError(f"Products not found for product_id values: {missing_ids.tolist()}")

    sales["revenue"] = sales["quantity"] * sales["price"]
    sales["month"] = sales["date"].dt.to_period("M").astype(str)

    total_revenue = sales["revenue"].sum()
    revenue_by_product = sales.groupby("product")["revenue"].sum().sort_values(ascending=False)
    revenue_by_category = sales.groupby("category")["revenue"].sum().sort_values(ascending=False)
    revenue_by_month = sales.groupby("month")["revenue"].sum().sort_values(ascending=False)
    average_order_position_revenue = sales["revenue"].mean()

    pivot = sales.pivot_table(
        index="product",
        columns="month",
        values="revenue",
        aggfunc="sum",
        fill_value=0,
    )
    pivot.columns.name = None

    sales.to_csv(CLEANED_FILE, index=False, date_format="%Y-%m-%d")
    pivot.to_csv(PIVOT_FILE)

    print(f"Общая выручка: {total_revenue:.2f}")
    print(f"Товар с максимальной выручкой: {revenue_by_product.idxmax()} "
          f"({revenue_by_product.max():.2f})")
    print("\nВыручка по категориям:")
    print(revenue_by_category.to_string())
    print(f"\nМесяц с максимальной выручкой: {revenue_by_month.idxmax()} "
          f"({revenue_by_month.max():.2f})")
    print(f"\nСредняя выручка с позиции заказа: {average_order_position_revenue:.2f}")
    print(f"Категория с минимальной выручкой: {revenue_by_category.idxmin()} "
          f"({revenue_by_category.min():.2f})")
    print(f"\nСохранены файлы:\n- {CLEANED_FILE.name}\n- {PIVOT_FILE.name}")


if __name__ == "__main__":
    main()
