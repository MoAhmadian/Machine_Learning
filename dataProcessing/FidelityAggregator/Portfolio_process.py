import pandas as pd
from pathlib import Path
import re
from datetime import datetime
import sys


def get_fidelity_date_str():
    today = datetime.today()
    return today.strftime("%B-%d-%Y")   # Example: "May-07-2026"

def clean_portfolio_file(date_str):
    """
    Cleans the raw Fidelity CSV BEFORE pandas loads it.
    Fixes formatting issues that break pandas.
    """

    cleaned_lines = []
    input_path = Path("data") / f"Portfolio_Positions_{date_str}.csv"
    
    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}")
        sys.exit(1)
    
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:

            # All unwanted lines start with a double quote
            if line.startswith('"'):
                break

            # Skip empty lines before disclaimer block
            if line.strip() == "":
                continue

            # 1. Remove trailing comma at end of row
            line = re.sub(r",\s*$", "", line)

            # 2. Remove $, + signs (but keep negative sign)
            line = re.sub(r"\$", "", line)
            line = re.sub(r"\+", "", line)

            # 3. Remove special characters (® etc.) and force ASCII
            line = line.encode("ascii", "ignore").decode("ascii")

            # 4. Remove all asterisks (*)
            line = line.replace("*", "")

            if not line.endswith("\n"):
                line = line + "\n"

            cleaned_lines.append(line)

    # Write cleaned file
    output_path = input_path
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(cleaned_lines)

    print(f"Cleaned file written to: {output_path}")


def process_portfolio(date_str):
    file_path = Path("data") / f"Portfolio_Positions_{date_str}.csv"

    # Load CSV with correct header
    df = pd.read_csv(file_path)
    df.columns = [
        "Account_Number",
        "Account_Name",
        "Symbol",
        "Description",
        "Last_price",
        "Quantity",
        "Current_value",
        "Average_cost",
        "Cost_total",
        "Percent_of_account",
        "Total_gain",
        "Total_gain_percent",
    ]

    # Normalize column names
    df.columns = df.columns.str.strip()

    # Drop columns you don't want
    df = df.drop(columns=["Account_Number", "Description"], errors="ignore")

    print(df.head())  # debug print

    # Drop percent columns
    df = df.drop(columns=["Percent_of_account", "Total_gain_percent"], errors="ignore")

    # Clean numeric columns
    numeric_cols = [
        "Last_price",
        "Quantity",
        "Current_value",
        "Average_cost",
        "Cost_total",
        "Total_gain"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = (
                pd.to_numeric(
                    df[col]
                    .astype(str)
                    .str.replace(r"[,\$+]", "", regex=True)
                    .str.replace("(", "-", regex=False)
                    .str.replace(")", "", regex=False)
                    .replace({"": None, "--": None, "NA": None, "N/A": None}),
                    errors="coerce",
                )
                .fillna(0.0)
            )

    # Remove empty symbols
    df = df[df["Symbol"].notna() & (df["Symbol"] != "")]

    # Remove non-stock symbols
    bad_symbols = ["SPAXX", "FDRXX", "Pending activity", "USD"]
    df = df[~df["Symbol"].isin(bad_symbols)]

    # Compute weighted total cost per row
    df["row_cost"] = df["Quantity"] * df["Average_cost"]

    # Group by SYMBOL
    grouped = df.groupby("Symbol").agg(
        Qty=("Quantity", "sum"),
        Last_price=("Last_price", "last"),
        total_cost=("row_cost", "sum"),
        current_value=("Current_value", "sum")
    ).reset_index()

    # Weighted average cost
    grouped["cost"] = grouped["total_cost"] / grouped["Qty"]

    # Gain/loss
    grouped["gain"] = grouped["current_value"] - grouped["total_cost"]
    grouped["gain_percentage"] = (grouped["gain"] / grouped["total_cost"]) * 100

    # Percentage of portfolio
    total_value = grouped["current_value"].sum()
    grouped["percentage_of_portfolio"] = grouped["current_value"] / total_value * 100

    grouped["current_value"] = grouped["Qty"] * grouped["Last_price"]

    # Final output
    final_df = grouped[
        [
            "Symbol",
            "Qty",
            "Last_price",
            "current_value",
            "cost",
            "total_cost",
            "percentage_of_portfolio",
            "gain",
            "gain_percentage"
        ]
    ]

    final_df = final_df.sort_values("percentage_of_portfolio", ascending=False)
    final_df = final_df.round(2)

    output_path = Path("data") / f"Processed_{date_str}.csv"
    final_df.to_csv(output_path, index=False)

    print(f"Processed file saved to: {output_path}")

def run_today():
    date_str = get_fidelity_date_str()
    clean_portfolio_file(date_str)
    process_portfolio(date_str)

run_today()
