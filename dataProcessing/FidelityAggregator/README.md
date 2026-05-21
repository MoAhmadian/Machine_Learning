# FidelityAggregator

## Overview
This project aggregates position data across multiple Fidelity brokerage accounts based on ticker symbols. The Fidelity website does not provide built-in functionality to view aggregate positions across accounts, so this tool consolidates data from individual account exports.

## Features
- **Multi-Account Aggregation**: Combines positions from multiple Fidelity account exports
- **Data Cleaning**: Automatically cleans raw CSV exports (removes special characters, currency symbols, etc.)
- **Position Consolidation**: Groups positions by ticker symbol across all accounts
- **Performance Analysis**: Calculates gains/losses and portfolio percentages
- **Weighted Averaging**: Computes weighted average cost across accounts

## Requirements
- Python 3.7+
- pandas
- pathlib (standard library)
- re (standard library)
- datetime (standard library)

## Installation
```bash
pip install pandas
```

## Usage

### Step 1: Export from Fidelity
Export your portfolio positions from Fidelity as CSV:
1. Log into Fidelity
2. Navigate to your account positions
3. Download the CSV file as `Portfolio_Positions_[DATE].csv`
4. Place it in the `data/` subdirectory

### Step 2: Run the Processor
```bash
python Portfolio_process.py
```

The script will:
1. Auto-detect today's date
2. Clean the raw CSV file
3. Process and aggregate the positions
4. Output a consolidated file: `Processed_[DATE].csv`

## Input CSV Format
Expected columns from Fidelity export:
- Account Number
- Account Name
- Symbol
- Description
- Last Price
- Quantity
- Current Value
- Average Cost
- Cost Total
- Percent of Account
- Total Gain
- Total Gain %

## Output CSV Format
The processed file contains:
- **Symbol**: Ticker symbol
- **Qty**: Total quantity across all accounts
- **Last_price**: Current price per share
- **current_value**: Total current market value
- **cost**: Weighted average cost basis
- **total_cost**: Total cost basis across all accounts
- **percentage_of_portfolio**: Position as % of total portfolio
- **gain**: Absolute gain/loss in dollars
- **gain_percentage**: Gain/loss as percentage

## Example Output
```
Symbol,Qty,Last_price,current_value,cost,total_cost,percentage_of_portfolio,gain,gain_percentage
APPL,150.0,178.45,26767.50,165.32,24798.0,35.21,1969.5,7.93
MSFT,100.0,385.12,38512.0,350.0,35000.0,50.61,3512.0,10.03
```

## Notes
- Date format: `Month-Day-Year` (e.g., "May-07-2026")
- The script automatically removes cash symbols (SPAXX, FDRXX) and pending activities
- All monetary values are converted to floats for calculation
- Results are sorted by portfolio percentage (largest positions first)

## Troubleshooting

**File not found error**: Ensure the CSV file is in the `data/` folder with the correct naming convention.

**Parsing errors**: Make sure you're exporting the portfolio in the standard Fidelity CSV format. The script handles common formatting issues automatically.

**Percentage calculations**: Gain percentage is calculated as `(gain / total_cost) * 100`
