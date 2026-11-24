
import pandas as pd
import csv
import re

file_path = "order-export-2026-10-28-12-24-00-uFhcdDpvynCP.csv"

# Detect CSV delimiter
with open(file_path, "r", encoding="utf-8-sig") as f:
    sample = f.read(65536)

delimiter = csv.Sniffer().sniff(
    sample, delimiters=",;\t|"
).delimiter

# Read only the fee_total column
df = pd.read_csv(
    file_path,
    sep=delimiter,
    usecols=["fee_total"],
    dtype="string",
    encoding="utf-8-sig"
)

raw = df["fee_total"].str.strip()

# Apply the same conversion logic as the cleaning pipeline
cleaned = (
    raw
    .str.replace(",", "", regex=False)
    .str.replace(r"^[₹$€£]\s*", "", regex=True)
)

cleaned = cleaned.replace({
    "": pd.NA,
    "nan": pd.NA,
    "None": pd.NA
})

numeric = pd.to_numeric(cleaned, errors="coerce")

# Find non-empty values that failed conversion
invalid_mask = raw.notna() & raw.ne("") & numeric.isna()

invalid_values = raw[invalid_mask].value_counts(dropna=False)

print("\nINVALID FEE VALUES")
print("------------------")
print("Total invalid:", int(invalid_mask.sum()))

print("\nInvalid value frequencies:")
print(invalid_values.to_string())

print("\nExamples of invalid values:")
print(raw[invalid_mask].head(20).to_list())
