import pandas as pd
df = pd.read_csv("data/dataset.csv", encoding="utf-8-sig")
# Check for Unicode curly quotes
for col in df.select_dtypes(include=["object"]).columns:
    try:
        content = str(df[col].values)
        # U+201C = left double quote, U+201D = right double quote
        if chr(0x201C) in content or chr(0x201D) in content:
            print(f"Found curly quotes in column: {col}")
    except:
        pass
print("Check for Unicode curly quotes completed.")
