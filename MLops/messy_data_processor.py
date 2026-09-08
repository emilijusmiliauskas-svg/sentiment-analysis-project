import pandas as pd


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans and processes the dataframe."""
    # Clean column names by stripping whitespace and converting to lower case
    df.columns = df.columns.str.strip().str.lower()

    # Ensure 'price' is a numeric column, coercing errors
    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")

    return df


if __name__ == "__main__":
    data = {
        "Product Name": ["  Laptop ", " Mouse", "Keyboard"],
        "Price": ["1200", "25.50", "75"],
    }
    initial_df = pd.DataFrame(data)

    cleaned_df = process_data(initial_df)
    print("Cleaned DataFrame:")  # noqa: T201
    print(cleaned_df)  # noqa: T201
