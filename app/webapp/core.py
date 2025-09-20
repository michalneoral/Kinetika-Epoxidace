

def extract_df(df_raw, row_start, row_end, col_start, col_end):
    """Extract and format DataFrame from a raw Excel sheet."""
    sub = df_raw.iloc[row_start-1:row_end, col_start-1:col_end]
    sub.columns = sub.iloc[0]
    return sub.iloc[1:].reset_index(drop=True)


def extract_df_dict(df_raw, dict_vals):
    row_start = int(dict_vals['row_start'])
    row_end = int(dict_vals['row_end'])
    col_start = int(dict_vals['col_start'])
    col_end = int(dict_vals['col_end'])
    return extract_df(df_raw, row_start, row_end, col_start, col_end)

def validate_table(df, key_column, required_values):
    """Check that required values exist in the given column."""
    if key_column not in df.columns:
        return False, f"❌ Sloupec '{key_column}' nebyl nalezen."
    missing = [v for v in required_values if v not in df[key_column].values]
    if missing:
        return False, f"⚠️ Chybějící hodnoty ve sloupci '{key_column}': {', '.join(missing)}"
    return True, f"✅ Tabulka obsahuje všechny požadované hodnoty."
