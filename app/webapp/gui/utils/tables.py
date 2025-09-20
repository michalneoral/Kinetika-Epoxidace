import pandas as pd


def sanitize_df_for_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Step 1: Replace NaNs and convert to string
    col_names = [str(c) if pd.notna(c) else f'Unnamed_{i}' for i, c in enumerate(df.columns)]

    # Step 2: Deduplicate manually
    seen = {}
    deduped_names = []
    for name in col_names:
        if name not in seen:
            seen[name] = 1
            deduped_names.append(name)
        else:
            seen[name] += 1
            deduped_names.append(f'{name}_{seen[name]}')

    df.columns = deduped_names
    df.index = df.index.astype(str)
    return df
