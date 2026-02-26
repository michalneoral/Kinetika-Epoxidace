from __future__ import annotations
from typing import List
import pandas as pd

def sanitize_df_for_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols: List[str] = []
    for i,c in enumerate(df.columns):
        if pd.isna(c) or str(c).strip()=="":
            cols.append(f"Unnamed_{i}")
        else:
            cols.append(str(c))
    df.columns = cols
    seen={}
    new=[]
    for c in df.columns:
        if c not in seen:
            seen[c]=0
            new.append(c)
        else:
            seen[c]+=1
            new.append(f"{c}_{seen[c]}")
    df.columns=new
    return df
