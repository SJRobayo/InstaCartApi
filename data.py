import pandas as pd
DATA_PATH = "csv/df_final.csv"

df = pd.read_csv(DATA_PATH)
print(df.head(-1))