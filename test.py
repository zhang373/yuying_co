import pandas as pd
for data_name in ["L11005uncleaned.xlsx", "real_L1L2_uncleaned.xlsx"]:
    df = pd.read_excel("./dataset/"+data_name)
    print(df.columns.tolist())