import pandas as pd
f=pd.read_csv("fer2013.csv")
keep_col = ['emotion','Usage']
new_f = f[keep_col]
new_f.to_csv("enhanceddata.csv", index=True)
