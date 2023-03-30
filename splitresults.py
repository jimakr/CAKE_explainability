import pandas as pd

a = pd.read_excel('results.xlsx')
a = a.round(3)
b = a.groupby(['Dataset', 'Distil', 'Embedding model'])
b = list(b)

for i, df in b:
    df = df.drop(['Dataset', 'Distil', 'Embedding model'], axis=1)
    df.to_excel(f'tables/{str(i)}.xlsx', index=False)
