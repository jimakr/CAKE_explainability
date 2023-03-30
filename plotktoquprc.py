import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

a = pd.read_excel('results_cake_1-20.xlsx')
b = a.groupby(['Dataset', 'Distil', 'Embedding model'])
b = list(b)
xs = []
name = []
for i, df in b:
    xs.append(df['AUPRC'])
    name.append(i)

color = {'ESNLI': 'red', 'HOC(S)':'blue', 'HX': 'green', 'Hummingbird': 'purple', 'Movies':'orange', 'Movies(S)': 'black'}
index = np.arange(20) + 1
printed = []
for dat, na in zip(xs, name):
    if na[1] == False:
        plt.plot(index, dat, color=color[na[0]], label=na[0])
        printed.append(na[0])
for dat, na in zip(xs, name):
    if na[1] == True:
        plt.plot(index, dat, '--', color=color[na[0]], label='_nolegend_')


plt.legend()
plt.show()

