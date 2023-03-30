import pandas as pd
import numpy as np
from itertools import compress, product
import os

default = [10, 0.05, 'svm', 'isotonic', (2, 2)]
data = pd.read_excel('datasets.xlsx')
datasets = list(data['id'])
datasets_m = np.ones_like(datasets)

embeding = ['Sbert', 'Trained']
embeding_m = [0, 1]

# topn = list(np.arange(20) + 1)
# topn_m = list(np.ones(20, dtype=int))

topn = [5, 10, 15, 20, 100]
topn_m = [1, 1, 1, 1, 1]

threshold = [0.05, 0.2, 0.3, 0]
threshold_m = [0, 0, 0, 1]

model = ['svm', 'knn', 'logistic', 'tree', 'bayes', 'cosine', 'zero', 'cosine_pre', 'perdoc_cosine']
model_m = [0, 0, 0, 0, 0, 1, 1, 0, 1]

calibration = ['isotonic', 'sigmoid']
calibration_m = [1, 0]

distil = [True, False]
distil_m = [1, 1]

candidate = ['PatternRank', (1, 3), (2, 2), 'Yake', 'Rake', (2, 3), 'single']
candidate_m = [1, 0, 0, 1, 0, 0, 1]

parameters = [datasets, distil, embeding, topn, threshold, model, calibration, candidate]
parameters_m = [datasets_m, distil_m, embeding_m, topn_m, threshold_m, model_m, calibration_m, candidate_m]
docmetrics = ['AUPRC', 'F', 'FTP', 'NZW', 'time', 'f1_fidelity']

combos = [list(compress(i, m)) for i, m in zip(parameters, parameters_m)]
df = pd.DataFrame(list(product(*combos)))
df = df.set_axis(['Dataset', 'Distil', 'Embedding model', 'Keyphrase Number', 'Threshold', 'Ranking', 'Calibration', 'Candidate Generation'], axis=1, copy=False)

if os.path.exists('parameters.xlsx'):
    old_df = pd.read_excel('parameters.xlsx')
    df = pd.merge(df, old_df,  how='outer', on=['Dataset', 'Distil', 'Embedding model', 'Keyphrase Number', 'Threshold', 'Ranking', 'Calibration', 'Candidate Generation'])

    # for i in docmetrics:
    #     df[f'{i}_x'].fillna(df[f'{i}_y'], inplace=True)
    #
    # df.drop([f'{i}_y' for i in docmetrics], axis=1, inplace=True)
    # df.rename(columns={f'{i}_x': f'{i}' for i in docmetrics}, inplace=True)
else:
    df = df.reindex(df.columns.tolist() + docmetrics, axis=1)
#
# combos_default = [list(compress(i, map(lambda x:not x, m))) for i, m in zip(parameters, parameters_m)][2:]
#
# combinations = []
# for i, sublist in enumerate(combos_default):
#     for value in sublist:
#         new_combination = default.copy()
#         new_combination[i] = value
#         combinations.append(new_combination)
#
# combo = []
# for i in datasets:
#     combo.extend([[i, True, ] + b for b in combinations])
# for i in datasets:
#     combo.extend([[i, False, ] + b for b in combinations])
#
# combo = pd.DataFrame(combo)
# combo = combo.set_axis(['Dataset', 'Distil', 'Embedding model', 'Keyphrase Number', 'Threshold', 'Ranking', 'Calibration', 'Candidate Generation'], axis=1, copy=False)
# combo.to_excel('combo.xlsx', index=False)
# final = pd.concat([df, combo])


df.to_excel('parameters.xlsx', index=False)
