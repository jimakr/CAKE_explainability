import numpy as np


def print_results(name, techniques, metrics, label_names):
    results = {'AUPRC': '-'}
    for metric in metrics.keys():
        print(metric)
        temp_metric = np.array(metrics[metric])
        for i in range(len(techniques)):
            label_score = []
            for label in range(len(label_names)):
                tempo = [k for k in temp_metric[:, i, label]
                         if str(k) != 'nan']
                if len(tempo) == 0:
                    tempo.append(0)
                label_score.append(np.array(tempo))
            temp_mean = []
            for k in label_score:
                temp_mean.append(k.mean())

            temp_mean = np.array(temp_mean).mean()
            results[metric] = temp_mean
            print(techniques[i], ' {} | {}'.format(round(temp_mean, 5), ' '.join(
                [str(round(label_score[o].mean(), 5)) for o in range(len(label_names))])))
    return results


