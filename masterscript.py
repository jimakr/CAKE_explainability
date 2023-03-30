import numpy as np
import pandas as pd
import heavenCashe
import importlib
import os
import time
import warnings
import torch
from tqdm import tqdm
from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, average_precision_score
from scipy.special import softmax
from myExplainers import MyExplainer
from optimus_code.myEvaluation import MyEvaluation
from optimus_code.helper import print_results
from optimus_code.dataset import Dataset
from optimus_code.modelwrapper import ModelWrap
from transformers_interpret import SequenceClassificationExplainer
from transformers import Trainer, TrainingArguments
from keyphraseexpln import CAKE

# read and select parameters to run and save
old_parameters = pd.read_excel('parameters.xlsx')
parameters = old_parameters[old_parameters['F'].isnull()]
# only one group of parameters will run
parameters = parameters.groupby(['Dataset', 'Distil', 'Embedding model'])
parameters = list(parameters)
if len(parameters) == 0:
    print('no parameters left')
    # exit(0)
    parameters = {'Dataset': 'Movies', 'Distil': True, 'Embedding model': 'Trained', 'Keyphrase Number': 10, 'Threshold': 0, 'Ranking': 'perdoc_cosine', 'Calibration': 'isotonic', 'Candidate Generation': 'PatternRank', 'AUPRC': np.nan, 'F': np.nan, 'FTP': np.nan, 'NZW': np.nan, 'time': np.nan, 'f1_fidelity': np.nan}
    parameters = {i: [j] for i, j in parameters.items()}
    parameters = pd.DataFrame.from_dict(parameters)
else:
    parameters = parameters[0][1]

dataset_param = pd.read_excel('datasets.xlsx')
dataset_param = dataset_param.set_index('id')
dataset_param = dataset_param.loc[parameters.iloc[0]['Dataset']].to_dict()

# load the chosen dataset parameters
dataname = dataset_param['dataname']
existing_rationales = dataset_param['rationale']
task = dataset_param['task'] + '_label'
sentence_level = dataset_param['sentence']
distil = int(parameters.iloc[0]['Distil'])
labels = dataset_param['labels']
saverate = 100  # for the cashe


# load oprimus pretrained models and tokenizers
distil = 'Distil' if distil else ''
multi = 'Multilabel' if task == 'multi_label' else ''

transformers = importlib.import_module("transformers")
used_tokenizer = getattr(transformers, f"{distil}BertTokenizerFast")

myTransformer = importlib.import_module("optimus_code.myTransformer")
used_model = getattr(myTransformer, f"{distil}BertFor{multi}SequenceClassification")

cased_string = '' if dataname=='Ethos'or dataname=='AIS' else 'un'
tokenizer = used_tokenizer.from_pretrained(f'{distil.lower()}bert-base-{cased_string}cased')

model_path = f'./models/{distil.lower()}bert_{dataname.lower()}'
tokenizer.save_pretrained(model_path + '/')
model = used_model.from_pretrained(model_path, output_attentions=False, output_hidden_states=False)

training_arguments = TrainingArguments(evaluation_strategy='epoch', save_strategy='epoch', logging_strategy='epoch',
                                       log_level='critical', output_dir='./results', num_train_epochs=1,disable_tqdm=True ,
                                       per_device_train_batch_size=8, per_device_eval_batch_size=8,
                                       warmup_steps=200, weight_decay=0.01, logging_dir='./logs'
                                       )
trainer = Trainer(model=model, args=training_arguments)
model = ModelWrap(trainer, tokenizer, task, labels)

# loading dataset
dataloader = Dataset(path='./')
if dataname == 'HOC':
    x, y, label_names, rationales = dataloader.load_hoc()
elif dataname == 'HX':
    x, y, label_names, rationales = dataloader.load_hatexplain(tokenizer)
elif dataname == 'Movies' and not sentence_level:
    x, y, label_names, rationales = dataloader.load_movies(level='token')
elif dataname == 'Movies' and sentence_level:
    x, y, label_names, rationales = dataloader.load_movies(level='sentence')
elif dataname == 'Hummingbird':
    x, y, label_names, rationales = dataloader.load_hummingbird()
elif dataname == 'Ethos':
    x, y, label_names = dataloader.load_ethos()
    label_names = label_names[1:]  # Ethos
elif dataname == 'ESNLI':
    dataset, label_names = dataloader.load_esnli()
    train_texts, test_texts, test_rationales, validation_texts, train_labels, test_labels, validation_labels = dataset
elif dataname == 'AIS':
    x, y, _ = dataloader.load_AIS()
    label_names = ['no stroke', 'ischemic stroke']

if dataname != 'ESNLI':
    indices = np.arange(len(y))
    train_texts, test_texts, train_labels, test_labels, _, test_indexes = train_test_split(x, list(y), indices,
                                                                                           test_size=.2,
                                                                                           random_state=42)
    if existing_rationales:
        test_rationales = [rationales[x] for x in test_indexes]

    size = (0.1 * len(y)) / len(train_labels)
    train_texts, validation_texts, train_labels, validation_labels = train_test_split(list(train_texts), train_labels,
                                                                                      test_size=size, random_state=42)

# for movies dataset we remove extra long(>512) instances
if dataname=='Movies':
    train_texts.append(test_texts[84])
    train_labels.append(test_labels[84])
    train_texts.append(test_texts[72])
    train_labels.append(test_labels[72])
    test_texts.pop(84)
    test_labels.pop(84)
    test_rationales.pop(84)
    test_texts.pop(72)
    test_labels.pop(72)
    test_rationales.pop(72)
    test_texts.pop(63)
    test_labels.pop(63)
    test_rationales.pop(63)

# rationale preprocessing the same way as in optimus different notebooks.
# each dataset requires different processing
if existing_rationales and dataname == 'HOC':
    test_label_rationales = []
    for test_rational in test_rationales:
        label_rationales = []
        for label in range(labels):
            label_rationales.append([])
        for sentence in test_rational:
            for label in range(labels):
                if label_names[label] in sentence:
                    label_rationales[label].append(1)
                else:
                    label_rationales[label].append(0)
        test_label_rationales.append(label_rationales)
    final_rationales = test_label_rationales

elif existing_rationales and dataname == 'HX':
    test_test_rationales = []
    for test_rational in test_rationales:
        test_test_rationales.append([0, test_rational])
    final_rationales = test_test_rationales
elif existing_rationales and dataname == 'ESNLI':
    for i in range(len(test_rationales)):
        if (test_rationales[i][0] == []):
            test_rationales[i][0] = [0] * len(test_rationales[i][1])
            test_rationales[i][1] = list(test_rationales[i][1])
        else:
            test_rationales[i][1] = [0] * len(test_rationales[i][0])
            test_rationales[i][0] = list(test_rationales[i][0])
    test_test_rationales = test_rationales

    new_rationale = []
    for i in range(2000):
        rationale = []
        test_t = test_texts[i].split(' ')
        for j in range(2):
            label_rational = []
            for k in range(len(test_t)):
                for r in tokenizer.tokenize(test_t[k]):
                    # if r == '.':
                    #    print(r)
                    # print(r)
                    rationall = 1 if test_test_rationales[i][j][k] > 0 else 0
                    label_rational.append(rationall)
            rationale.append(label_rational)
        new_rationale.append(rationale)
    final_rationales = new_rationale

elif existing_rationales and dataname == 'Movies' and not sentence_level:
    test_test_rationales = []
    for i in range(len(test_rationales)):
        if (test_labels[i] == 1):
            test_test_rationales.append([[0] * len(test_rationales[i]), test_rationales[i]])
        else:
            test_test_rationales.append([test_rationales[i], [0] * len(test_rationales[i])])

    new_rationale = []
    for i in range(len(test_test_rationales)):
        rationale = []
        test_t = test_texts[i].split(' ')
        for j in range(2):
            label_rational = []
            for k in range(len(test_t)):
                for r in tokenizer.tokenize(test_t[k]):
                    # if r == '.':
                    #    print(r)
                    # print(r)
                    rationall = 1 if test_test_rationales[i][j][k] > 0 else 0
                    label_rational.append(rationall)
            rationale.append(label_rational)
        new_rationale.append(rationale)
    final_rationales = new_rationale

elif existing_rationales and dataname == 'Movies' and sentence_level:
    test_test_rationales = []
    for i in range(len(test_rationales)):
        if (test_labels[i] == 1):
            test_test_rationales.append([[0] * len(test_rationales[i][:-1]), list(test_rationales[i][:-1])])
        else:
            test_test_rationales.append([list(test_rationales[i][:-1]), [0] * len(test_rationales[i][:-1])])
    final_rationales = test_test_rationales

elif existing_rationales and dataname == 'Hummingbird':
    new_rationale = []
    for i in range(len(test_rationales)):
        rationale = []
        test_t = test_texts[i].split(' ')
        for j in range(6):
            label_rational = []
            for k in range(len(test_t)):
                for r in tokenizer.tokenize(test_t[k]):
                    rationall = 1 if test_rationales[i][j][k] > 0 else 0
                    label_rational.append(rationall)
            rationale.append(label_rational)
        new_rationale.append(rationale)

    final_rationales = new_rationale
else:
    final_rationales = test_texts

# initialize cake or cakes
if parameters.iloc[0]['Embedding model'] == 'Trained':
    cake = CAKE(label_names, 15, model_path=model_path)
else:
    cake = CAKE(label_names, 15)

# train patternrank
cake.train_vectorizer(train_texts)
# calculate document embeddings
cake.document_and_word_embed(train_texts, unique=f'{distil}Bert')

# cashe wrap the single predict function of myModel.
predictor = heavenCashe.instansheCashe(dataname, f'{distil}Bert', saverate)(model.single_predict)

train_predictions = []
for train_text in train_texts:
    outputs = predictor(train_text.encode('ascii', 'ignore').decode())
    train_predictions.append(outputs)

predictions = []
for test_text in test_texts:
    outputs = predictor(test_text)
    predictions.append(outputs)

if task == 'multi_label':
    import tensorflow as tf

    a = tf.constant(predictions, dtype=tf.float32)
    b = tf.keras.activations.sigmoid(a)
    predictions = b.numpy()

    a = tf.constant(train_predictions, dtype=tf.float32)
    b = tf.keras.activations.sigmoid(a)
    train_predictions = b.numpy()

    pred_labels = []
    for prediction in predictions:
        pred_labels.append([1 if i >= 0.5 else 0 for i in prediction])

    train_pred_labels = []
    for prediction in train_predictions:
        train_pred_labels.append([1 if i >= 0.5 else 0 for i in prediction])
    multi = True

elif task == 'single_label':
    pred_labels = []
    for prediction in predictions:
        pred_labels.append(np.argmax(softmax(prediction)))

    train_pred_labels = []
    for prediction in train_predictions:
        train_pred_labels.append(np.argmax(softmax(prediction)))

    multi = False

print(average_precision_score(test_labels, pred_labels, average='macro'),
      f1_score(test_labels, pred_labels, average='macro'))

my_explainers = MyExplainer(label_names, model, sentence_level, '‡')
my_explainers.initialize_keyphraser(cake)

ig_explainer = SequenceClassificationExplainer(trainer.model, tokenizer, custom_labels=label_names)
ig_cashe = heavenCashe.instansheCashe(dataname, f'{distil}Bert_ig', saverate, 2)(ig_explainer)
my_explainers.initiate_ig(ig_cashe)

my_evaluators = MyEvaluation(label_names, predictor, sentence_level, True)
my_evaluatorsP = MyEvaluation(label_names, predictor, sentence_level, False)

if existing_rationales:
    # evaluation =  {'F':my_evaluators.faithfulness, 'FTP': my_evaluators.faithful_truthfulness_penalty,
    #           'NZW': my_evaluators.nzw, 'AUPRC': my_evaluators.auprc}
    evaluationP = {'F': my_evaluatorsP.faithfulness, 'FTP': my_evaluatorsP.faithful_truthfulness_penalty,
                   'NZW': my_evaluatorsP.nzw, 'AUPRC': my_evaluators.auprc}
    empty_metric = {'F': [], 'FTP': [], 'NZW': [], 'AUPRC': []}
else:
    # evaluation =  {'F':my_evaluators.faithfulness, 'FTP': my_evaluators.faithful_truthfulness_penalty,
    #           'NZW': my_evaluators.nzw}
    evaluationP = {'F': my_evaluatorsP.faithfulness, 'FTP': my_evaluatorsP.faithful_truthfulness_penalty,
                   'NZW': my_evaluatorsP.nzw}
    empty_metric = {'F': [], 'FTP': [], 'NZW': []}


if not os.path.exists(f'./Results/{dataname}/'):
    os.makedirs(f'./Results/{dataname}/', exist_ok=True)

# load parameters and loop over all combinations
for i, params in enumerate(parameters.iloc):
    param = params.to_dict()
    print(param)
    cake.change_para(param['Keyphrase Number'], param['Candidate Generation'], param['Threshold'])
    cake.train_cake(np.array(train_pred_labels), multi=multi, classifier_data_model=f'{param["Ranking"]} {dataname} {distil}', calibration=param['Calibration'])

    # main loop same as in optimus for compatibility
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # metrics = deepcopy(empty_metric)
        metricsP = deepcopy(empty_metric)
        time_r = [[]]
        techniques = [my_explainers.cake_explainer]
        for ind in tqdm(range(0, len(test_texts))):
            torch.cuda.empty_cache()
            test_rational = deepcopy(final_rationales[ind])
            instance = test_texts[ind]
            if dataname == 'HOC' and existing_rationales and len(instance.split('.')) - 1 < len(test_rational[0]):
                for label in range(labels):
                    test_rational[label] = test_rational[label][:len(instance.split('.')) - 1]

            my_evaluators.clear_states()
            my_evaluatorsP.clear_states()
            prediction = predictor(instance)
            enc = model.tokenizer([instance, instance], truncation=True, padding=True)[0]
            mask = enc.attention_mask
            tokens = enc.tokens

            if not sentence_level or tokens.count('.') >= 2:
                interpretations = []
                for tech_counter, technique in enumerate(techniques):
                    ts = time.time()
                    temp = technique(instance, prediction, tokens, mask, None, None)
                    temp_tokens = tokens.copy()
                    if sentence_level:
                        temp_tokens = temp[0].copy()[0]
                        temp = temp[1].copy()
                    interpretations.append([np.array(i) / np.max(1e-15 + np.abs(np.array(i))) for i in temp])
                    time_r[tech_counter].append(time.time() - ts)
                # for metric in metrics.keys():
                #     evaluated = []
                #     for interpretation in interpretations:
                #         evaluated.append(
                #             evaluation[metric](interpretation, None, instance, prediction, temp_tokens,None, None, test_rational))
                #     metrics[metric].append(evaluated)
                #     my_evaluatorsP.saved_state = my_evaluators.saved_state.copy()
                # my_evaluators.clear_states()
                for metric in metricsP.keys():
                    evaluatedP = []
                    for interpretation in interpretations:
                        evaluatedP.append(evaluationP[metric](interpretation, None, instance, prediction, temp_tokens, None, None, test_rational))
                        # evaluatedP.append(np.where(prediction > 0.5, metrics[metric][ind][0], np.nan))
                    metricsP[metric].append(evaluatedP)


    time_r = np.array(time_r[0])
    print(f'time it took {time_r.mean()}')

    results = print_results('(P)', ['Keybert'], metricsP, label_names)

    results['time'] = time_r.mean()
    fid = cake.calculate_f1(test_texts, pred_labels)
    results['f1_fidelity'] = fid
    print(f'fidelity on test data:{fid}')
    # place missing values on dataframe
    for key, val in results.items():
        parameters.iat[i, parameters.columns.get_loc(key)] = val

# merge the new data on the old dataframe and save
df = pd.merge(parameters, old_parameters,  how='outer', on=['Dataset', 'Distil', 'Embedding model', 'Keyphrase Number', 'Threshold', 'Ranking', 'Calibration', 'Candidate Generation'])
docmetrics = ['AUPRC', 'F', 'FTP', 'NZW', 'time', 'f1_fidelity']

for i in docmetrics:
    df[f'{i}_x'].fillna(df[f'{i}_y'], inplace=True)

# drop the _x and _y suffixes
df.drop([f'{i}_y' for i in docmetrics], axis=1, inplace=True)
df.rename(columns={f'{i}_x': f'{i}' for i in docmetrics}, inplace=True)
df.to_excel('parameters.xlsx', index=False)













